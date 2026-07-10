import Foundation
import MLX
import MLXToolKit
import VACE
import WanCore

/// MLXEngine package: Wan2.1-VACE-1.3B (the all-in-one image-grounding model) exposing
/// the canonical `textToVideo` + `videoEdit` surfaces from ONE loaded pipeline. VACE is
/// unified by the VCU mechanism: every mode is just a different `(condition frames, mask)`
/// pair, so one pipeline backs them all:
///   - `textToVideo` (plain prompt) → t2v (degenerate VCU: gray frames + all-reactive mask).
///   - `textToVideo` + `initImage` → first-frame i2v (the image is frozen at frame 0).
///   - `videoEdit` (source video) → v2v (regenerate the whole clip toward the prompt).
///
/// Engine-owned lifecycle (C13): construct from `VACEConfiguration`, page the working set
/// in with `load()`, drive `run(_:)`, reclaim with `unload()`. Lifecycle is isolated to
/// `InferenceActor`; the non-`Sendable` `VACEPipeline` never crosses the boundary.
/// Cancellation is honored at every denoising-step boundary via the core's `onStep`.
///
/// The generation engine (Context-Adapter DiT forward, VCU mask-encode, FlowUniPC denoise,
/// 16-ch WanVAE decode) is parity-locked / smoke-validated against the Wan2.1 oracle.
@InferenceActor
public final class MLXVACEPackage: ModelPackage {
    public typealias Configuration = VACEConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Wan-AI/Wan2.1-VACE-1.3B",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // DERIVED, pending a live in-app re-measure (the APP-VALIDATION entry). VACE-1.3B
                // is a light tier: the fp32 Context-Adapter DiT (backbone + 15-layer branch) is
                // ~9 GB resident with the 16-ch WanVAE; umT5 is paged in per request and evicted
                // before denoise (§2.4), so the peak phase is the umT5 fp32 encode (~22 GB) — which
                // is why the 16 GB consumer target needs fp8 umT5 (the documented lever, not yet
                // wired). residentBytes here is the max-simultaneous estimate (umT5 phase + headroom);
                // re-ground on the measured phys_footprint after the live run, like TI2V did.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 30_000_000_000),  // → fp32 compute (derived)
                    QuantFootprint(quant: .int4, residentBytes: 26_000_000_000),  // DiT int4; umT5 phase dominates
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max  // pending the live re-measure (light tier — likely drops after)
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.6),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "vace-1.3b-t2v",
                    summary: "Wan2.1-VACE-1.3B all-in-one image-grounding (MLX). Prompt → video; "
                        + "add a `T2VRequest.initImage` to animate from a frozen first frame (i2v). "
                        + "832×480 native, 81 frames. The lightest Apache image-grounding tier.",
                    modes: [.quality, .fast]
                ),
                VEditContract.descriptor(
                    name: "vace-1.3b-edit",
                    summary: "Video editing via Wan2.1-VACE-1.3B (MLX): regenerate a source clip "
                        + "toward the prompt (v2v). Masked-region inpaint rides `metaData` (a "
                        + "follow-up); pose/depth control follows via Apple Vision preprocessing.",
                    modes: [.quality, .fast]
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline (Context-Adapter DiT + 16-ch WanVAE + tokenizer), paged in by
    /// `load()`. umT5 is NOT resident — paged in per request and evicted before denoise (§2.4).
    private var pipeline: VACEPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        // E15 experiment hook: `VACE_DIT_DTYPE=bf16` runs the bf16-fused attention path
        // (pair with `WAN_FP32_SDPA=0`); default fp32. No rebuild needed to flip configs.
        let ditDType: DType =
            ProcessInfo.processInfo.environment["VACE_DIT_DTYPE"] == "bf16" ? .bfloat16 : .float32
        pipeline = try await VACEPipeline.fromPretrained(modelDir: directory, ditDType: ditDType)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before the notLoaded
        // guard, capability validation, or dispatch (run-lifecycle program, engine 0.27.0).
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runVACET2V(t2v, pipeline: pipeline)
        case .videoEdit:
            guard let vedit = request as? VEditRequest else {
                throw PackageError.configurationMismatch(
                    expected: "VEditRequest", got: String(describing: type(of: request)))
            }
            return try await runVACEVideoEdit(vedit, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // Surfaces run via the shared `runVACET2V` / `runVACEVideoEdit` functions (VACESurfaces.swift),
    // reused by the dual-expert A14B package (`MLXVACEFunPackage`).
}

extension MLXVACEPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXVACEPackage.self)
    }
}

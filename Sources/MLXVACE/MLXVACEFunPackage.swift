import Foundation
import MLX
import MLXToolKit
import VACE
import WanCore

/// MLXEngine package: **Wan2.2-VACE-Fun-A14B** — the all-in-one control/editing model at the
/// QUALITY/PRO tier. Same canonical surfaces as the consumer `MLXVACEPackage` (1.3B), at A14B
/// dual-expert fidelity: `textToVideo` (+ `initImage` i2v) + `videoEdit` (v2v / inpaint), all
/// from one VCU-unified pipeline.
///
/// **Why a SEPARATE package from the 1.3B `MLXVACEPackage` (not a second config):** the model
/// CODE is shared (one `VACEPipeline`, the dual-expert switch lives inside it), but the engine
/// manifest is STATIC per package type and the two tiers have ~4× different footprints — 1.3B is
/// the consumer/16 GB tier, A14B is the pro/128 GB tier. A single manifest cannot be honest for
/// both (declare 1.3B → the governor under-reserves A14B → OOM; declare A14B → the 1.3B consumer
/// tier over-reserves and won't admit on consumer hardware). So this is its own `ModelPackage`
/// with the pro-tier footprint, and the engine's multi-package-per-capability selection picks
/// 1.3B vs A14B by specialty/memory — exactly the tier-ladder intent. (Refines the scoping doc's
/// tentative "second config" lean, which was about the shared model code.)
///
/// Engine-owned lifecycle (C13): construct from `VACEConfiguration` (default repo `.funA14B`),
/// page in with `load()` (auto-detects the dual-expert `high_noise_model/` + `low_noise_model/`
/// checkpoint layout), drive `run(_:)`, reclaim with `unload()`. Cancellation honored per step.
@InferenceActor
public final class MLXVACEFunPackage: ModelPackage {
    public typealias Configuration = VACEConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "alibaba-pai/Wan2.2-VACE-Fun-A14B",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // DERIVED from the Bernini-A14B analog (the same dual-expert A14B backbone +
                // 16-ch WanVAE + umT5), pending the live in-app re-measure (APP-VALIDATION /
                // scoping doc §5). VACE-Fun adds only the small per-expert 8-layer VACE branch on
                // top of A14B, so Bernini-A14B's measured peak is the conservative floor here:
                //   bf16 112 GB / int4 67 GB (Bernini `BerniniRPackage`). Re-ground on the measured
                // phys at the production frame count BEFORE declaring the tier — the dual-expert
                // residency + the shared umT5 floor are the walls, addressed by sequential expert
                // paging + fp8 umT5 + the `Memory.cacheLimit` decode cap + residentBytes=max(phase).
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 115_000_000_000),  // A14B + branch (derived)
                    QuantFootprint(quant: .int4, residentBytes: 70_000_000_000),   // int4 DiT; umT5 phase dominates
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [
                // Quality/pro tier — the high-fidelity counterpart to the 1.3B consumer model.
                SpecialtyWeight(.general, strength: 0.9),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "vace-fun-a14b-t2v",
                    summary: "Wan2.2-VACE-Fun-A14B all-in-one image-grounding (MLX), quality tier. "
                        + "Prompt → video; add a `T2VRequest.initImage` to animate from a frozen "
                        + "first frame (i2v). Dual-expert A14B fidelity. The pro control/editing tier.",
                    modes: [.quality, .fast]
                ),
                VEditContract.descriptor(
                    name: "vace-fun-a14b-edit",
                    summary: "Video editing via Wan2.2-VACE-Fun-A14B (MLX): regenerate a source clip "
                        + "toward the prompt (v2v) at A14B quality. Masked-region inpaint rides "
                        + "`metaData`; pose/depth control follows via Apple Vision preprocessing.",
                    modes: [.quality, .fast]
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident dual-expert pipeline (two Context-Adapter experts + 16-ch WanVAE + tokenizer),
    /// paged in by `load()`. umT5 is NOT resident — paged in per request and evicted before
    /// denoise (§2.4).
    private var pipeline: VACEPipeline?

    public nonisolated init(configuration: Configuration = .funA14B) {
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
}

extension MLXVACEFunPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXVACEFunPackage.self)
    }
}

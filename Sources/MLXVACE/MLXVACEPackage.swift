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
        pipeline = try await VACEPipeline.fromPretrained(modelDir: directory)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runT2V(t2v, pipeline: pipeline)
        case .videoEdit:
            guard let vedit = request as? VEditRequest else {
                throw PackageError.configurationMismatch(
                    expected: "VEditRequest", got: String(describing: type(of: request)))
            }
            return try await runVideoEdit(vedit, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surfaces

    private func runT2V(_ request: T2VRequest, pipeline: VACEPipeline) async throws -> T2VResponse {
        try Task.checkCancellation()
        let numFrames = request.numFrames ?? 81
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480
        let steps = resolveSteps(mode: request.mode, steps: request.steps)
        let onStep: (Int, Int, MLXArray) throws -> Void = { _, _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }

        let frames: MLXArray
        if let initImage = request.initImage {
            // i2v: the init image is the (frozen) first frame. [1,3,1,H,W] → [3,H,W].
            let image = try decodeReferencePixels(initImage, width: width, height: height)
                .squeezed(axis: 0).squeezed(axis: 1)
            frames = try pipeline.i2v(
                image: image, prompt: request.prompt, negativePrompt: request.negativePrompt,
                numFrames: numFrames, steps: steps, guideScale: request.guidanceScale,
                seed: request.seed, onStep: onStep)
        } else {
            frames = try pipeline.t2v(
                prompt: request.prompt, negativePrompt: request.negativePrompt,
                width: width, height: height, numFrames: numFrames, steps: steps,
                guideScale: request.guidanceScale, seed: request.seed, onStep: onStep)
        }
        return try await framesToVideoResponse(frames, fps: fps)
    }

    private func runVideoEdit(_ request: VEditRequest, pipeline: VACEPipeline) async throws
        -> VEditResponse
    {
        try Task.checkCancellation()
        let numFrames = request.numFrames ?? 81
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480
        let steps = resolveSteps(mode: request.mode, steps: request.steps)
        let onStep: (Int, Int, MLXArray) throws -> Void = { _, _, _ in
            try Task.checkCancellation()
        }

        // Source video → [3, T, H, W]; v2v = regenerate the whole clip (all-reactive mask).
        let video = try await decodeVideoPixels(
            request.video, width: width, height: height, numFrames: numFrames)
            .squeezed(axis: 0)  // [3, T, H, W]
        let t = video.dim(1)
        let mask = MLXArray.ones([1, t, height, width])
        let frames = try pipeline.generate(
            prompt: request.prompt, negativePrompt: request.negativePrompt,
            frames: video, mask: mask, steps: steps, guideScale: request.guidanceScale,
            seed: request.seed, onStep: onStep)
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return VEditResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }

    private func framesToVideoResponse(_ frames: MLXArray, fps: Double) async throws -> T2VResponse {
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }
}

extension MLXVACEPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXVACEPackage.self)
    }
}

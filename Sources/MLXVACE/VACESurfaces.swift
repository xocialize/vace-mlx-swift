import Foundation
import MLX
import MLXToolKit
import VACE

// Shared surface runners — the `textToVideo` / `videoEdit` request → response logic for the
// VACE pipeline, factored out so BOTH engine packages reuse them: `MLXVACEPackage` (the
// consumer 1.3B tier) and `MLXVACEFunPackage` (the quality/pro dual-expert A14B tier). The
// pipeline is identical across tiers (the dual-expert switch lives inside `VACEPipeline`); only
// the package manifest (footprint/tier/surfaces) differs, which is why the surface logic is
// shared and the packages are thin wrappers around it.
//
// `@InferenceActor` to match the packages: `VACEPipeline` is non-Sendable and only ever touched
// on that actor (C13 — the engine isolates the lifecycle there).

@InferenceActor
func runVACET2V(_ request: T2VRequest, pipeline: VACEPipeline) async throws -> T2VResponse {
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
    // Post-core checkpoint: the streaming VAE decode bails per temporal chunk
    // (non-throwing) — discard a truncated result and rethrow here.
    try Task.checkCancellation()
    return try await framesToVACEVideoResponse(frames, fps: fps)
}

@InferenceActor
func runVACEVideoEdit(_ request: VEditRequest, pipeline: VACEPipeline) async throws -> VEditResponse {
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
    // Post-core checkpoint: the streaming VAE decode bails per temporal chunk
    // (non-throwing) — discard a truncated result and rethrow here. (The VCU build's
    // encodeStreaming also bails per chunk; the pipeline throws through onStep.)
    try Task.checkCancellation()
    let mp4 = try await encodeMP4(frames: frames, fps: fps)
    return VEditResponse(
        video: Video(format: .mp4, data: mp4,
                     durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
}

@InferenceActor
func framesToVACEVideoResponse(_ frames: MLXArray, fps: Double) async throws -> T2VResponse {
    let mp4 = try await encodeMP4(frames: frames, fps: fps)
    return T2VResponse(
        video: Video(format: .mp4, data: mp4,
                     durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
}

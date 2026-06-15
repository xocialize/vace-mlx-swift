import Foundation
import MLX
import WanCore

/// VACE single-expert denoise loop. Mirrors the parity-locked TI2V `denoiseTI2V` (same FlowUniPC
/// scheduler, same CFG `uncond + gs·(cond−uncond)` wiring) — the only difference is the forward is
/// `VaceWanModel(vaceContext:)`, so the structural conditioning rides the VCU (NOT per-token
/// timesteps). Correct by construction: the forward is bit-exact (`VaceBranchParityTests`), the
/// scheduler is wan-core's (parity-locked via TI2V), and the loop wiring is identical to the
/// parity-locked TI2V loop. The same VCU feeds both CFG arms; only the text context differs.
public func denoiseVACE(
    model: VaceWanModel,
    config: WanConfig,
    contextCond: MLXArray,      // raw umT5 features [L, text_dim] (positive)
    contextNull: MLXArray,      // raw umT5 features [L, text_dim] (negative/uncond)
    vaceContext: MLXArray,      // the VCU [96, tLat, hLat, wLat]
    noise: MLXArray,            // [C, tLat, hLat, wLat]
    steps: Int,
    shift: Double,
    guideScale: Double,
    vaceContextScale: Float = 1.0,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let cfg = guideScale > 1.0

    // Pre-embed the text context (the DiT's text MLP); CFG → [cond, uncond] (B=2).
    let contextCfg = model.embedText(cfg ? [contextCond, contextNull] : [contextCond])
    eval(contextCfg)

    let (tLat, hLat, wLat) = (noise.dim(1), noise.dim(2), noise.dim(3))
    let ps = config.patchSize
    let seqLen = (tLat / ps[0]) * (hLat / ps[1]) * (wLat / ps[2])

    let sched = FlowUniPCScheduler(numTrainTimesteps: config.numTrainTimesteps)
    sched.setTimesteps(steps, shift: shift)
    let timesteps = sched.timesteps

    var latents = noise
    // Cap the buffer cache during the denoise — VACE's HEAVY phase (seqLen ≫ wanLargeSeq ⇒ fp32
    // SDPA in every block). The per-block eval bounds the live graph and the per-step `clearCache`
    // reclaims at step boundaries, but only `Memory.cacheLimit` bounds the IN-step high-water:
    // freed block transients otherwise accumulate to ~the peak (the E15 ~105 GB plateau). This is
    // the TI2V decode lever (ti2v `c98cdc6`) applied to VACE's actual heavy phase. Env
    // `DENOISE_CACHE_MB` overrides (0 = max reclaim); restored after the loop.
    let prevCacheLimit = Memory.cacheLimit
    let capMB = ProcessInfo.processInfo.environment["DENOISE_CACHE_MB"].flatMap { Int($0) } ?? 2048
    Memory.cacheLimit = capMB * 1_000_000
    defer { Memory.cacheLimit = prevCacheLimit }
    for i in 0..<steps {
        let t = Float(timesteps[i])
        let noisePred: MLXArray
        if cfg {
            let preds = model(
                [latents, latents], t: MLXArray([t, t]), context: .embedded(contextCfg),
                seqLen: seqLen, vaceContext: [vaceContext, vaceContext], vaceContextScale: vaceContextScale)
            noisePred = preds[1] + Float(guideScale) * (preds[0] - preds[1])
        } else {
            let preds = model(
                [latents], t: MLXArray([t]), context: .embedded(contextCfg),
                seqLen: seqLen, vaceContext: [vaceContext], vaceContextScale: vaceContextScale)
            noisePred = preds[0]
        }
        let stepped = sched.step(
            modelOutput: noisePred.expandedDimensions(axis: 0), timestep: t,
            sample: latents.expandedDimensions(axis: 0))
        latents = stepped.squeezed(axis: 0)
        eval(latents)
        MLX.GPU.clearCache()  // per-step buffer-cache discipline
        try onStep?(i, steps, latents)
    }
    return latents
}

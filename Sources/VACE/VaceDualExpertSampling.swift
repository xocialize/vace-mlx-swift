import Foundation
import MLX
import WanCore

/// Dual-expert VACE denoise loop (Wan2.2-VACE-Fun-A14B). The marginal net-new of the
/// VACE-Fun-A14B port: it is exactly the single-expert `denoiseVACE` (VCU / Context-Adapter
/// hint injection) with Bernini-A14B's timestep-boundary expert switch dropped in.
///
/// Each expert is a complete `VaceWanModel` (backbone + its OWN 8-layer VACE branch — the
/// branch ships per-expert in the released checkpoint, §3 of `ENH-vace-fun-a14b.md`), so the
/// switch carries the branch with the expert it belongs to — no shared-branch interleave.
/// The structural conditioning (the VCU) is the SAME for both experts; only the backbone +
/// branch weights (and the per-expert text MLP / guide scale) differ across the boundary.
///
/// Correct by construction: the per-expert forward is the parity-locked `VaceWanModel`
/// forward (`VaceBranchParityTests`); the scheduler is wan-core's FlowUniPC (parity-locked via
/// TI2V); the boundary switch is Bernini-A14B's validated `denoiseT2V` logic, indexing-for-
/// indexing (`isHigh ? high : low`, guide scale `isHigh ? gs.1 : gs.0`).
public func denoiseVACEDualExpert(
    high: VaceWanModel,
    low: VaceWanModel,
    config: WanConfig,
    contextCond: MLXArray,      // raw umT5 features [L, text_dim] (positive)
    contextNull: MLXArray,      // raw umT5 features [L, text_dim] (negative/uncond)
    vaceContext: MLXArray?,     // the VCU [96, tLat, hLat, wLat]; nil = no control (base t2v)
    noise: MLXArray,            // [C, tLat, hLat, wLat]
    steps: Int,
    shift: Double,
    guideScaleOverride: Double? = nil,  // a per-request scalar → applied to BOTH phases
    vaceContextScale: Float = 1.0,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    // Per-expert guide scale, resolved exactly as Bernini's `T2VOptions.fromConfig`:
    // config.sampleGuideScale is [belowBoundary, atOrAboveBoundary]; a scalar config
    // (single-expert) reuses the one value for both phases. A per-request scalar override
    // (the engine surface carries one `Double`) applies to both phases. CFG engages when > 1.
    let gs = config.sampleGuideScale
    let gsLow = guideScaleOverride ?? gs[0]                          // below boundary (low-noise phase)
    let gsHigh = guideScaleOverride ?? (gs.count > 1 ? gs[1] : gs[0])  // at/above boundary (high-noise phase)
    let cfg = max(gsLow, gsHigh) > 1.0

    // Pre-embed the text context per expert (each expert has its OWN text MLP — Bernini
    // embeds high/low separately). CFG → [cond, uncond] (B=2); CFG-free → [cond] (B=1).
    let textIn = cfg ? [contextCond, contextNull] : [contextCond]
    let contextCfgHigh = high.embedText(textIn)
    let contextCfgLow = low.embedText(textIn)
    eval(contextCfgHigh, contextCfgLow)

    let (tLat, hLat, wLat) = (noise.dim(1), noise.dim(2), noise.dim(3))
    let ps = config.patchSize
    let seqLen = (tLat / ps[0]) * (hLat / ps[1]) * (wLat / ps[2])

    let sched = FlowUniPCScheduler(numTrainTimesteps: config.numTrainTimesteps)
    sched.setTimesteps(steps, shift: shift)
    let timesteps = sched.timesteps
    let boundary = config.boundaryTimestep

    var latents = noise
    // Cap the buffer cache during the denoise — VACE-Fun's HEAVY phase (seqLen ≫ wanLargeSeq
    // ⇒ fp32 SDPA in every block, per expert). Matches the single-expert `denoiseVACE`. Env
    // `DENOISE_CACHE_MB` overrides (0 = max reclaim); restored after the loop.
    let prevCacheLimit = Memory.cacheLimit
    let capMB = ProcessInfo.processInfo.environment["DENOISE_CACHE_MB"].flatMap { Int($0) } ?? 2048
    Memory.cacheLimit = capMB * 1_000_000
    defer { Memory.cacheLimit = prevCacheLimit }
    let stepNote = "seqLen=\(seqLen) cfg=\(cfg ? 2 : 1) dual"

    for i in 0..<steps {
        let t = Float(timesteps[i])
        let isHigh = Double(timesteps[i]) >= boundary
        let model = isHigh ? high : low
        let ctx = isHigh ? contextCfgHigh : contextCfgLow
        let guideScale = isHigh ? gsHigh : gsLow

        WanProfiler.shared.region("denoise", "step", index: i, note: stepNote) {
            let noisePred: MLXArray
            if cfg {
                let preds = model(
                    [latents, latents], t: MLXArray([t, t]), context: .embedded(ctx),
                    seqLen: seqLen, vaceContext: vaceContext.map { [$0, $0] },
                    vaceContextScale: vaceContextScale)
                noisePred = preds[1] + Float(guideScale) * (preds[0] - preds[1])
            } else {
                let preds = model(
                    [latents], t: MLXArray([t]), context: .embedded(ctx),
                    seqLen: seqLen, vaceContext: vaceContext.map { [$0] },
                    vaceContextScale: vaceContextScale)
                noisePred = preds[0]
            }
            let stepped = sched.step(
                modelOutput: noisePred.expandedDimensions(axis: 0), timestep: t,
                sample: latents.expandedDimensions(axis: 0))
            latents = stepped.squeezed(axis: 0)
            eval(latents)
        }
        MLX.Memory.clearCache()  // per-step buffer-cache discipline
        vaceMemLog("denoise step \(i + 1)/\(steps) (\(isHigh ? "high" : "low"))")
        WanDebug.stats("denoise step \(i + 1)/\(steps)", latents)
        try onStep?(i, steps, latents)
    }
    return latents
}

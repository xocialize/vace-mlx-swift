import MLX
import MLXNN
import WanCore

/// Wan2.1-VACE-1.3B DiT. Subclasses wan-core `WanModel` (so the backbone keys are inherited
/// unchanged) and adds the VACE Context-Adapter branch: `vace_patch_embedding_proj` + `vace_blocks`
/// (15 `VaceWanAttentionBlock`s). The branch runs on the embedded latent + the same per-block kwargs,
/// and its hints inject through wan-core's `runBlocks(blockResiduals:)` seam.
/// Oracle: `wan/modules/vace_model.py` `VaceWanModel`.
public final class VaceWanModel: WanModel {
    @ModuleInfo(key: "vace_patch_embedding_proj") public var vacePatchEmbeddingProj: Linear
    @ModuleInfo(key: "vace_blocks") public var vaceBlocks: [VaceWanAttentionBlock]

    /// Main-block indices that receive a hint (config `vace_layers`, e.g. `[0,2,…,28]`).
    public let vaceLayers: [Int]
    private let vacePatchSize: [Int]

    public init(config: WanConfig, vaceLayers: [Int], vaceInDim: Int) {
        let patchProduct = config.patchSize.reduce(1, *)  // pt·ph·pw = 1·2·2 = 4
        self._vacePatchEmbeddingProj.wrappedValue = Linear(vaceInDim * patchProduct, config.dim)
        self._vaceBlocks.wrappedValue = (0..<vaceLayers.count).map { i in
            VaceWanAttentionBlock(
                dim: config.dim, ffnDim: config.ffnDim, numHeads: config.numHeads,
                crossAttnNorm: config.crossAttnNorm, eps: Float(config.eps), hasBeforeProj: i == 0)
        }
        self.vaceLayers = vaceLayers
        self.vacePatchSize = config.patchSize
        super.init(config)
    }

    /// Patch-embed the 96-ch VCU via `vace_patch_embedding_proj` — the same patch-extraction reshape
    /// as the backbone `patchify`, but the VACE projection. `v`: `[vaceInDim, F, H, W]` → `[1, L, dim]`.
    /// (Single-batch / no-pad path — matches the parity fixture; production pads to seqLen.)
    public func vacePatchify(_ v: MLXArray) -> MLXArray {
        let (c, f, h, w) = (v.dim(0), v.dim(1), v.dim(2), v.dim(3))
        let (pt, ph, pw) = (vacePatchSize[0], vacePatchSize[1], vacePatchSize[2])
        let (fOut, hOut, wOut) = (f / pt, h / ph, w / pw)
        var x = v.reshaped(c, fOut, pt, hOut, ph, wOut, pw)
        x = x.transposed(1, 3, 5, 0, 2, 4, 6)
        x = x.reshaped(fOut * hOut * wOut, -1)
        return vacePatchEmbeddingProj(x).expandedDimensions(axis: 0)  // [1, L, dim]
    }

    /// Run the vace branch → the per-main-block residual hints (already scaled).
    public func vaceHints(
        _ state: ForwardState, vaceContext: [MLXArray], scale: Float
    ) -> [Int: MLXArray] {
        var c = vacePatchify(vaceContext[0])
        var residuals: [Int: MLXArray] = [:]
        // At large seqLen the per-block self-attention runs in fp32 (see `wanLargeSeq`),
        // so the 15-block branch + its accumulated hints would build ONE unbounded fp32
        // lazy graph and materialize all at once — the E15 pathology (107 GB / 44 min @
        // 480p). `eval` each block to BOUND the graph, exactly as `WanModel.runBlocks`
        // does for the main blocks; only the small (≤15) concrete hint tensors stay live.
        let evalEachBlock = state.x.dim(1) >= wanLargeSeq
        for (k, block) in vaceBlocks.enumerated() {
            let (newC, hint) = block.vaceForward(
                c, x: state.x, e: state.e0, seqLens: state.seqLensList,
                gridSizes: state.gridSizes, freqs: freqs, context: state.contextBatch,
                attnMask: state.attnMask)
            c = newC
            let r = hint * scale
            residuals[vaceLayers[k]] = r
            if evalEachBlock { eval(c, r) }
        }
        return residuals
    }

    /// Full VACE forward: `embed` → vace-branch hints → `runBlocks` (inject) → `finish`.
    public func callAsFunction(
        _ x: [MLXArray], t: MLXArray, context: WanTextContext, seqLen: Int,
        vaceContext: [MLXArray], vaceContextScale: Float = 1.0
    ) -> [MLXArray] {
        var state = embed(x, t: t, context: context, seqLen: seqLen)
        let residuals = vaceHints(state, vaceContext: vaceContext, scale: vaceContextScale)
        runBlocks(&state, blockResiduals: residuals)
        return finish(state)
    }
}

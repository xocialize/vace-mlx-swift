import MLX
import MLXNN
import WanCore

/// A VACE Context-Adapter block: a wan-core `WanAttentionBlock` + `before_proj` (block-0 only,
/// zero-init in the checkpoint) + `after_proj` (all blocks, zero-init). Subclassing keeps the keys
/// flat (`self_attn`/`cross_attn`/`ffn`/`modulation`/`norm3`/`before_proj`/`after_proj`), matching
/// the converted `vace_blocks.N.*`. Oracle: `wan/modules/vace_model.py` `VaceWanAttentionBlock`.
public final class VaceWanAttentionBlock: WanAttentionBlock {
    @ModuleInfo(key: "before_proj") public var beforeProj: Linear?
    @ModuleInfo(key: "after_proj") public var afterProj: Linear

    public init(
        dim: Int, ffnDim: Int, numHeads: Int, crossAttnNorm: Bool, eps: Float, hasBeforeProj: Bool
    ) {
        self._beforeProj.wrappedValue = hasBeforeProj ? Linear(dim, dim) : nil
        self._afterProj.wrappedValue = Linear(dim, dim)
        super.init(
            dim: dim, ffnDim: ffnDim, numHeads: numHeads, qkNorm: true,
            crossAttnNorm: crossAttnNorm, eps: eps)
    }

    /// `forward(c, x)`: block-0 mixes the embedded main latent (`c = before_proj(c) + x`), runs the
    /// `WanAttentionBlock`, and returns `(c, hint = after_proj(c))`.
    public func vaceForward(
        _ c0: MLXArray, x: MLXArray, e: MLXArray, seqLens: [Int],
        gridSizes: [(Int, Int, Int)], freqs: MLXArray, context: MLXArray, attnMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        var c = c0
        if let beforeProj { c = beforeProj(c) + x }
        c = super.callAsFunction(
            c, e: e, seqLens: seqLens, gridSizes: gridSizes, freqs: freqs,
            context: context, contextLens: nil, crossKVCache: nil, ropeCosSin: nil, attnMask: attnMask)
        return (c, afterProj(c))
    }
}

import MLX
import WanCore

/// Builds the VACE "Video Condition Unit" (VCU) — the `vace_in_dim`-channel context fed to
/// `vace_patch_embedding`. VCU = `cat(z0, mask64)` where `z0 = cat(enc(inactive), enc(reactive))`
/// (32 ch for the 16-ch WanVAE) and `mask64` is the pixel mask reshaped to 64 latent-res channels
/// (`16 + 16 + 64 = 96`). Oracle: `wan/modules/wan_vace.py` `vace_encode_masks` / `vace_latent`.
public enum VaceVCU {
    /// Pixel mask `[1, depth, H, W]` → 64-ch latent-res space-to-depth `[64, newDepth, H/s, W/s]`.
    /// Verbatim port of `vace_encode_masks` (vae_stride `[4, 8, 8]`).
    public static func maskSpaceToDepth(_ mask: MLXArray, vaeStride: [Int] = [4, 8, 8]) -> MLXArray {
        let depth = mask.dim(1), h = mask.dim(2), w = mask.dim(3)
        let s = vaeStride[1]  // 8 (assumes vaeStride[1] == vaeStride[2])
        let hOut = 2 * (h / (s * 2))  // = h/8 when divisible
        let wOut = 2 * (w / (s * 2))
        let newDepth = (depth + 3) / vaeStride[0]

        var m = mask[0]                              // [depth, H, W]
        m = m.reshaped(depth, hOut, s, wOut, s)      // [depth, hOut, 8, wOut, 8]
        m = m.transposed(2, 4, 0, 1, 3)              // [8, 8, depth, hOut, wOut]
        m = m.reshaped(s * s, depth, hOut, wOut)     // [64, depth, hOut, wOut]

        // Temporal nearest-exact along the frame axis (no-op when newDepth == depth, e.g. single frame).
        if newDepth != depth {
            let idx = (0..<newDepth).map { i -> Int in
                let v = Int(((Double(i) + 0.5) * Double(depth) / Double(newDepth) - 0.5).rounded())
                return Swift.max(0, Swift.min(depth - 1, v))
            }
            m = m.take(MLXArray(idx.map { Int32($0) }), axis: 1)  // [64, newDepth, hOut, wOut]
        }
        return m
    }

    /// Encode condition frames to the 32-ch `z0` latent — verbatim `vace_encode_frames` (the
    /// masked branch). Splits the frames into the inactive (`frames·(1−mask)`) and reactive
    /// (`frames·mask`) videos, VAE-encodes each (the 16-ch WanVAE, parity-locked), and concats
    /// → `[32, Tl, Hl, Wl]`. `frames`: `[3, T, H, W]` in [-1, 1] (channels-first); `mask`:
    /// `[1, T, H, W]` in {0,1}. The mask is thresholded at 0.5 like the oracle.
    public static func encodeFrames(vae: WanVAE, frames: MLXArray, mask: MLXArray) -> MLXArray {
        let m = MLX.where(mask .> 0.5, MLXArray(Float(1)), MLXArray(Float(0)))  // [1, T, H, W]
        let inactive = frames * (1 - m)                  // [3, T, H, W] (broadcast over channels)
        let reactive = frames * m
        // E15 Addendum-6: this is the ~106 GB / ~23-min phase — two full-res fp32 VAE encodes that,
        // un-eval'd, build one giant graph (the lazy-graph signature in a path the denoise per-block
        // eval never covered). `eval` + `clearCache` each encode so the inactive working set frees
        // before the reactive one (the pipeline also caps the cache around this whole block).
        // `encodeStreaming` (the decodeStreaming analog, wan-core) evals each temporal chunk's
        // full-res forward before the next, so the live working set is ONE chunk — not the whole
        // sequence — and the cap no longer thrashes a multi-chunk graph (E15 Addendum 8: the cure
        // for the ~41 GB live + glacial encode). Bit-identical to `vae.encode`. clearCache between
        // the two so inactive fully releases before reactive.
        vaceMemLog("VCU: pre inactive-encode")
        let zIn = encodeStreaming(vae: vae, inactive.expandedDimensions(axis: 0))[0]  // [16, Tl, Hl, Wl]
        eval(zIn); MLX.GPU.clearCache()
        vaceMemLog("VCU: inactive encoded")
        let zRe = encodeStreaming(vae: vae, reactive.expandedDimensions(axis: 0))[0]  // [16, Tl, Hl, Wl]
        eval(zRe); MLX.GPU.clearCache()
        vaceMemLog("VCU: reactive encoded")
        return concatenated([zIn, zRe], axis: 0)         // [32, Tl, Hl, Wl]
    }

    /// VCU = `cat(z0[32], mask64[64]) = [96, newDepth, hLat, wLat]`. `z0` is the VAE-encoded
    /// inactive⊕reactive latent (channels-first), `mask` the pixel 0/1 mask.
    public static func build(z0: MLXArray, mask: MLXArray, vaeStride: [Int] = [4, 8, 8]) -> MLXArray {
        concatenated([z0, maskSpaceToDepth(mask, vaeStride: vaeStride)], axis: 0)
    }

    /// Full VCU from condition frames + pixel mask: `encodeFrames → build`. Both halves are
    /// parity-locked (WanVAE.encode 3e-6, mask space-to-depth <1e-6), so the composite is
    /// correct by construction. Returns `[96, Tl, Hl, Wl]`.
    public static func buildVCU(
        vae: WanVAE, frames: MLXArray, mask: MLXArray, vaeStride: [Int] = [4, 8, 8]
    ) -> MLXArray {
        build(z0: encodeFrames(vae: vae, frames: frames, mask: mask), mask: mask, vaeStride: vaeStride)
    }
}

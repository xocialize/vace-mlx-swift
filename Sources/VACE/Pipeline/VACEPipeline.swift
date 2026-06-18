// VACEPipeline — the all-in-one entry for Wan2.1-VACE-1.3B. Owns the component
// loads (the VACE Context-Adapter DiT + 16-ch WanVAE + umT5 tokenizer) and the
// (prompt + condition frames + mask) → frames path. VACE is unified by the VCU
// mechanism: every mode (inpaint / i2v / flf2v / v2v / depth-pose control) is just
// a different way to construct the (frames, mask) pair — the pipeline body is the
// same. umT5 is paged in per request and evicted before denoise (the §2.4 lever) —
// the consumer-tier (16 GB) memory recipe; the 1.3B DiT + light 16-ch VAE stay
// resident (no halo-tiling needed at this tier).
//
// The whole body composes parity-locked pieces: VaceVCU.buildVCU (WanVAE.encode
// 3e-6 + mask space-to-depth <1e-6), denoiseVACE (bit-exact forward + wan-core
// FlowUniPC), WanVAE.decode. Correct by construction.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import WanCore

/// Per-phase memory breakdown (E15 Addendum-4 root-cause): logs MLX's `active` vs `cache` vs
/// `peak` + the live `cacheLimit` at each pipeline phase, to resolve WHAT the ~106 GB floor is
/// and WHEN it appears — the floor is dtype- AND seqLen-invariant, so it's not weights/activations/
/// attention. `active ≈ 100` ⇒ a live/fixed allocation (chase what). `cache ≈ 100` with
/// `cacheLimit` small ⇒ the cap isn't trimming this phase (chase why). Opt-in via `VACE_MEM_LOG=1`.
func vaceMemLog(_ phase: String) {
    guard ProcessInfo.processInfo.environment["VACE_MEM_LOG"] != nil else { return }
    let s = Memory.snapshot()
    func gb(_ b: Int) -> String { String(format: "%.1f", Double(b) / 1e9) }
    print("[VACE mem] \(phase): active=\(gb(s.activeMemory)) cache=\(gb(s.cacheMemory)) "
        + "peak=\(gb(s.peakMemory)) cacheLimit=\(gb(Memory.cacheLimit)) (GB)")
}

public final class VACEPipeline: @unchecked Sendable {
    public let config: WanConfig
    /// The VACE Context-Adapter DiT (`VaceWanModel`): the wan-core backbone + the
    /// 15-layer parallel branch. Held resident (1.3B fp32 ≈ 8 GB — fits the tier).
    public let model: VaceWanModel
    /// 16-ch WanVAE — encode (condition frames → z0) and decode (latent → frames).
    public let vae: WanVAE
    /// Checkpoint dir — kept so umT5 can be (re)loaded per request and evicted
    /// before denoise (§2.4), rather than held resident.
    public let modelDir: URL
    public let tokenizer: any Tokenizer
    public let vaceLayers: [Int]

    public init(
        config: WanConfig, model: VaceWanModel, vae: WanVAE,
        modelDir: URL, tokenizer: any Tokenizer, vaceLayers: [Int]
    ) {
        self.config = config
        self.model = model
        self.vae = vae
        self.modelDir = modelDir
        self.tokenizer = tokenizer
        self.vaceLayers = vaceLayers
    }

    /// Load all components from a converted checkpoint directory (flat layout:
    /// `model.safetensors` (backbone + vace_* branch) + `vae.safetensors` +
    /// `t5_encoder.safetensors` + `config.json`). Tokenizer from google/umt5-xxl.
    /// - ditDType: DiT compute precision. Defaults to **fp32** — the video-scale
    ///   (large-seqLen) correctness path (Metal bf16 attention is unstable over long
    ///   sequences); the converted weights are fp32. Ignored for quantized checkpoints.
    public static func fromPretrained(
        modelDir: URL, ditDType: DType = .float32
    ) async throws -> VACEPipeline {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        // vace_layers = every other backbone layer ([0,2,…,28] for 30L = 15 injection points);
        // vace_in_dim 96 = the VCU width. (Not in the reused backbone config.json — derived,
        // matching the oracle + the parity tests.)
        let vaceLayers = Array(stride(from: 0, to: config.numLayers, by: 2))

        // Self-certifying config line (E15): prove from the console WHICH path this process
        // actually took — "set in the scheme" ≠ "engaged in this run". Covers the experiment
        // knobs (DiT dtype + the fp32-SDPA upcast) and the memory caps. The runtime analog of
        // the artifact/label check, for config the binary can't reveal statically.
        let env = ProcessInfo.processInfo.environment
        print("[VACE config] ditDType=\(ditDType) "
            + "WAN_FP32_SDPA=\(wanForceFp32SdpaLargeSeq ? 1 : 0) (wanLargeSeq=\(wanLargeSeq)) "
            + "DENOISE_CACHE_MB=\(env["DENOISE_CACHE_MB"] ?? "2048") "
            + "DECODE_CACHE_MB=\(env["DECODE_CACHE_MB"] ?? "2048")")

        let model = try loadDiT(
            modelDir: modelDir, config: config, vaceLayers: vaceLayers, ditDType: ditDType)
        vaceMemLog("DiT loaded")

        // 16-ch WanVAE (encoder + decoder), fp32 on the CPU stream (parity + watchdog).
        let vae = WanVAE(zDim: config.vaeZDim, encoder: true)
        let vaeWeights = try Device.withDefaultDevice(.cpu) {
            let loaded = try MLX.loadArrays(
                url: modelDir.appendingPathComponent("vae.safetensors"))
            WeightLoader.materialize(loaded)
            return loaded
        }
        try vae.update(
            parameters: ModuleParameters.unflattened(vaeWeights), verify: [.noUnusedKeys])

        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return VACEPipeline(
            config: config, model: model, vae: vae,
            modelDir: modelDir, tokenizer: tokenizer, vaceLayers: vaceLayers)
    }

    /// Build + load the VACE DiT (fp32 compute for video-scale correctness). Drops a
    /// stray `freqs` table if present (the precomputed RoPE table is rebuilt in-model).
    static func loadDiT(
        modelDir: URL, config: WanConfig, vaceLayers: [Int], ditDType: DType
    ) throws -> VaceWanModel {
        let model = VaceWanModel(config: config, vaceLayers: vaceLayers, vaceInDim: 96)
        var weights = try WeightLoader.loadSafetensors(
            url: modelDir.appendingPathComponent("model.safetensors"))
        weights = weights.filter { $0.key != "freqs" }
        // Cast to the requested compute dtype (fp32 default; .bfloat16 for the E15 bf16-fused
        // experiment — mirrors mlx-video, which runs the DiT in bf16 with fp32-internal softmax).
        weights = weights.mapValues { $0.asType(ditDType) }
        WeightLoader.materialize(weights)
        try model.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        eval(model.parameters())
        return model
    }

    // MARK: - umT5 (§2.4 post-encode eviction)

    private func loadTextEncoder() throws -> UMT5EncoderModel {
        let textEncoder = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(t5Weights)
        try textEncoder.update(
            parameters: ModuleParameters.unflattened(t5Weights), verify: [.noUnusedKeys])
        return textEncoder
    }

    /// §2.4: load umT5, run `body` to produce its text contexts, drop the encoder and
    /// reclaim its working set before returning. ⚠️ `body` MUST `eval` everything it returns.
    func withTextEncoder<R>(_ body: (UMT5EncoderModel) throws -> R) throws -> R {
        var encoder: UMT5EncoderModel? = try loadTextEncoder()
        let result = try body(encoder!)
        encoder = nil
        MLX.Memory.clearCache()
        return result
    }

    // MARK: - Generation

    /// The universal VACE entry: structural conditioning rides the VCU built from the
    /// condition `frames` + `mask`; the mode (inpaint / i2v / flf2v / control) is purely
    /// how the caller constructs that pair. Relay: umT5 encode→evict → VCU build (VAE
    /// encode) → VACE denoise → VAE decode. Returns frames `[1, 3, T', H', W']` in [-1, 1].
    /// - frames: condition video `[3, T, H, W]` in [-1, 1] (channels-first).
    /// - mask: per-pixel 0/1 `[1, T, H, W]` (1 = regenerate/reactive, 0 = keep/inactive).
    public func generate(
        prompt: String,
        negativePrompt: String? = nil,
        frames: MLXArray,
        mask: MLXArray,
        steps: Int? = nil,
        guideScale: Double? = nil,
        vaceContextScale: Float = 1.0,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt
        vaceMemLog("generate.start (model+vae resident)")

        // §2.4: page umT5 in, encode cond/uncond, evict before denoise.
        let (contextCond, contextNull) = try WanProfiler.shared.region("phase", "text_encode") {
            try withTextEncoder { enc -> (MLXArray, MLXArray) in
                vaceMemLog("umT5 loaded (pre-encode)")
                let c = encodeText(
                    encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
                let n = encodeText(
                    encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
                eval(c, n)
                return (c, n)
            }
        }
        vaceMemLog("umT5 evicted (post-encode)")

        // Build the VCU (fp32 VAE encode). E15 Addendum-6: THIS is the ~106 GB / ~23-min phase —
        // two full-res fp32 VAE encodes of the condition frames. Cap the buffer cache here too (the
        // TI2V decode lever, ti2v `c98cdc6`): this phase runs BEFORE `denoiseVACE`, so the denoise cap
        // never reached it, and the freed full-res conv intermediates otherwise accumulate to the box
        // ceiling. Env `VCU_CACHE_MB` (default 2048).
        // ENCODE_DEVICE=gpu runs the streaming VAE *encode* on the GPU stream — the encode-side twin of
        // DECODE_DEVICE. Default stays .cpu (fp32 parity / cold-load-watchdog avoidance). `encodeStreaming`
        // evals per temporal chunk, so per-chunk command buffers are short and the whole-seq watchdog
        // risk is bounded (same proof as the decode GPU fix). Unblocks i2v/v2v from the single-core CPU
        // encode wall (i2v 49f sat ~12 min in this phase at ~107% CPU / GPU idle — W7).
        vaceMemLog("VCU build: entry")
        WanDebug.stats("vcu frames (pre-encode)", frames)
        let encodeDevice: Device = (ProcessInfo.processInfo.environment["ENCODE_DEVICE"] == "gpu") ? .gpu : .cpu
        let vcu = WanProfiler.shared.region("phase", "vcu_build") {
            Device.withDefaultDevice(encodeDevice) { () -> MLXArray in
                let prevCacheLimit = Memory.cacheLimit
                let capMB = ProcessInfo.processInfo.environment["VCU_CACHE_MB"].flatMap { Int($0) } ?? 2048
                Memory.cacheLimit = capMB * 1_000_000
                defer { Memory.cacheLimit = prevCacheLimit }
                MLX.Memory.clearCache()
                let v = VaceVCU.buildVCU(vae: vae, frames: frames, mask: mask)
                eval(v)
                return v
            }
        }
        vaceMemLog("VCU built (pre-denoise)")
        WanDebug.stats("vcu (post-encode)", vcu)
        // Noise matches the VCU's latent geometry: [zDim, Tl, Hl, Wl].
        let (tLat, hLat, wLat) = (vcu.dim(1), vcu.dim(2), vcu.dim(3))
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        let latent = try WanProfiler.shared.region("phase", "denoise") {
            let l = try denoiseVACE(
                model: model, config: config, contextCond: contextCond, contextNull: contextNull,
                vaceContext: vcu, noise: noise, steps: steps ?? config.sampleSteps,
                shift: config.sampleShift, guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
                vaceContextScale: vaceContextScale, onStep: onStep)
            eval(l)
            return l
        }
        vaceMemLog("denoise done (pre-clearCache)")
        MLX.Memory.clearCache()  // drop the denoise working set before the decode
        vaceMemLog("post-denoise clearCache (pre-decode)")
        WanDebug.stats("latent (pre-decode)", latent)

        let frames = WanProfiler.shared.region("phase", "decode") { decodeLatent(latent) }
        WanDebug.stats("frames (post-decode)", frames)
        vaceMemLog("decode done")
        return frames
    }

    /// Decode a channels-first DiT latent `[C, Tl, Hl, Wl]` → frames `[1, 3, T', H', W']`
    /// in [-1, 1]. The 16-ch WanVAE decode runs on the CPU stream (fp32 parity / watchdog).
    ///
    /// **Streaming** (`decodeStreaming`, one latent chunk at a time): E15 Addendum-12 measured the
    /// whole-seq decode holding all frames live as the t2v residual — ~40 min + the entire 41→92 GB
    /// climb. Streaming bounds the live set to one chunk (runtime + memory stop scaling with frames),
    /// bit-identical. Plus the `Memory.cacheLimit` cap (ti2v `c98cdc6`) so freed full-res conv
    /// intermediates reclaim continuously instead of accumulating to the phys high-water. Env
    /// `DECODE_CACHE_MB` overrides (0 = max reclaim). (A chunked decode is also the safe GPU
    /// candidate later — short per-chunk command buffers dodge the whole-seq watchdog-resubmit risk.)
    public func decodeLatent(_ latent: MLXArray) -> MLXArray {
        let prevCacheLimit = Memory.cacheLimit
        let capMB = ProcessInfo.processInfo.environment["DECODE_CACHE_MB"].flatMap { Int($0) } ?? 2048
        Memory.cacheLimit = capMB * 1_000_000
        defer { Memory.cacheLimit = prevCacheLimit }
        MLX.Memory.clearCache()  // drop the denoise cache before the capped decode begins
        // DECODE_DEVICE=gpu runs the streaming VAE decode on the GPU stream. Default stays .cpu
        // (fp32 parity / cold-load-watchdog avoidance). Per-chunk command buffers are short, so the
        // whole-seq watchdog-resubmit risk is bounded — this toggle is the A/B for the CPU-bound wall.
        let decodeDevice: Device = (ProcessInfo.processInfo.environment["DECODE_DEVICE"] == "gpu") ? .gpu : .cpu
        return Device.withDefaultDevice(decodeDevice) {
            let video = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0), chunkLat: 1)
            eval(video)
            return video
        }
    }

    // MARK: - Modes (no preprocessing — the consumer-first set)

    /// Pure text-to-video — **no control branch**. VACE's backbone IS Wan2.1-T2V-1.3B unchanged,
    /// so t2v with no condition is base Wan t2v (coherent), and skipping the VCU sidesteps the
    /// entire control-frame VAE encode (E15: the ~106 GB / glacial VCU-build wall — there is no
    /// control signal to encode for pure t2v). `denoiseVACE(vaceContext: nil)` runs the base
    /// `WanModel` forward. No preprocessing.
    public func t2v(
        prompt: String, negativePrompt: String? = nil,
        width: Int = 832, height: Int = 480, numFrames: Int = 81,
        steps: Int? = nil, guideScale: Double? = nil, seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt
        vaceMemLog("t2v start (no control)")

        // §2.4: page umT5 in, encode cond/uncond, evict before denoise.
        let (contextCond, contextNull) = try WanProfiler.shared.region("phase", "text_encode") {
            try withTextEncoder { enc -> (MLXArray, MLXArray) in
                let c = encodeText(
                    encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
                let n = encodeText(
                    encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
                eval(c, n)
                return (c, n)
            }
        }
        vaceMemLog("umT5 evicted (post-encode)")

        // Latent geometry straight from the vae strides — no VCU to size it from.
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        let latent = try WanProfiler.shared.region("phase", "denoise") {
            let l = try denoiseVACE(
                model: model, config: config, contextCond: contextCond, contextNull: contextNull,
                vaceContext: nil, noise: noise, steps: steps ?? config.sampleSteps,
                shift: config.sampleShift, guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
                onStep: onStep)
            eval(l)
            return l
        }
        vaceMemLog("denoise done (pre-decode)")
        MLX.Memory.clearCache()
        WanDebug.stats("latent (pre-decode)", latent)

        let frames = WanProfiler.shared.region("phase", "decode") { decodeLatent(latent) }
        WanDebug.stats("frames (post-decode)", frames)
        vaceMemLog("decode done")
        return frames
    }

    /// First-frame image-to-video. The image occupies (and stays frozen at) frame 0
    /// (inactive); the rest is generated (reactive). No preprocessing.
    /// - image: `[3, H, W]` in [-1, 1] (channels-first).
    public func i2v(
        image: MLXArray, prompt: String, negativePrompt: String? = nil,
        numFrames: Int = 81, steps: Int? = nil, guideScale: Double? = nil,
        seed: UInt64? = nil, onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let (h, w) = (image.dim(1), image.dim(2))
        // frame 0 = the image, rest = 0 (gray); mask 0 = keep frame 0, 1 = generate the rest.
        let rest = MLXArray.zeros([3, numFrames - 1, h, w])
        let frames = concatenated([image.expandedDimensions(axis: 1), rest], axis: 1)  // [3, T, H, W]
        let m0 = MLXArray.zeros([1, 1, h, w])
        let mRest = MLXArray.ones([1, numFrames - 1, h, w])
        let mask = concatenated([m0, mRest], axis: 1)  // [1, T, H, W]
        return try generate(
            prompt: prompt, negativePrompt: negativePrompt, frames: frames, mask: mask,
            steps: steps, guideScale: guideScale, seed: seed, onStep: onStep)
    }

    /// Inpainting / video-editing: regenerate the masked region of a source video,
    /// keep the rest. No preprocessing.
    /// - video: `[3, T, H, W]` in [-1, 1]; mask: `[1, T, H, W]` 0/1 (1 = regenerate).
    public func inpaint(
        video: MLXArray, mask: MLXArray, prompt: String, negativePrompt: String? = nil,
        steps: Int? = nil, guideScale: Double? = nil, seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        try generate(
            prompt: prompt, negativePrompt: negativePrompt, frames: video, mask: mask,
            steps: steps, guideScale: guideScale, seed: seed, onStep: onStep)
    }
}

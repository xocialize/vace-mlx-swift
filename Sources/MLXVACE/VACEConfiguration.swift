import Foundation
import WanCore
import MLXToolKit

/// Init-time configuration for `MLXVACEPackage` (C9): which variant and where the flat
/// checkpoint lives. Per-request prompt/size/steps ride the canonical `T2VRequest` /
/// `VEditRequest`, not here.
///
/// Checkpoint resolution order at `load()`:
///   1. `modelDirectory` (a resolved flat dir: `model.safetensors` (backbone + vace_*) +
///      `vae.safetensors` + `t5_encoder.safetensors` + `config.json`)
///   2. HF download of `repo` into the local cache (`WeightLoader.snapshotDownload`)
public struct VACEConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Published variant repo id (also the provenance source).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the chosen variant — selection metadata; the loader auto-detects
    /// the actual quantization from the checkpoint's config.json.
    public var quant: Quant
    /// Resolved local checkpoint folder. Environment-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    /// Engine-chosen models root (future auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "Wan-AI/Wan2.1-VACE-1.3B",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The int4 consumer config (the lightest as-shipped tier).
    public static var int4: VACEConfiguration {
        VACEConfiguration(quant: .int4)
    }

    /// VACE-Fun-A14B (quality/pro tier): the dual-expert all-in-one control/editing model.
    /// Same surface as the 1.3B consumer tier at A14B fidelity. The pipeline auto-detects the
    /// dual-expert `high_noise_model/` + `low_noise_model/` checkpoint layout at `load()`.
    public static var funA14B: VACEConfiguration {
        VACEConfiguration(repo: "alibaba-pai/Wan2.2-VACE-Fun-A14B")
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the resolved flat checkpoint into the OS file
/// cache before `load()`'s GPU evals, so a cold load-time `eval` never faults weights off
/// slow/external storage inside a live Metal command buffer (the cold-load GPU watchdog,
/// `kIOGPUCommandBufferCallbackErrorTimeout`). VACE-1.3B is a light backbone (unlikely to bite),
/// but the conformance is family-uniform. The whole resolved `modelDirectory` is paged; only the
/// config knows the path, execution is the engine's (`WeightPrewarmer`, best-effort). Nil on the
/// HF-download path → no-op.
extension VACEConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { [modelDirectory].compactMap { $0 } }
}

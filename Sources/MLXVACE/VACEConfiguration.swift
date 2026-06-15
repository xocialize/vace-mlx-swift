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
public struct VACEConfiguration: PackageConfiguration, ModelStorable {
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

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}

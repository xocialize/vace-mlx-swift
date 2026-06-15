import Foundation
import MLX
import MLXNN
import XCTest

@testable import VACE
import WanCore

/// Exercises the E15 fix path: a single VaceWanModel forward at seqLen ≥ wanLargeSeq (1024),
/// where `vaceHints` now evals each vace block (the bounded path). The parity/smoke tests all
/// run BELOW the threshold, so this is the only gate that actually executes the `evalEachBlock`
/// branch — it catches a shape/logic regression in the fix before a GPU run spends time on it.
/// Forward-only (synthetic inputs, no umT5/VAE) so it stays cheap on the CPU stream.
final class VaceLargeSeqTests: XCTestCase {
    static let mlxDir = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/vace-1.3b-measure/models/vace-1.3b-mlx")

    func testLargeSeqForwardFinite() throws {
        let weights = Self.mlxDir.appendingPathComponent("model.safetensors")
        if !FileManager.default.fileExists(atPath: weights.path) {
            throw XCTSkip("VACE weights not present")
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(
                from: Self.mlxDir.appendingPathComponent("config.json"))
            let vaceLayers = Array(stride(from: 0, to: config.numLayers, by: 2))
            let model = VaceWanModel(config: config, vaceLayers: vaceLayers, vaceInDim: 96)
            let w = try WeightLoader.loadSafetensors(url: weights)
                .filter { $0.key != "freqs" }.mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(w), verify: [.noUnusedKeys])
            eval(model.parameters())

            // Latent geometry → seqLen = tTok·hTok·wTok with patch [1,2,2]:
            // [16, 5, 32, 32] → 5·16·16 = 1280  (> wanLargeSeq 1024 ⇒ evalEachBlock fires).
            let (tLat, hLat, wLat) = (5, 32, 32)
            let seqLen = tLat * (hLat / config.patchSize[1]) * (wLat / config.patchSize[2])
            XCTAssertGreaterThanOrEqual(seqLen, wanLargeSeq, "test must cross the threshold")

            let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])
            let vcu = MLXRandom.normal([96, tLat, hLat, wLat])
            let ctx = MLXRandom.normal([8, config.textDim])

            let out = model(
                [noise], t: MLXArray([Float(500)]), context: .raw([ctx]),
                seqLen: seqLen, vaceContext: [vcu])
            eval(out)
            let o = out[0]
            let mx = o.max().item(Float.self), mn = o.min().item(Float.self)
            print("[VACE large-seq] seqLen=\(seqLen) out \(o.shape) range [\(mn), \(mx)]")
            XCTAssertEqual(o.shape, [config.vaeZDim, tLat, hLat, wLat], "forward output shape")
            XCTAssertTrue(mx.isFinite && mn.isFinite, "non-finite at large seqLen")
        }
    }
}

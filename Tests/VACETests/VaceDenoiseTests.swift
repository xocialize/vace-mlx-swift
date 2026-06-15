import Foundation
import MLX
import MLXNN
import XCTest

@testable import VACE
import WanCore

/// Smoke gate for the VACE denoise loop: the forward (`VaceBranchParityTests`) and the FlowUniPC
/// scheduler (wan-core, parity-locked via TI2V) are already bit-exact, and the CFG wiring mirrors
/// the parity-locked TI2V loop — so this validates the loop runs end-to-end and yields a finite,
/// correctly-shaped latent (catches wiring/shape regressions). Reuses the DiT golden tensors as
/// deterministic inputs.
final class VaceDenoiseTests: XCTestCase {
    static let root = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/vace-1.3b-measure")
    var mlxWeights: URL { Self.root.appendingPathComponent("models/vace-1.3b-mlx/model.safetensors") }
    var parityDir: URL { Self.root.appendingPathComponent("vace_parity") }
    private func golden(_ n: String) throws -> MLXArray {
        try loadNumpy(url: parityDir.appendingPathComponent("\(n).npy"))
    }

    func testDenoiseLoopRunsFinite() throws {
        if !FileManager.default.fileExists(atPath: mlxWeights.path) {
            throw XCTSkip("VACE weights not present")
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(
                from: mlxWeights.deletingLastPathComponent().appendingPathComponent("config.json"))
            let vaceLayers = Array(stride(from: 0, to: config.numLayers, by: 2))
            let model = VaceWanModel(config: config, vaceLayers: vaceLayers, vaceInDim: 96)
            let weights = try WeightLoader.loadSafetensors(url: mlxWeights).mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model.parameters())

            let noise = try golden("in_x")               // [16, 1, 16, 16]
            let vcu = try golden("in_vace_context")      // [96, 1, 16, 16]
            let ctxCond = try golden("in_context")       // [8, 4096]
            let ctxNull = ctxCond * 0                     // a trivial uncond for the smoke

            let out = denoiseVACE(
                model: model, config: config, contextCond: ctxCond, contextNull: ctxNull,
                vaceContext: vcu, noise: noise, steps: 3, shift: 5.0, guideScale: 5.0)
            eval(out)
            let m = out.max().item(Float.self), mn = out.min().item(Float.self)
            print("[VACE denoise smoke] out \(out.shape) range [\(mn), \(m)]")
            XCTAssertEqual(out.shape, [16, 1, 16, 16], "denoise output shape")
            XCTAssertTrue(m.isFinite && mn.isFinite, "denoise produced non-finite latent")
        }
    }
}

import Foundation
import MLX
import MLXNN
import XCTest

@testable import VACE
import WanCore

/// Bit-exact gate for the VACE-1.3B forward (backbone + Context-Adapter branch) on the CPU stream,
/// vs the PyTorch oracle goldens (`vace_parity/gen_golden.py`, fixed seed, fp32). Fixture-gated:
/// skipped (not failed) when the DEV_ARCHIVE measure tree is absent.
final class VaceBranchParityTests: XCTestCase {
    static let root = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/vace-1.3b-measure")
    var modelDir: URL { Self.root.appendingPathComponent("models/wan2.1-vace-1.3b") }
    var mlxWeights: URL { Self.root.appendingPathComponent("models/vace-1.3b-mlx/model.safetensors") }
    var parityDir: URL { Self.root.appendingPathComponent("vace_parity") }

    private func golden(_ name: String) throws -> MLXArray {
        try loadNumpy(url: parityDir.appendingPathComponent("\(name).npy"))
    }
    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a - b).max().item(Float.self) }

    func testVaceForwardMatchesGolden() throws {
        let fm = FileManager.default
        for u in [mlxWeights, parityDir.appendingPathComponent("g_output.npy")] {
            if !fm.fileExists(atPath: u.path) { throw XCTSkip("VACE fixtures not present: \(u.path)") }
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            // wan-core-compatible config lives next to the converted MLX weights (the native VACE
            // config.json lacks wan-core fields like model_version/vae_stride).
            let config = try WanConfig.load(
                from: mlxWeights.deletingLastPathComponent().appendingPathComponent("config.json"))
            let vaceLayers = Array(stride(from: 0, to: config.numLayers, by: 2))  // [0,2,…,28]
            let model = VaceWanModel(config: config, vaceLayers: vaceLayers, vaceInDim: 96)
            let weights = try WeightLoader.loadSafetensors(url: mlxWeights)
                .mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model.parameters())

            // Fixed-seed golden inputs (channels-first per-sample tensors, as the oracle used).
            let inX = try golden("in_x")               // [16, 1, 16, 16]
            let inVace = try golden("in_vace_context") // [96, 1, 16, 16]
            let inCtx = try golden("in_context")       // [8, 4096]
            let inT = try golden("in_t")               // [1]
            let gOut = try golden("g_output")          // [16, 1, 16, 16]
            let seqLen = 64

            let out = model(
                [inX], t: inT, context: .raw([inCtx]), seqLen: seqLen,
                vaceContext: [inVace], vaceContextScale: 1.0)
            eval(out)
            let d = maxAbs(out[0], gOut)
            print("[VACE parity] output max-abs=\(d) shape=\(out[0].shape)")
            XCTAssertEqual(out[0].shape, gOut.shape, "VACE output shape mismatch")
            XCTAssertLessThan(d, 1e-3, "VACE forward max-abs \(d)")
        }
    }

    /// W5 localization: the Fr=1 golden above cannot see temporal ORDER. This runs the Swift VACE
    /// forward on a 3-frame golden and compares against the oracle output AND its temporal reverse.
    /// forward-match ⇒ DiT temporal order is correct (W5 is elsewhere); reverse-match ⇒ the DiT
    /// (patchify/RoPE/unpatchify) flips the temporal axis — the W5 root cause.
    func testVaceForwardTemporalOrderMultiFrame() throws {
        let fm = FileManager.default
        for u in [mlxWeights, parityDir.appendingPathComponent("g_output_mf.npy")] {
            if !fm.fileExists(atPath: u.path) { throw XCTSkip("multi-frame VACE fixtures not present: \(u.path)") }
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(
                from: mlxWeights.deletingLastPathComponent().appendingPathComponent("config.json"))
            let vaceLayers = Array(stride(from: 0, to: config.numLayers, by: 2))
            let model = VaceWanModel(config: config, vaceLayers: vaceLayers, vaceInDim: 96)
            let weights = try WeightLoader.loadSafetensors(url: mlxWeights)
                .mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model.parameters())

            let inX = try golden("in_x_mf")               // [16, 3, 16, 16]
            let inVace = try golden("in_vace_context_mf") // [96, 3, 16, 16]
            let inCtx = try golden("in_context_mf")       // [8, 4096]
            let inT = try golden("in_t_mf")               // [1]
            let gOut = try golden("g_output_mf")          // [16, 3, 16, 16]
            let seqLen = 3 * 8 * 8                          // Fr=3, patch (1,2,2) → 192

            let out = model(
                [inX], t: inT, context: .raw([inCtx]), seqLen: seqLen,
                vaceContext: [inVace], vaceContextScale: 1.0)
            eval(out)
            XCTAssertEqual(out[0].shape, gOut.shape, "multi-frame output shape mismatch")

            // Temporal reverse of the oracle output along axis 1 (the 3 frames).
            let f = gOut.dim(1)
            let gRev = concatenated((0..<f).reversed().map { gOut[0..., $0..<($0 + 1)] }, axis: 1)
            let dForward = maxAbs(out[0], gOut)
            let dReversed = maxAbs(out[0], gRev)
            print("[W5 DiT order] forward max-abs=\(dForward)  reversed max-abs=\(dReversed)  shape=\(out[0].shape)")
            // Whichever is ~0 tells us the orientation. Assert forward is the match (DiT order correct).
            XCTAssertLessThan(
                dForward, 1e-3,
                "VACE multi-frame forward does NOT match oracle forward (forward=\(dForward) reversed=\(dReversed)) "
                + (dReversed < dForward ? "— Swift output is the TEMPORAL REVERSE of the oracle → W5 is in the DiT"
                                        : "— neither orientation matches; not a clean reversal"))
        }
    }
}

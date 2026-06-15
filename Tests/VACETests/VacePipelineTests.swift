import Foundation
import MLX
import MLXNN
import XCTest

@testable import VACE
import WanCore

/// End-to-end smoke gate for the full `VACEPipeline.generate` relay: umT5 encode →
/// VCU build (the net-new `VaceVCU.encodeFrames` inactive/reactive split + parity-locked
/// mask space-to-depth) → `denoiseVACE` → WanVAE decode. Each component is already
/// parity-locked/smoke-tested in isolation; this confirms they wire together and yield
/// finite, correctly-shaped frames. Runs on the CPU stream (CLI metallib boundary) with
/// tiny dims to keep umT5/denoise tractable. Skips unless the full checkpoint is present.
final class VacePipelineTests: XCTestCase {
    static let mlxDir = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/vace-1.3b-measure/models/vace-1.3b-mlx")

    func testGenerateRelayRunsFinite() async throws {
        let needed = ["model.safetensors", "vae.safetensors", "t5_encoder.safetensors", "config.json"]
        for f in needed where !FileManager.default.fileExists(
            atPath: Self.mlxDir.appendingPathComponent(f).path) {
            throw XCTSkip("VACE checkpoint incomplete (missing \(f))")
        }

        try await Device.withDefaultDevice(Device(.cpu)) {
            let pipe = try await VACEPipeline.fromPretrained(modelDir: Self.mlxDir)

            // Tiny synthetic condition: 5 frames @ 128² (→ tLat 2, 16² latent), full-regen mask.
            let (t, h, w) = (5, 128, 128)
            let frames = MLXRandom.uniform(low: -1.0, high: 1.0, [3, t, h, w])
            let mask = MLXArray.ones([1, t, h, w])  // all-reactive (t2v-style control)

            let out = try pipe.generate(
                prompt: "a calm ocean at sunset", frames: frames, mask: mask,
                steps: 2, guideScale: 5.0, seed: 0)
            eval(out)
            let mx = out.max().item(Float.self), mn = out.min().item(Float.self)
            print("[VACE pipeline smoke] frames \(out.shape) range [\(mn), \(mx)]")
            // Decode → [1, 3, T', H', W'] in [-1, 1]; T' = (tLat-1)*4+1.
            XCTAssertEqual(out.dim(0), 1, "batch")
            XCTAssertEqual(out.dim(1), 3, "channels")
            XCTAssertTrue(mx.isFinite && mn.isFinite, "non-finite frames")
            XCTAssertGreaterThanOrEqual(mn, -1.0001)
            XCTAssertLessThanOrEqual(mx, 1.0001)
        }
    }
}

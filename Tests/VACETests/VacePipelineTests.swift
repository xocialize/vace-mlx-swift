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

    /// W5 DEFINITIVE end-to-end temporal-direction check. i2v freezes a known image at frame 0
    /// (mask 0 = keep). With a strong top-bright/bottom-dark first frame, the frozen frame carries a
    /// large top−bottom luminance contrast (~+1.8). Forward order ⇒ that contrast peaks at OUTPUT
    /// frame 0; a clean temporal reversal ⇒ it peaks at the LAST output frame. Computable (no need to
    /// watch). Tiny dims + CPU to stay tractable. Skips unless the full checkpoint is present.
    func testI2VTemporalDirectionEndToEnd() async throws {
        let needed = ["model.safetensors", "vae.safetensors", "t5_encoder.safetensors", "config.json"]
        for f in needed where !FileManager.default.fileExists(
            atPath: Self.mlxDir.appendingPathComponent(f).path) {
            throw XCTSkip("VACE checkpoint incomplete (missing \(f))")
        }

        try await Device.withDefaultDevice(Device(.cpu)) {
            let pipe = try await VACEPipeline.fromPretrained(modelDir: Self.mlxDir)

            let (h, w) = (128, 128)
            // Distinctive first frame: top half bright (+0.9), bottom half dark (-0.9).
            let top = MLXArray.ones([3, h / 2, w]) * Float(0.9)
            let bot = MLXArray.ones([3, h / 2, w]) * Float(-0.9)
            let image = concatenated([top, bot], axis: 1)  // [3, H, W] in [-1, 1]

            let out = try pipe.i2v(
                image: image, prompt: "a calm ocean at sunset", numFrames: 9, steps: 4, seed: 0)
            eval(out)
            let tOut = out.dim(2)

            func contrast(_ f: Int) -> Float {
                let frame = out[0, 0..., f, 0..., 0...]            // [3, H, W]
                let t = frame[0..., 0..<(h / 2), 0...].mean().item(Float.self)
                let b = frame[0..., (h / 2)..., 0...].mean().item(Float.self)
                return t - b
            }
            let perFrame = (0..<tOut).map { contrast($0) }
            let argmax = perFrame.firstIndex(of: perFrame.max()!)!
            print("[W5 i2v dir] per-frame top−bottom contrast (\(tOut)): "
                + "\(perFrame.map { String(format: "%.3f", $0) })  argmax=\(argmax)")
            // The frozen input frame (highest contrast) MUST be the FIRST output frame (forward order).
            XCTAssertLessThan(
                argmax, tOut / 2,
                "i2v frozen first-frame surfaced at output frame \(argmax)/\(tOut) — if near the END, "
                + "the pipeline REVERSES temporal order (W5). contrasts=\(perFrame)")
        }
    }
}

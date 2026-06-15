import Foundation
import MLX
import XCTest

@testable import VACE
import WanCore

/// Bit-exact gate for the VCU mask space-to-depth (the net-new VACE pixel-mask → 64-ch latent
/// encode) vs the verbatim oracle `vace_encode_masks` (`gen_mask_golden.py`).
final class VaceVCUParityTests: XCTestCase {
    static let parity = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/vace-1.3b-measure/vace_parity")
    private func golden(_ n: String) throws -> MLXArray {
        try loadNumpy(url: Self.parity.appendingPathComponent("\(n).npy"))
    }

    func testMaskSpaceToDepthMatchesGolden() throws {
        let p = Self.parity.appendingPathComponent("g_mask64.npy")
        if !FileManager.default.fileExists(atPath: p.path) { throw XCTSkip("VCU mask golden not present") }
        try Device.withDefaultDevice(Device(.cpu)) {
            let inMask = try golden("in_mask")   // [1, 1, 128, 128]
            let gMask = try golden("g_mask64")   // [64, 1, 16, 16]
            let out = VaceVCU.maskSpaceToDepth(inMask)
            eval(out)
            XCTAssertEqual(out.shape, gMask.shape, "mask64 shape mismatch")
            let d = abs(out - gMask).max().item(Float.self)
            print("[VCU mask parity] max-abs=\(d) shape=\(out.shape)")
            XCTAssertLessThan(d, 1e-6, "mask space-to-depth max-abs \(d)")
        }
    }
}

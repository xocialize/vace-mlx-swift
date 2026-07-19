import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import MLXVACE

// W8 orientation gate: reference decode must round-trip with the encode convention.
//
// `FrameEncode.encodePNG` builds its CGImage straight from the tensor bytes, so tensor row 0 →
// image row 0 (top). `FrameDecode.rgbCHW` must read back the same way. A flip here put i2v/v2v conditioned frames upside-down
// relative to t2v, which has no pixel input and stayed correct — the asymmetry that made it hard
// to see (operator-confirmed ground/sky swap, device-independent).
//
// This is the regression guard W8 never had: the flip was fixed here in vace `3014274` but the
// same one-helper bug silently survived in the bernini-r sibling until W9, because nothing gated
// the pixel→tensor boundary. Every `.npy` parity fixture is latent-space and injected downstream
// of it, so FrameDecode is not in their call graph.
//
// Deliberately MLX-free (pure CoreGraphics) so it runs in the offline tier — the MLX-backed
// targets can't load a default metallib under `swift test`.

@Suite struct OrientationTests {

    /// PNG bytes for an asymmetric marker — top half pure RED, bottom half pure BLUE — built with
    /// the same CGImage construction `FrameEncode.encodePNG` uses (row 0 = top).
    private func asymmetricPNG(width: Int, height: Int) throws -> Data {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 3
                if y < height / 2 { rgb[i] = 255 } else { rgb[i + 2] = 255 }
            }
        }
        let cfData = CFDataCreate(nil, rgb, rgb.count)!
        let provider = try #require(CGDataProvider(data: cfData))
        let image = try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let out = NSMutableData()
        let dest = try #require(
            CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, nil)
        #expect(CGImageDestinationFinalize(dest))
        return out as Data
    }

    @Test func referenceDecodeRoundTripsEncodeOrientation() throws {
        let (w, h) = (8, 8)
        let png = try asymmetricPNG(width: w, height: h)
        let chw = rgbCHW(try cgImage(from: png), width: w, height: h)
        #expect(chw.count == 3 * h * w)

        let plane = h * w
        // Row 0 must still be RED and the last row BLUE. A vertical flip swaps them.
        #expect(chw[0] > 0.5, "row 0 lost RED — reference decode is vertically flipped (W8)")
        #expect(chw[2 * plane] < -0.5, "row 0 gained BLUE — reference decode is flipped (W8)")
        #expect(
            chw[(h - 1) * w] < -0.5, "last row gained RED — reference decode is flipped (W8)")
        #expect(
            chw[2 * plane + (h - 1) * w] > 0.5,
            "last row lost BLUE — reference decode is vertically flipped (W8)")
    }
}

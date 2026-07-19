// Canonical-artifact DECODE for the editing surfaces — the reverse of
// FrameEncode. `Image` → reference pixels [1,3,1,H,W]; `Video` → source pixels
// [1,3,T,H,W], both in [-1,1], CHW, top-down (matching the oracle's PIL
// preprocessing). Pure CoreGraphics/AVFoundation; produces the MLXArrays the
// core editing methods (`r2v`/`videoEdit`) consume.

import AVFoundation
import WanCore
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXToolKit

enum FrameDecodeError: Error {
    case imageDecode
    case videoNoFrames
}

/// CGImage → RGB float [3, H, W] in [-1, 1], read row 0 = image top — matching
/// the pipeline's decode/output convention (`FrameEncode` writes latent row 0 →
/// video row 0, no flip; t2v output is correctly oriented). W8: a prior explicit
/// vertical flip here (`translateBy(height)` + `scaleBy(y:-1)`) was the ONLY flip
/// in the conditioned path, so i2v/v2v frames entered upside-down vs t2v/decode
/// (operator-confirmed ground/sky swap, device-independent). Drawing the CGImage
/// straight into this `premultipliedLast` context already yields the top-down
/// raster the rest of the pipeline uses, so no flip is applied.
func rgbCHW(_ cg: CGImage, width: Int, height: Int) -> [Float] {
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let ctx = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high  // ≈ bicubic, matching PIL.BICUBIC
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

    var chw = [Float](repeating: 0, count: 3 * height * width)
    let plane = height * width
    for y in 0..<height {
        for x in 0..<width {
            let p = (y * width + x) * 4
            let i = y * width + x
            chw[i] = Float(rgba[p]) / 255 * 2 - 1  // R
            chw[plane + i] = Float(rgba[p + 1]) / 255 * 2 - 1  // G
            chw[2 * plane + i] = Float(rgba[p + 2]) / 255 * 2 - 1  // B
        }
    }
    return chw
}

func cgImage(from data: Data) throws -> CGImage {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw FrameDecodeError.imageDecode }
    return cg
}

/// Reference `Image` → pixels [1, 3, 1, H, W] in [-1, 1] (one temporal frame).
func decodeReferencePixels(_ image: Image, width: Int, height: Int) throws -> MLXArray {
    let chw = try rgbCHW(cgImage(from: image.data), width: width, height: height)
    return MLXArray(chw, [1, 3, 1, height, width])
}

/// Source `Video` → pixels [1, 3, T, H, W] in [-1, 1], sampling up to `numFrames`
/// frames evenly across the clip (the oracle's `_preprocess_video`).
func decodeVideoPixels(_ video: Video, width: Int, height: Int, numFrames: Int) async throws
    -> MLXArray
{
    // AVAsset is file-based; stage the bytes to a temp file.
    let url = FileManager.default.temporaryDirectory
        .appending(path: "vace-src-\(UUID().uuidString).\(video.format.rawValue)")
    try video.data.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let asset = AVURLAsset(url: url)
    let reader = try AVAssetReader(asset: asset)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw FrameDecodeError.videoNoFrames
    }
    let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    reader.startReading()

    // Collect all frames as CGImages, then sample evenly.
    var frames: [CGImage] = []
    while let sample = output.copyNextSampleBuffer(),
          let pixelBuffer = CMSampleBufferGetImageBuffer(sample) {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        if let cg = CIContext().createCGImage(ci, from: ci.extent) {
            frames.append(cg)
        }
    }
    guard !frames.isEmpty else { throw FrameDecodeError.videoNoFrames }

    let n = min(frames.count, numFrames)
    let idx = (0..<n).map { Int(Double($0) * Double(frames.count - 1) / Double(max(n - 1, 1))) }

    // [3, T, H, W]: lay each frame's CHW into the temporal slot.
    let plane = height * width
    var thw = [Float](repeating: 0, count: 3 * n * plane)
    for (t, i) in idx.enumerated() {
        let chw = rgbCHW(frames[i], width: width, height: height)
        for c in 0..<3 {
            for j in 0..<plane {
                thw[c * n * plane + t * plane + j] = chw[c * plane + j]
            }
        }
    }
    return MLXArray(thw, [1, 3, n, height, width])
}

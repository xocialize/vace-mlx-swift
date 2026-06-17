// RunVACE — GPU profiling CLI for the Wan family's "painfully slow" consumer (VACE-1.3B).
// Runs one real t2v generation (the cleanest baseline — base Wan forward, no VCU) under the
// shared `WanProfiler` and dumps the per-phase / per-step CSV breakdown that proves where the
// wall-clock goes. This is P1 of the profiling program: get the VACE-1.3B numbers first.
//
//   WAN_PROFILE=1 swift run -c release RunVACE "a red fox in snow" \
//     [--frames 17] [--width 832] [--height 480] [--steps 8] [--seed 42] \
//     [--model-dir /Volumes/DEV_ARCHIVE/vace-1.3b-measure/models/vace-1.3b-mlx] [--out /tmp/vace] [--profile]
//
//   --profile          force WAN_PROFILE=1 + dump the CSV at the end (convenience).
//   WAN_PROFILE_DEEP=blocks   additionally break the DiT forward down per-block.
//
// If the SPM-CLI metallib boundary bites (MLX error: failed to load the default metallib), run
// under Xcode / xcodebuild instead — the workspace convention for live GPU inference.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers
import VACE
import WanCore

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func writePNG(_ frame: MLXArray, to url: URL) throws {
    // frame: [3, H, W] in [-1, 1]
    let h = frame.dim(1), w = frame.dim(2)
    let rgb = clip((frame.asType(.float32) + 1) * Float(127.5), min: 0, max: 255)
        .asType(.uint8).transposed(1, 2, 0)  // [H, W, 3]
    eval(rgb)
    let bytes: [UInt8] = rgb.asArray(UInt8.self)
    let data = CFDataCreate(nil, bytes, bytes.count)!
    let provider = CGDataProvider(data: data)!
    let image = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

@main
struct RunVACE {
    static func main() async throws {
        // `--profile` is self-contained: set WAN_PROFILE before the lazy `WanProfiler.shared`
        // singleton is first touched (nothing touches it during load — only the generate path).
        let forceProfile = CommandLine.arguments.contains("--profile")
        if forceProfile { setenv("WAN_PROFILE", "1", 1) }

        let positional = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") }
        let prompt = positional.first
            ?? "A red fox standing in fresh snow, golden hour, photorealistic"
        let numFrames = argValue("--frames").flatMap(Int.init) ?? 17
        let width = argValue("--width").flatMap(Int.init) ?? 832
        let height = argValue("--height").flatMap(Int.init) ?? 480
        // Default to a SHORT step count: per-step ms (the headline) is step-count-independent, so
        // a few steps profile the cost without a full 40-step wait. Raise for a true-runtime number.
        let steps = argValue("--steps").flatMap(Int.init) ?? 8
        let seed = argValue("--seed").flatMap(UInt64.init) ?? 42
        let modelDir = URL(filePath: argValue("--model-dir")
            ?? "/Volumes/DEV_ARCHIVE/vace-1.3b-measure/models/vace-1.3b-mlx")
        let outDir = URL(filePath: argValue("--out") ?? "/tmp/vace")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        print("Loading VACE pipeline from \(modelDir.path) …")
        let tLoad = Date()
        let pipeline = try await VACEPipeline.fromPretrained(modelDir: modelDir)
        print(String(format: "  load: %.1fs  (phys %.1f GB)",
                     -tLoad.timeIntervalSinceNow, Double(physFootprintBytes()) / 1e9))

        print("Generating \(numFrames) frame(s) @ \(width)x\(height), \(steps) steps (t2v / no control)")
        print("  prompt: \(prompt)")
        let tGen = Date()
        var lastStep = Date()
        let progress: (Int, Int, MLXArray) -> Void = { step, total, _ in
            let dt = -lastStep.timeIntervalSinceNow; lastStep = Date()
            print(String(format: "  step %d/%d  %.1fs  active %.1f GB  phys %.1f GB",
                         step + 1, total, dt, Double(Memory.snapshot().activeMemory) / 1e9,
                         Double(physFootprintBytes()) / 1e9))
        }
        let frames = try pipeline.t2v(
            prompt: prompt, width: width, height: height, numFrames: numFrames,
            steps: steps, seed: seed, onStep: progress)
        print(String(format: "  generate: %.1fs total", -tGen.timeIntervalSinceNow))

        let t = frames.dim(2)
        for i in 0..<t {
            try writePNG(frames[0, 0..., i, 0..., 0...],
                         to: outDir.appending(path: String(format: "frame_%03d.png", i)))
        }
        print("Wrote \(t) frame(s) -> \(outDir.path)")
        print(String(format: "Peak phys footprint: %.1f GB", Double(physFootprintBytes()) / 1e9))

        if WanProfiler.shared.enabled {
            // tLat/hLat/wLat from the 16-ch WanVAE strides [4,8,8] + patchSize [1,2,2] → seqLen.
            let tLat = (numFrames - 1) / 4 + 1
            let seqLen = tLat * (height / 8 / 2) * (width / 8 / 2)
            print("\n========== WanProfiler CSV (VACE-1.3B t2v) ==========")
            WanProfiler.shared.dumpCSV(denominators: [
                "step": Double(steps),
                "frame": Double(t),
                "1k_token": Double(seqLen) / 1000.0,
            ])
        }
    }
}

// CancellationTests.swift — BOTH VACE tiers through the engine's CAN gate (offline, no
// MLX kernels): MLXVACEPackage (consumer 1.3B) and MLXVACEFunPackage (dual-expert A14B).
// CAN-1/2 drive each real run() pre-cancelled (the entry checkpoint fires before the
// notLoaded guard or weights); CAN-3 is the document of record for the checkpoint cadence
// (identical across tiers — the surface runners in VACESurfaces.swift are shared): every
// sampler path (t2v / i2v / v2v generate, single- and dual-expert) threads a throwing
// `onStep` closure — `try Task.checkCancellation()` once per denoising step — and the
// wan-core streaming VAE decode bails per temporal chunk (`Task.isCancelled` in
// decodeStreaming; the VCU build's encodeStreaming likewise), with the surface runners'
// post-core `try Task.checkCancellation()` discarding a truncated result.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXVACE

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MLXVACEPackage(configuration: VACEConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunFunA14B() async {
        let package = MLXVACEFunPackage(configuration: VACEConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    /// The shared cadence (VACESurfaces.swift runners drive both tiers identically).
    private var cadence: CancellationConformance.CheckpointPosture {
        .cadence([
            // Per denoising step: the throwing onStep closure threaded into every
            // sampler path (VaceSampling / VaceDualExpertSampling via runVACET2V /
            // runVACEVideoEdit in VACESurfaces.swift).
            .init(phase: .denoise, unit: .step),
            // Per VAE-decode temporal chunk: wan-core decodeStreaming bails on
            // Task.isCancelled; the surface runners' post-core checkpoint rethrows.
            .init(phase: .decode, unit: .chunk),
        ])
    }

    func testCANCadenceDeclaration() {
        // textToVideo + videoEdit are long-run capabilities — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXVACEPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: MLXVACEPackage.manifest, posture: cadence)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationFunA14B() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXVACEFunPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: MLXVACEFunPackage.manifest, posture: cadence)
        XCTAssertTrue(report.passed, report.summary)
    }
}

import WanCore
import MLXToolKit

extension Mode {
    /// Fewer denoising steps — the quicker path (VACE-1.3B denoises with FlowUniPC; no
    /// alternate scheduler is parity-validated yet, so `.fast` only trims the step count).
    public static let fast: Mode = "fast"
    /// The reference quality path (config-default steps); the package default.
    public static let quality: Mode = "quality"
}

/// Resolve a request `mode` (+ any explicit `steps`) to a denoise step count. An explicit
/// `steps` always wins; otherwise `.fast` → 20, else nil (the core's config default).
func resolveSteps(mode: Mode?, steps: Int?) -> Int? {
    if let steps { return steps }
    switch mode {
    case .fast: return 20
    default: return nil  // nil / .quality / unknown → config-default steps
    }
}

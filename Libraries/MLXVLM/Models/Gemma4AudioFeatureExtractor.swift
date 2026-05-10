// Gemma4AudioFeatureExtractor.swift
// USM (Universal Speech Model) audio feature extractor for Gemma 4.
//
// Ported from:
//   3rd_party/mlx-vlm/mlx_vlm/models/gemma4/audio_feature_extractor.py
//
// Pipeline:
//   raw waveform → semicausal pad → unfold frames → preemphasis → window → FFT →
//   magnitude → mel filter bank → log → (optional per-bin normalization)

import Accelerate
import Foundation
import MLX

// MARK: - Mel Filter Bank

/// Create an HTK-scale mel filter bank matrix of shape (numFreqBins, numMelFilters).
private func melFilterBank(
    numFrequencyBins: Int,
    numMelFilters: Int,
    minFrequency: Float,
    maxFrequency: Float,
    samplingRate: Int
) -> [[Float]] {
    func hzToMel(_ freq: Float) -> Float {
        2595.0 * log10(1.0 + freq / 700.0)
    }
    func melToHz(_ mel: Float) -> Float {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let melMin = hzToMel(minFrequency)
    let melMax = hzToMel(maxFrequency)

    // numMelFilters + 2 evenly spaced points in mel space
    var melPoints = [Float](repeating: 0, count: numMelFilters + 2)
    for i in 0..<(numMelFilters + 2) {
        melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(numMelFilters + 1)
    }
    let freqPoints = melPoints.map { melToHz($0) }

    // All FFT bin center frequencies
    var allFreqs = [Float](repeating: 0, count: numFrequencyBins)
    for i in 0..<numFrequencyBins {
        allFreqs[i] = Float(i) * Float(samplingRate) / Float(2 * (numFrequencyBins - 1))
    }

    // Build triangular filters
    var filterBank = [[Float]](repeating: [Float](repeating: 0, count: numMelFilters), count: numFrequencyBins)
    for i in 0..<numMelFilters {
        let lower = freqPoints[i]
        let center = freqPoints[i + 1]
        let upper = freqPoints[i + 2]

        for j in 0..<numFrequencyBins {
            let rising = (allFreqs[j] - lower) / max(center - lower, 1e-10)
            let falling = (upper - allFreqs[j]) / max(upper - center, 1e-10)
            filterBank[j][i] = max(0, min(rising, falling))
        }
    }

    return filterBank
}

// MARK: - Feature Extractor

/// USM-compatible audio feature extractor for Gemma 4.
/// Extracts log-mel spectrograms from raw 16kHz mono PCM waveforms.
/// `@unchecked Sendable` because the only stored MLX state (the mel filter
/// bank and the Hann window) is initialised once and never mutated;
/// concurrent reads are safe.
public final class Gemma4AudioFeatureExtractor: @unchecked Sendable {

    public let featureSize: Int  // Number of mel bins (128)
    public let samplingRate: Int  // Expected sample rate (16000)
    public let frameLength: Int  // Samples per frame (320 @ 20ms)
    public let hopLength: Int  // Hop between frames (160 @ 10ms)
    public let fftLength: Int  // FFT size (512)
    public let melFloor: Float  // Floor for log(mel + floor) (1e-3)

    private let window: [Float]  // Periodic Hann window
    /// Flat row-major mel filter bank, shape `[numFreqBins, featureSize]`.
    /// Stored 1-D so the inner mel-projection step can run as a single
    /// `vDSP_mmul([numFrames, numFreqBins] @ [numFreqBins, featureSize])`
    /// instead of the previous 98M-scalar-op nested-loop hot path that was
    /// taking ~15s per 30s clip in pure Swift.
    private let melFiltersFlat: [Float]
    private let perBinMean: [Float]?
    private let perBinStddev: [Float]?

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16_000,
        frameLengthMs: Float = 20.0,
        hopLengthMs: Float = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        fftOverdrive: Bool = false,
        melFloor: Float = 1e-3,
        perBinMean: [Float]? = nil,
        perBinStddev: [Float]? = nil
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.melFloor = melFloor
        self.perBinMean = perBinMean
        self.perBinStddev = perBinStddev

        self.frameLength = Int(round(Float(samplingRate) * frameLengthMs / 1000.0))
        self.hopLength = Int(round(Float(samplingRate) * hopLengthMs / 1000.0))

        var fft = 1
        while fft < frameLength { fft *= 2 }
        if fftOverdrive { fft *= 2 }
        self.fftLength = fft

        // Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
        // where N = frameLength (periodic, NOT symmetric which would use N-1)
        var win = [Float](repeating: 0, count: frameLength)
        let twoPiOverN = 2.0 * Float.pi / Float(frameLength)
        for n in 0..<frameLength {
            win[n] = 0.5 - 0.5 * cos(twoPiOverN * Float(n))
        }
        self.window = win

        let mel2D = melFilterBank(
            numFrequencyBins: fft / 2 + 1,
            numMelFilters: featureSize,
            minFrequency: minFrequency,
            maxFrequency: maxFrequency,
            samplingRate: samplingRate
        )
        // Flatten row-major `[numFreqBins, featureSize]` for vDSP_mmul.
        var flat = [Float]()
        flat.reserveCapacity(mel2D.count * featureSize)
        for row in mel2D { flat.append(contentsOf: row) }
        self.melFiltersFlat = flat
    }

    /// Extract log-mel spectrogram and attention mask from a raw waveform.
    ///
    /// Audio longer than 30 seconds is truncated. Use ``extractChunks(waveform:chunkSamples:)``
    /// for arbitrarily-long recordings — it splits the input into 30 s windows
    /// and batches them through the audio tower in a single forward pass.
    ///
    /// - Parameters:
    ///   - waveform: 1-D Float array of audio samples at `samplingRate` Hz.
    ///   - validSampleCount: Optional explicit count of real (non-padding)
    ///     samples. When `nil`, the whole truncated waveform is treated as
    ///     valid. Used by ``extractChunks(waveform:chunkSamples:)`` so the
    ///     last short chunk gets a correct frame mask after zero-padding.
    /// - Returns: `(features, mask)` where:
    ///   - `features`: MLXArray of shape `[1, numFrames, featureSize]`
    ///   - `mask`: MLXArray of shape `[1, numFrames]` — `true` = valid audio frame
    public func extract(
        waveform: [Float],
        validSampleCount: Int? = nil
    ) -> (features: MLXArray, mask: MLXArray) {
        let maxLength = 480_000  // 30s max
        var wav = waveform
        if wav.count > maxLength {
            wav = Array(wav.prefix(maxLength))
        }

        // Pad to multiple of 128 samples
        let padMultiple = 128
        let remainder = wav.count % padMultiple
        // `originalLength` is the count of real samples whose corresponding
        // frames should be marked valid in the attention mask. Callers from
        // `extractChunks` pass a smaller `validSampleCount` when zero-padding
        // a short last chunk up to the chunk size so its trailing frames
        // are correctly marked invalid.
        let originalLength = validSampleCount.map { min(max($0, 0), wav.count) } ?? wav.count
        if remainder != 0 {
            wav.append(contentsOf: [Float](repeating: 0, count: padMultiple - remainder))
        }

        // Build attention mask for waveform (1 = valid, 0 = padding)
        var attentionMask = [Int32](repeating: 1, count: wav.count)
        for i in originalLength..<wav.count {
            attentionMask[i] = 0
        }

        // Semicausal left-padding: prepend frame_length // 2 zeros
        let padLeft = frameLength / 2
        wav = [Float](repeating: 0, count: padLeft) + wav
        attentionMask = [Int32](repeating: 0, count: padLeft) + attentionMask

        // Frame unfold: window of size (frameLength + 1), step = hopLength
        let frameSizeForUnfold = frameLength + 1
        let numFrames = (wav.count - frameSizeForUnfold) / hopLength + 1
        if numFrames <= 0 {
            // Too short — return empty
            let emptyFeatures = MLXArray.zeros([1, 0, featureSize])
            let emptyMask = MLXArray.zeros([1, 0]).asType(Bool.self)
            return (emptyFeatures, emptyMask)
        }

        // FFT setup once per call.
        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            let emptyFeatures = MLXArray.zeros([1, numFrames, featureSize])
            let emptyMask = MLXArray.zeros([1, numFrames]).asType(Bool.self)
            return (emptyFeatures, emptyMask)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let numFreqBins = fftLength / 2 + 1
        let halfLen = fftLength / 2

        // Pre-allocate flat magnitude matrix `[numFrames, numFreqBins]` so the
        // mel projection downstream can be one vDSP_mmul.
        var allMagnitudes = [Float](repeating: 0, count: numFrames * numFreqBins)

        // Per-frame split-complex buffers (reused across frames).
        var realHalf = [Float](repeating: 0, count: halfLen)
        var imagHalf = [Float](repeating: 0, count: halfLen)

        // Heavyweight inner loop: window, FFT, magnitude, into row of
        // allMagnitudes. Direct buffer access (no array-of-arrays churn).
        wav.withUnsafeBufferPointer { wavPtr in
            window.withUnsafeBufferPointer { winPtr in
                allMagnitudes.withUnsafeMutableBufferPointer { magsPtr in
                    realHalf.withUnsafeMutableBufferPointer { rBuf in
                        imagHalf.withUnsafeMutableBufferPointer { iBuf in
                            var split = DSPSplitComplex(
                                realp: rBuf.baseAddress!,
                                imagp: iBuf.baseAddress!)
                            // Per-frame windowing scratch (reused).
                            var windowed = [Float](repeating: 0, count: frameLength)
                            for i in 0 ..< numFrames {
                                let start = i * hopLength
                                // Vectorized windowing: windowed = frame * window
                                // (vDSP_vmul on contiguous buffers).
                                vDSP_vmul(
                                    wavPtr.baseAddress!.advanced(by: start), 1,
                                    winPtr.baseAddress!, 1,
                                    &windowed, 1, vDSP_Length(frameLength))

                                // Pack windowed[2k]→realp, windowed[2k+1]→imagp
                                // for vDSP_fft_zrip's split-complex input. Use
                                // vDSP_ctoz (interpret pairs as DSPComplex)
                                // for the contiguous portion; pad rest with 0.
                                let pairsInWindow = min(halfLen, frameLength / 2)
                                windowed.withUnsafeBufferPointer { wPtr in
                                    wPtr.baseAddress!.withMemoryRebound(
                                        to: DSPComplex.self, capacity: pairsInWindow
                                    ) { complexPtr in
                                        vDSP_ctoz(complexPtr, 2, &split, 1,
                                                  vDSP_Length(pairsInWindow))
                                    }
                                }
                                // Zero the remainder if frameLength is odd.
                                if pairsInWindow < halfLen {
                                    for k in pairsInWindow ..< halfLen {
                                        rBuf[k] = 0; iBuf[k] = 0
                                    }
                                }

                                vDSP_fft_zrip(fftSetup, &split, 1, log2n,
                                              FFTDirection(kFFTDirection_Forward))

                                // vDSP_fft_zrip scales by 2× relative to numpy's
                                // rfft for bins 1..N/2-1, but DC and Nyquist are
                                // un-scaled. Multiply middle bins by 0.5.
                                let row = magsPtr.baseAddress!.advanced(by: i * numFreqBins)
                                row[0] = abs(rBuf[0]) * 0.5  // DC
                                if numFreqBins > halfLen {
                                    row[halfLen] = abs(iBuf[0]) * 0.5  // Nyquist
                                }
                                // Magnitudes for middle bins.
                                for k in 1 ..< halfLen {
                                    let r = rBuf[k]
                                    let im = iBuf[k]
                                    row[k] = sqrt(r * r + im * im) * 0.5
                                }
                            }
                        }
                    }
                }
            }
        }

        // Mel projection: [numFrames, numFreqBins] @ [numFreqBins, featureSize]
        // → [numFrames, featureSize]. One vDSP_mmul replaces the previous
        // 98M-scalar Swift inner loop and dominates the speedup.
        var melFlat = [Float](repeating: 0, count: numFrames * featureSize)
        allMagnitudes.withUnsafeBufferPointer { magsPtr in
            melFiltersFlat.withUnsafeBufferPointer { mfPtr in
                melFlat.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_mmul(
                        magsPtr.baseAddress!, 1,
                        mfPtr.baseAddress!, 1,
                        outPtr.baseAddress!, 1,
                        vDSP_Length(numFrames),
                        vDSP_Length(featureSize),
                        vDSP_Length(numFreqBins))
                }
            }
        }

        // Apply mel floor + log + optional per-bin normalisation.
        // melFlat[i, m] = log(mel + floor) [- mean[m]] [/ stddev[m]]
        let mean = perBinMean, stddev = perBinStddev
        for i in 0 ..< numFrames {
            let row = i * featureSize
            for m in 0 ..< featureSize {
                var v = log(melFlat[row + m] + melFloor)
                if let mean, mean.count == featureSize { v -= mean[m] }
                if let stddev, stddev.count == featureSize { v /= stddev[m] }
                melFlat[row + m] = v
            }
        }

        // Build frame-level attention mask: a frame is valid iff the last
        // sample in its unfold window was real audio.
        var frameMask = [Bool](repeating: false, count: numFrames)
        for i in 0 ..< numFrames {
            let frameEndIdx = i * hopLength + frameSizeForUnfold - 1
            if frameEndIdx < attentionMask.count {
                frameMask[i] = attentionMask[frameEndIdx] == 1
            }
        }

        // Zero out padded frames (matching HF: spec * mask[..., None]).
        for i in 0 ..< numFrames where !frameMask[i] {
            let row = i * featureSize
            for m in 0 ..< featureSize { melFlat[row + m] = 0 }
        }

        // Hand the flat mel buffer to MLX directly — no intermediate
        // [[Float]] → flat conversion needed.
        let features = MLXArray(melFlat, [1, numFrames, featureSize])

        // Match Hugging Face Gemma4AudioFeatureExtractor: true = valid, false = padding
        let mask = MLXArray(frameMask, [1, numFrames])

        return (features, mask)
    }

    /// Extract log-mel spectrograms for arbitrarily-long audio by splitting it
    /// into fixed-length windows that match the audio tower's native context
    /// (default 30 seconds at 16 kHz, the same as Whisper / Gemma 4 audio).
    /// Each window is zero-padded to `chunkSamples` so they all have identical
    /// shape and can be batched through the encoder in a single forward pass;
    /// padded frames are marked invalid in the per-chunk frame mask.
    ///
    /// Mirrors `mtmd_audio_preprocessor_gemma4a::preprocess` in llama.cpp's
    /// `tools/mtmd/mtmd-audio.cpp`, which splits long audio into 30-second
    /// windows before feeding the Conformer encoder.
    ///
    /// - Parameters:
    ///   - waveform: Raw waveform as Float array at `samplingRate` Hz, any length
    ///   - chunkSamples: Window size in samples (default 30 s × 16 kHz = 480 000)
    /// - Returns: `(features, mask, validTokensPerChunk)` where:
    ///   - `features`: MLXArray of shape `[N_chunks, framesPerChunk, featureSize]`
    ///   - `mask`: MLXArray of shape `[N_chunks, framesPerChunk]` — `true` = valid frame
    ///   - `validTokensPerChunk`: For each chunk, the number of audio-tokens
    ///     that should appear in the LM prompt (= number of post-subsample
    ///     conformer frames whose entire receptive field is real audio, not
    ///     zero-pad). Mirrors HF transformers' Gemma4AudioProcessor's
    ///     `_compute_audio_num_tokens` so a 9.8 s clip emits ~245 tokens (not
    ///     750), preventing the LM from seeing trailing zero-embedding tokens
    ///     in the audio block.
    public func extractChunks(
        waveform: [Float],
        chunkSamples: Int = 480_000,
        audioMsPerToken: Float = 40.0,
        audioSeqLength: Int = 750
    ) -> (features: MLXArray, mask: MLXArray, validTokensPerChunk: [Int]) {
        precondition(chunkSamples > 0, "chunkSamples must be positive")
        let total = max(waveform.count, 1)
        let nChunks = max(1, (total + chunkSamples - 1) / chunkSamples)

        var features: [MLXArray] = []
        var masks: [MLXArray] = []
        var validTokens: [Int] = []
        features.reserveCapacity(nChunks)
        masks.reserveCapacity(nChunks)
        validTokens.reserveCapacity(nChunks)

        for c in 0 ..< nChunks {
            let start = c * chunkSamples
            let end = min(start + chunkSamples, waveform.count)
            // Zero-pad each window to exactly `chunkSamples` so all chunks have
            // identical frame counts — the audio tower can then process them
            // as a single batched forward pass.
            var window = [Float](repeating: 0, count: chunkSamples)
            if end > start {
                for i in start ..< end {
                    window[i - start] = waveform[i]
                }
            }
            let validCount = end - start
            let (chunkFeatures, chunkMask) = extract(
                waveform: window, validSampleCount: validCount
            )
            features.append(chunkFeatures)
            masks.append(chunkMask)

            // Number of post-subsample audio tokens for this chunk's *real*
            // duration. Mirrors HF transformers `Gemma4Processor._compute_audio_num_tokens`:
            //   ceil(duration_ms / audio_ms_per_token), capped at audio_seq_length.
            let chunkDurationMs = Float(validCount) / Float(samplingRate) * 1000.0
            let n = Int((chunkDurationMs / audioMsPerToken).rounded(.up))
            validTokens.append(min(max(n, 1), audioSeqLength))
        }

        let stackedFeatures: MLXArray
        let stackedMask: MLXArray
        if features.count == 1 {
            stackedFeatures = features[0]
            stackedMask = masks[0]
        } else {
            // Each tensor is [1, frames, mel] / [1, frames]; concatenate along axis 0
            // to get [N, frames, mel] / [N, frames].
            stackedFeatures = concatenated(features, axis: 0)
            stackedMask = concatenated(masks, axis: 0)
        }
        return (stackedFeatures, stackedMask, validTokens)
    }
}

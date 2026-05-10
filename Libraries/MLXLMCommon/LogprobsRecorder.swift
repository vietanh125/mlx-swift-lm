// Per-token logprob capture for streaming generation.
//
// Wraps an arbitrary `LogitProcessor` (e.g. a grammar-masking processor)
// and records the post-processing log-probabilities of each sampled
// token plus its top-K alternatives. The recorded distribution reflects
// whatever constraints `inner` applied — matching llama.cpp's behaviour
// of reporting logprobs *under the constrained distribution* when
// grammars are active.

import Foundation
import MLX
import MLXNN

// MARK: - OpenAI-compatible record types

/// One alternative considered at a generation step alongside the sampled token.
public struct TopTokenLogprob: Sendable, Codable {
    public let id: Int
    public let token: String
    public let bytes: [UInt8]
    public let logprob: Float

    public init(id: Int, token: String, bytes: [UInt8], logprob: Float) {
        self.id = id
        self.token = token
        self.bytes = bytes
        self.logprob = logprob
    }
}

/// Per-token record matching the OpenAI / llama.cpp `logprobs.content[]` schema.
public struct TokenLogprobRecord: Sendable, Codable {
    public let id: Int
    public let token: String
    public let bytes: [UInt8]
    public let logprob: Float
    public let topLogprobs: [TopTokenLogprob]

    enum CodingKeys: String, CodingKey {
        case id, token, bytes, logprob
        case topLogprobs = "top_logprobs"
    }

    public init(
        id: Int, token: String, bytes: [UInt8], logprob: Float,
        topLogprobs: [TopTokenLogprob]
    ) {
        self.id = id
        self.token = token
        self.bytes = bytes
        self.logprob = logprob
        self.topLogprobs = topLogprobs
    }
}

/// Internal raw entry — IDs only, no detokenization. Decoded into
/// ``TokenLogprobRecord`` by ``LogprobsRecorder/decoded(using:)``.
struct RawLogprobEntry: Sendable {
    let sampledId: Int
    let sampledLogprob: Float
    /// Top-K (id, logprob) pairs in descending order of logprob.
    let topK: [(Int, Float)]
}

// MARK: - Recorder

/// `LogitProcessor` that wraps an inner processor (typically a
/// `GrammarMaskedLogitProcessor`) and records the post-processing
/// log-probabilities of each sampled token plus the top-K alternatives.
/// The recorded distribution reflects whatever constraints `inner` applied
/// — matching llama.cpp's behaviour of returning logprobs *under the
/// constrained distribution* when grammars are active.
///
/// Tokenizer-agnostic: stores raw token IDs only. Call
/// ``decoded(using:)`` after generation to attach token strings + bytes.
public final class LogprobsRecorder: LogitProcessor, @unchecked Sendable {
    private let topK: Int
    private var inner: LogitProcessor?

    private let lock = NSLock()
    private var lastProcessedLogits: MLXArray?
    private var rawRecords: [RawLogprobEntry] = []

    public init(topK: Int, inner: LogitProcessor? = nil) {
        self.topK = max(1, topK)
        self.inner = inner
    }

    /// Number of token records currently captured.
    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        return rawRecords.count
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        rawRecords.removeAll(keepingCapacity: true)
        lastProcessedLogits = nil
    }

    public func prompt(_ prompt: MLXArray) {
        inner?.prompt(prompt)
        lock.lock()
        rawRecords.removeAll(keepingCapacity: true)
        lastProcessedLogits = nil
        lock.unlock()
    }

    public func process(logits: MLXArray) -> MLXArray {
        let processed = inner?.process(logits: logits) ?? logits
        lock.lock()
        lastProcessedLogits = processed
        lock.unlock()
        return processed
    }

    public func didSample(token: MLXArray) {
        inner?.didSample(token: token)

        lock.lock()
        guard let processed = lastProcessedLogits else {
            lock.unlock()
            return
        }
        let k = topK
        lock.unlock()

        // Squeeze any leading singleton dim so we work in 1-D.
        let flat: MLXArray
        if processed.shape.count >= 2 {
            flat = processed.reshaped([processed.size])
        } else {
            flat = processed
        }

        let logProbs = MLXNN.logSoftmax(flat, axis: -1)
        let total = logProbs.size
        let take = min(k, total)
        // argSort is ascending; the last `take` entries are the top-K.
        let order = MLX.argSort(logProbs, axis: -1)
        let topIndices = order[(total - take)...]
        MLX.eval(logProbs, topIndices)

        let sampledIdScalar = token.reshaped([token.size])[0].item(Int32.self)
        let sampledId = Int(sampledIdScalar)
        let sampledLp = logProbs[sampledId].item(Float.self)

        let topIdxArr = topIndices.asArray(Int32.self)
        var topPairs: [(Int, Float)] = []
        topPairs.reserveCapacity(take)
        // Reverse to get descending by logprob. Filter out -∞ entries that
        // come from grammar-masked tokens — they're not encodable as JSON
        // and would never have been sampled anyway. Mirrors llama.cpp,
        // which only reports tokens with finite probability under the
        // constrained distribution.
        for i in stride(from: topIdxArr.count - 1, through: 0, by: -1) {
            let id = Int(topIdxArr[i])
            let lp = logProbs[id].item(Float.self)
            if lp.isFinite {
                topPairs.append((id, lp))
            }
        }

        // Sampled token logprob: clamp to a finite floor for the same
        // JSON-encoder reason. In normal sampling this is always finite
        // (an -∞ token can't be sampled), but we guard defensively.
        let safeSampledLp: Float = sampledLp.isFinite ? sampledLp : -1e30

        let entry = RawLogprobEntry(
            sampledId: sampledId,
            sampledLogprob: safeSampledLp,
            topK: topPairs
        )
        lock.lock()
        rawRecords.append(entry)
        lock.unlock()
    }

    /// Decode raw IDs into token strings + UTF-8 bytes using the supplied tokenizer.
    /// Call once, after generation completes (when the tokenizer is in scope).
    public func decoded(using tokenizer: any Tokenizer) -> [TokenLogprobRecord] {
        lock.lock()
        let snapshot = rawRecords
        lock.unlock()

        var cache: [Int: (String, [UInt8])] = [:]
        func decode(_ id: Int) -> (String, [UInt8]) {
            if let hit = cache[id] { return hit }
            // SkipSpecialTokens=false: callers typically want special tokens
            // (e.g. "<|audio|>") rendered verbatim so the byte view matches
            // llama.cpp's `logprobs.content[].bytes`.
            let str = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            let pair = (str, Array(str.utf8))
            cache[id] = pair
            return pair
        }

        return snapshot.map { entry in
            let (sTok, sBytes) = decode(entry.sampledId)
            let topRecs = entry.topK.map { (id, lp) -> TopTokenLogprob in
                let (t, b) = decode(id)
                return TopTokenLogprob(id: id, token: t, bytes: b, logprob: lp)
            }
            return TokenLogprobRecord(
                id: entry.sampledId,
                token: sTok,
                bytes: sBytes,
                logprob: entry.sampledLogprob,
                topLogprobs: topRecs
            )
        }
    }
}

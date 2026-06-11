import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Gemma-4 MTP (multi-token-prediction) assistant + speculative iterator.
//
// Port of llama.cpp's Gemma-4 MTP support (ggml-org/llama.cpp PRs #22673,
// #23398, #24282; reference graph: `src/models/gemma4-assistant.cpp`,
// reference loop: `common/speculative.cpp` `common_speculative_impl_draft_mtp`).
//
// Architecture (E4B assistant: 4 layers, hidden 256):
//  - The assistant has NO K/V projections and NO KV cache of its own. Each of
//    its layers computes queries only and attends read-only into the TARGET
//    model's KV buffers: sliding layers into the target's last concrete
//    sliding-window cache, the full-attention layer into the target's last
//    concrete full-attention cache (llama.cpp `share()` mapping).
//  - Input per token: concat(targetEmbed(token) * sqrt(2560), h) -> 5120,
//    projected to 256 via `pre_projection`. `h` is recurrent: the target's
//    post-final-norm hidden for the first draft, then the assistant's own
//    `post_projection` output for subsequent drafts.
//  - All draft queries use the SAME RoPE position (the current context
//    length), matching the shared-memory drafting scheme in llama.cpp
//    (`speculative.cpp`, "with shared memory we use the same position for all
//    draft tokens").
//  - Output head is the assistant's own tied 256-dim embedding table.

// MARK: - Configuration

public struct Gemma4AssistantConfiguration: Codable, Sendable {
    /// Hidden size of the target backbone (2560 for E4B). Width of the
    /// recurrent `h` input/output and half of `pre_projection`'s input.
    public let backboneHiddenSize: Int
    public let textConfiguration: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case backboneHiddenSize = "backbone_hidden_size"
        case textConfiguration = "text_config"
    }
}

// MARK: - Assistant modules

private final class Gemma4AssistantAttention: Module {
    let numHeads: Int
    let headDim: Int
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        // Head dims must match the target's caches the layer attends into:
        // sliding -> target head_dim (256), full -> target global_head_dim (512).
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)

        // RoPE must place draft queries in the same rotary space as the
        // target's cached keys — same parameters as the target layer type.
        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    /// `keys`/`values`: read-only views of the target's KV buffers, already
    /// in `[B, kvHeads, T, headDim]` layout with RoPE applied at write time.
    func callAsFunction(
        _ x: MLXArray, keys: MLXArray, values: MLXArray, positionOffset: Int
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: positionOffset)

        // Single-position queries attending the full visible window — no mask
        // needed (the sliding cache's content IS the window; the full cache is
        // entirely visible to a causal query at the current position).
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys.asType(queries.dtype),
            values: values.asType(queries.dtype),
            scale: 1.0,
            mask: .none
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class Gemma4AssistantDecoderLayer: Module {
    let isSliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4AssistantAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self._selfAttention.wrappedValue = Gemma4AssistantAttention(
            config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, keys: MLXArray, values: MLXArray, positionOffset: Int
    ) -> MLXArray {
        var residual = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, keys: keys, values: values, positionOffset: positionOffset)
        h = residual + postAttentionLayerNorm(h)

        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = residual + postFeedforwardLayerNorm(h)

        return h * layerScalar
    }
}

// MARK: - Assistant model

public final class Gemma4AssistantModel: Module {
    public let config: Gemma4AssistantConfiguration
    private let embedScale: Float

    fileprivate final class Core: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [Gemma4AssistantDecoderLayer]
        @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift

        init(_ config: Gemma4TextConfiguration) {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                Gemma4AssistantDecoderLayer(config: config, layerIdx: $0)
            }
            self._norm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }
    }

    @ModuleInfo(key: "model") fileprivate var core: Core
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        self.embedScale = pow(Float(config.backboneHiddenSize), 0.5)
        self._core.wrappedValue = Core(config.textConfiguration)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            config.textConfiguration.hiddenSize, config.backboneHiddenSize, bias: false)
        super.init()
    }

    /// One assistant forward step.
    ///
    /// - Parameters:
    ///   - tokens: `[1, S]` token ids (S = 1 during drafting).
    ///   - hidden: `[1, S, backboneHiddenSize]` recurrent input — the target's
    ///     post-final-norm hidden for the first draft of a round, the previous
    ///     step's `hiddenNext` afterwards.
    ///   - targetEmbedTokens: the TARGET model's input-embedding table
    ///     (`Gemma4.mtpEmbedTokens`).
    ///   - fullKV / slidingKV: read-only `[B, kvHeads, T, headDim]` views of
    ///     the target's shared caches.
    ///   - positionOffset: current target context length; all draft queries
    ///     are placed at this RoPE position.
    public func callAsFunction(
        tokens: MLXArray,
        hidden: MLXArray,
        targetEmbedTokens: Embedding,
        fullKV: (keys: MLXArray, values: MLXArray),
        slidingKV: (keys: MLXArray, values: MLXArray),
        positionOffset: Int
    ) -> (logits: MLXArray, hiddenNext: MLXArray) {
        var x = targetEmbedTokens(tokens)
        x = x * MLXArray(embedScale, dtype: x.dtype)

        let xh = concatenated([x, hidden.asType(x.dtype)], axis: -1)
        var h = preProjection(xh)

        for layer in core.layers {
            let kv = layer.isSliding ? slidingKV : fullKV
            h = layer(h, keys: kv.keys, values: kv.values, positionOffset: positionOffset)
        }

        h = core.norm(h)
        let logits = core.embedTokens.asLinear(h)
        let hiddenNext = postProjection(h)
        return (logits, hiddenNext)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // `masked_embedding.{centroids.weight,token_ordering}` belong to an
        // HF-side masked-drafting mode that neither llama.cpp nor this port
        // uses (llama.cpp marks them TENSOR_NOT_REQUIRED).
        weights.filter { !$0.key.hasPrefix("masked_embedding.") }
    }

    /// Load an assistant checkpoint (config.json + safetensors) from a local
    /// directory, e.g. a HuggingFace snapshot of
    /// `mlx-community/gemma-4-E4B-it-assistant-bf16`.
    public static func load(directory: URL) throws -> Gemma4AssistantModel {
        let configData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self, from: configData)
        let model = Gemma4AssistantModel(config)

        var weights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for url in files where url.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: url) {
                weights[key] = value
            }
        }
        guard !weights.isEmpty else {
            throw NSError(
                domain: "Gemma4AssistantModel", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "no safetensors found in \(directory.path)"
                ])
        }
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}

// MARK: - Speculative iterator

/// Token iterator that drives Gemma-4 MTP speculative decoding.
///
/// Same call shape as `TokenIterator` (construct, then `while let t = next()`).
/// Output is distribution-identical to plain decoding: draft tokens are only
/// emitted when the target model — with the caller's sampler AND logit
/// processor applied — samples the same token at that position.
///
/// Caveat shared with `SpeculativeTokenIterator`: if `processor` is a
/// reference type, its state advances for verify-sampled tokens that end up
/// rejected. Use value-type processors or none (Scribion wires MTP for
/// unconstrained roles first).
public struct Gemma4MTPTokenIterator: Sequence, IteratorProtocol {
    public typealias Element = Int

    let target: Gemma4
    let assistant: Gemma4AssistantModel
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    public let maxTokens: Int?
    public private(set) var tokenCount = 0
    public private(set) var promptPrefillTime: TimeInterval = 0

    /// Acceptance telemetry for TSV / logs.
    public private(set) var draftTokensGenerated = 0
    public private(set) var draftTokensAccepted = 0

    let numDraftTokens: Int
    let pMin: Float
    let fullCacheIdx: Int
    let slidingCacheIdx: Int

    /// Last committed token — the first row of the next verify batch.
    private var idLast: MLXArray
    /// Target post-final-norm hidden at the last committed context position;
    /// nil until the bootstrap step has run.
    private var pendingH: MLXArray?

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var finished = false

    public init(
        input: LMInput,
        model: Gemma4,
        assistant: Gemma4AssistantModel,
        cache: [KVCache],
        processor: LogitProcessor? = nil,
        sampler: LogitSampler,
        prefillStepSize: Int = 512,
        maxTokens: Int? = nil,
        numDraftTokens: Int = 3,
        pMin: Float = 0
    ) throws {
        precondition(
            assistant.config.backboneHiddenSize == model.config.textConfiguration.hiddenSize,
            "assistant backbone width \(assistant.config.backboneHiddenSize) != target hidden size \(model.config.textConfiguration.hiddenSize)"
        )
        self.target = model
        self.assistant = assistant
        self.cache = cache
        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens
        self.numDraftTokens = Swift.max(numDraftTokens, 0)
        self.pMin = pMin
        let shared = model.mtpSharedCacheIndices
        self.fullCacheIdx = shared.full
        self.slidingCacheIdx = shared.sliding
        self.idLast = MLXArray([Int32(0)])

        let prefillStart = Date()
        self.processor?.prompt(input.text.tokens)
        switch try model.prepare(input, cache: cache, windowSize: prefillStepSize) {
        case .logits(let result):
            let token = sample(logits: result.logits[0..., -1, 0...])
            self.processor?.didSample(token: token)
            idLast = token
            eval(token)
            pendingTokens.append(token.item(Int.self))
        case .tokens:
            fatalError("Gemma4.prepare always returns .logits")
        }
        promptPrefillTime = Date().timeIntervalSince(prefillStart)
    }

    private mutating func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        return sampler.sample(logits: logits)
    }

    /// Read-only views of the target's shared KV buffers, valid for the whole
    /// drafting phase of one round (the target doesn't decode while drafting).
    private func sharedKVViews() -> (
        full: (keys: MLXArray, values: MLXArray),
        sliding: (keys: MLXArray, values: MLXArray)
    ) {
        guard let full = cache[fullCacheIdx] as? KVCacheSimple,
            let fullK = full.keys, let fullV = full.values
        else {
            fatalError("Gemma4 MTP expects a KVCacheSimple full-attention cache with content")
        }
        let fullView = (
            keys: fullK[.ellipsis, ..<full.offset, 0...],
            values: fullV[.ellipsis, ..<full.offset, 0...]
        )

        guard let sliding = cache[slidingCacheIdx] as? RotatingKVCache else {
            fatalError("Gemma4 MTP expects a RotatingKVCache sliding cache")
        }
        let state = sliding.state
        guard state.count == 2 else {
            fatalError("Gemma4 MTP: sliding cache has no content")
        }
        return (full: fullView, sliding: (keys: state[0], values: state[1]))
    }

    /// Roll back the `n` rejected draft positions from every target cache.
    private func trimCaches(_ n: Int) {
        guard n > 0 else { return }
        for layer in cache {
            if let rotating = layer as? RotatingKVCache {
                rotating.trimTail(n)
            } else {
                layer.trim(n)
            }
        }
    }

    private mutating func speculateRound() {
        let nPast = cache[fullCacheIdx].offset

        // Bootstrap (first round, or drafting disabled): plain decode step
        // that captures the hidden state seeding the recurrence.
        guard let h0 = pendingH, numDraftTokens > 0 else {
            let (logits, hidden) = target.mtpTextStep(
                idLast.reshaped([1, 1]), cache: cache)
            let token = sample(logits: logits[0..., -1, 0...])
            processor?.didSample(token: token)
            pendingH = hidden[0..., (hidden.dim(1) - 1)..., 0...]
            idLast = token
            eval(token)
            pendingTokens.append(token.item(Int.self))
            return
        }

        // 1. Draft: recurrent single-token assistant steps, greedy, at the
        //    fixed position nPast.
        //
        //    The whole round — 3 assistant steps AND the target verify — is
        //    built as one lazy MLX graph with a single device sync at the end.
        //    The drafted token arrays feed both the next assistant step's
        //    embedding lookup and the verify batch without ever leaving the
        //    GPU. With per-step `.item()` syncs instead, the round-trip stalls
        //    ate the entire MTP gain (measured 0.96x; same trap as the D2H
        //    transfers called out in llama.cpp PR #22673).
        //
        //    `pMin > 0` requires inspecting each draft's probability on the
        //    CPU, so that path keeps per-step syncs — it trades stalls for
        //    fewer wasted drafts.
        let views = sharedKVViews()
        var draftTokenArrays = [MLXArray]()
        var draftToken = idLast
        var draftH = h0
        for _ in 0 ..< numDraftTokens {
            let (logits, hNext) = assistant(
                tokens: draftToken.reshaped([1, 1]),
                hidden: draftH,
                targetEmbedTokens: target.mtpEmbedTokens,
                fullKV: views.full,
                slidingKV: views.sliding,
                positionOffset: nPast
            )
            let lastLogits = logits[0..., -1, 0...]  // [1, V]
            // .int32 to match idLast for the verify-batch concat (argMax
            // yields uint32).
            let token = argMax(lastLogits, axis: -1).asType(.int32)  // [1], lazy
            if pMin > 0 {
                let logit = lastLogits.max(axis: -1)
                let prob = exp(logit - logSumExp(lastLogits, axis: -1))
                eval(token, prob)
                if prob.item(Float.self) < pMin { break }
            }
            draftTokenArrays.append(token)
            draftToken = token
            draftH = hNext
        }
        draftTokensGenerated += draftTokenArrays.count

        // 2. Verify: one target batch over [idLast, d1...dk], chained lazily
        //    onto the draft graph.
        let verifyTokens = concatenated([idLast] + draftTokenArrays)
            .reshaped([1, draftTokenArrays.count + 1])
        let (logits, hidden) = target.mtpTextStep(verifyTokens, cache: cache)

        // 3. Sample the target at every verify position, then force the whole
        //    round's graph in one sync.
        let drafts: [Int]
        let sampled: [Int]
        if processor != nil {
            // Logit processors (grammar FSMs) advance per sampled token on
            // the CPU — this path keeps sequential syncs.
            let draftBatch = concatenated(draftTokenArrays)
            eval(draftBatch)
            drafts = draftBatch.asArray(Int.self)
            var tokens = [Int]()
            tokens.reserveCapacity(drafts.count + 1)
            for i in 0 ..< (drafts.count + 1) {
                let token = sample(logits: logits[0..., i, 0...])
                processor?.didSample(token: token)
                eval(token)
                tokens.append(token.item(Int.self))
            }
            sampled = tokens
        } else {
            let sampledBatch = sampler.sample(logits: logits.squeezed(axis: 0))
            let draftBatch = concatenated(draftTokenArrays)
            eval(sampledBatch, draftBatch)
            sampled = sampledBatch.asArray(Int.self)
            drafts = draftBatch.asArray(Int.self)
        }

        // 4. Accept the longest matching draft prefix; the target's sample at
        //    the first mismatch (or after the last accepted draft) is the
        //    bonus/correction token.
        var accepted = 0
        while accepted < drafts.count && sampled[accepted] == drafts[accepted] {
            accepted += 1
        }
        draftTokensAccepted += accepted

        pendingTokens.append(contentsOf: drafts[..<accepted])
        pendingTokens.append(sampled[accepted])

        // 5. Roll back rejected draft KV, carry the recurrent hidden of the
        //    last committed context position (row `accepted` of the verify
        //    batch), and set up the next round.
        trimCaches(drafts.count - accepted)
        pendingH = hidden[0..., accepted ... accepted, 0...]
        idLast = MLXArray([Int32(sampled[accepted])])
    }

    public mutating func next() -> Int? {
        if finished { return nil }
        if let maxTokens, tokenCount >= maxTokens { return nil }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        guard pendingIndex < pendingTokens.count else {
            finished = true
            return nil
        }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

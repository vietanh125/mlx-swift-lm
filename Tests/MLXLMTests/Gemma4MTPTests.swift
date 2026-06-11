import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

extension MLXTestingSuite {
    @Suite
    struct Gemma4MTPTests {

        // MARK: - RotatingKVCache.trimTail (no model required)

        @Test("trimTail removes rejected rows physically (rotated cache)")
        func testTrimTailAfterRotation() throws {
            let cache = RotatingKVCache(maxSize: 8, keep: 0, step: 4)

            func rows(_ range: Range<Int>) -> MLXArray {
                MLXArray(range.map { Float($0) }, [1, 1, range.count, 1])
            }

            // Fill past the window with multi-token updates (updateConcat path,
            // like prefill + verify batches).
            _ = cache.update(keys: rows(0 ..< 6), values: rows(0 ..< 6))
            _ = cache.update(keys: rows(6 ..< 12), values: rows(6 ..< 12))
            #expect(cache.offset == 12)

            // Verify batch of 4 (1 committed + 3 drafts), then reject 3.
            _ = cache.update(keys: rows(12 ..< 16), values: rows(12 ..< 16))
            cache.trimTail(3)
            #expect(cache.offset == 13)

            // The physical buffer must contain exactly the surviving window —
            // no stale rejected rows (13, 14, 15) anywhere.
            let state = cache.state
            #expect(state.count == 2)
            let kept = Set(state[0].flattened().asArray(Float.self).map { Int($0) })
            #expect(!kept.contains(13))
            #expect(!kept.contains(14))
            #expect(!kept.contains(15))
            #expect(kept.contains(12))

            // Next verify batch must append cleanly after the trim.
            _ = cache.update(keys: rows(13 ..< 17), values: rows(13 ..< 17))
            #expect(cache.offset == 17)
            let kept2 = Set(cache.state[0].flattened().asArray(Float.self).map { Int($0) })
            #expect(kept2.contains(16))
        }

        @Test("single-token update regrows buffer correctly after trimTail")
        func testSingleTokenUpdateAfterTrimTail() throws {
            let cache = RotatingKVCache(maxSize: 8, keep: 0, step: 4)

            func rows(_ range: Range<Int>) -> MLXArray {
                MLXArray(range.map { Float($0) }, [1, 1, range.count, 1])
            }

            _ = cache.update(keys: rows(0 ..< 12), values: rows(0 ..< 12))
            _ = cache.update(keys: rows(12 ..< 16), values: rows(12 ..< 16))
            cache.trimTail(3)  // buffer now shorter than maxSize, offset = 13

            // updateInPlace path (S == 1) — previously sized growth by
            // `maxCacheSize - offset`, which is negative here.
            let (k, _) = cache.update(keys: rows(13 ..< 14), values: rows(13 ..< 14))
            #expect(cache.offset == 14)
            let kept = Set(k.flattened().asArray(Float.self).map { Int($0) })
            #expect(kept.contains(13))
            #expect(!kept.contains(15))
        }

        // MARK: - End-to-end (gated: SCRIBION_MTP_IT=1, models in HF cache)

        static let runIT = ProcessInfo.processInfo.environment["SCRIBION_MTP_IT"] == "1"

        /// German clinical prompt in Gemma chat format, tokenized with the
        /// E4B tokenizer (`tokenizers` 0.22, add_special_tokens=false). Baked
        /// in because the fork carries no tokenizer implementation by design.
        static let promptIds: [Int] = [
            2, 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 23200, 50560,
            2535, 157688, 19983, 91695, 847, 702, 13182, 236761, 105599, 18252, 5723,
            150791, 68938, 7791, 4608, 96418, 8097, 2217, 685, 14141, 84336, 16175,
            236787, 33141, 236764, 236743, 236810, 236812, 32784, 236764, 82206, 5679,
            3232, 31801, 35315, 103635, 27270, 18342, 581, 113175, 501, 236764, 192684,
            793, 8668, 236743, 236800, 236828, 236764, 236819, 20098, 941, 37997, 628,
            33471, 3257, 145517, 27178, 12199, 502, 2438, 9315, 236761, 13043, 233643,
            1277, 9040, 630, 67368, 1716, 214227, 943, 76900, 42611, 236761, 127245,
            236743, 236828, 236812, 12124, 236786, 236752, 236764, 1834, 39325, 2946,
            1571, 236743, 236770, 236800, 236778, 236771, 236771, 236786, 575, 21603,
            643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340,
            236779, 887, 236813, 4368, 107,
        ]
        static let stopIds: Set<Int> = [1, 106]  // <eos>, <end_of_turn>

        @Test("MTP greedy output matches plain greedy decode", .enabled(if: runIT))
        func testMTPGreedyParity() async throws {
            let targetDir: URL
            if let override = ProcessInfo.processInfo.environment["SCRIBION_MTP_TARGET_DIR"] {
                targetDir = URL(fileURLWithPath: override)
            } else {
                targetDir = try Self.locateSnapshot(
                    repo: "models--mlx-community--gemma-4-E4B-it-qat-4bit")
            }
            let assistantDir = try Self.locateSnapshot(
                repo: "models--mlx-community--gemma-4-E4B-it-assistant-bf16")

            let resolved = ResolvedModelConfiguration(
                modelDirectory: targetDir,
                tokenizerDirectory: targetDir,
                name: "gemma-4-E4B-it-qat-4bit",
                defaultPrompt: "",
                extraEOSTokens: [],
                eosTokenIds: Self.stopIds,
                toolCallFormat: nil
            )
            let context = try await VLMModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: StubTokenizerLoader())
            guard let gemma = context.model as? Gemma4 else {
                Issue.record("expected Gemma4, got \(type(of: context.model))")
                return
            }
            let assistant = try Gemma4AssistantModel.load(directory: assistantDir)

            // Optional override with a workload-realistic prompt (JSON array
            // of token ids) — e.g. an extraction prompt, where draft
            // acceptance is much higher than on free-form notes.
            var promptIds = Self.promptIds
            var maxTokens = 120
            if let file = ProcessInfo.processInfo.environment["SCRIBION_MTP_PROMPT_FILE"],
                let data = FileManager.default.contents(atPath: file),
                let ids = try? JSONDecoder().decode([Int].self, from: data)
            {
                promptIds = ids
                maxTokens = 256
            }
            func makeInput() -> LMInput {
                LMInput(
                    text: .init(
                        tokens: MLXArray(promptIds.map { Int32($0) })
                            .expandedDimensions(axis: 0)))
            }

            // Warmup: page in weights and compile kernels so the timed runs
            // below are comparable (whichever runs first would otherwise eat
            // the cold-start cost — first observed as 0.5 tok/s plain vs 11
            // tok/s MTP, a meaningless 24x).
            MLX.GPU.set(cacheLimit: 8 * 1024 * 1024 * 1024)
            var warmupIter = try TokenIterator(
                input: makeInput(), model: gemma,
                cache: gemma.newCache(parameters: GenerateParameters?.none),
                processor: nil, sampler: ArgMaxSampler(),
                prefillStepSize: 512, maxTokens: 16)
            while warmupIter.next() != nil {}

            // Plain greedy decode.
            var plainIter = try TokenIterator(
                input: makeInput(), model: gemma,
                cache: gemma.newCache(parameters: GenerateParameters?.none),
                processor: nil, sampler: ArgMaxSampler(),
                prefillStepSize: 512, maxTokens: maxTokens)
            var plainTokens = [Int]()
            let plainStart = Date()
            while let t = plainIter.next() {
                if Self.stopIds.contains(t) { break }
                plainTokens.append(t)
            }
            let plainTime = Date().timeIntervalSince(plainStart)

            print(
                "[MTP-IT] plain: \(plainTokens.count) tok in "
                    + String(format: "%.2f", plainTime)
                    + "s (\(String(format: "%.1f", Double(plainTokens.count) / plainTime)) tok/s)")

            // Dump the greedy output ids for offline decoding/diffing
            // (checkpoint drift comparisons).
            if let dump = ProcessInfo.processInfo.environment["SCRIBION_MTP_DUMP_TOKENS"] {
                let data = try JSONEncoder().encode(plainTokens)
                try data.write(to: URL(fileURLWithPath: dump))
            }

            // MTP greedy decode across draft depths.
            let depths = (ProcessInfo.processInfo.environment["SCRIBION_MTP_DRAFTS"] ?? "1,2,3")
                .split(separator: ",").compactMap { Int($0) }
            for nDraft in depths {
                var mtpIter = try Gemma4MTPTokenIterator(
                    input: makeInput(), model: gemma, assistant: assistant,
                    cache: gemma.newCache(parameters: GenerateParameters?.none),
                    processor: nil, sampler: ArgMaxSampler(),
                    prefillStepSize: 512, maxTokens: maxTokens, numDraftTokens: nDraft)
                var mtpTokens = [Int]()
                let mtpStart = Date()
                while let t = mtpIter.next() {
                    if Self.stopIds.contains(t) { break }
                    mtpTokens.append(t)
                }
                let mtpTime = Date().timeIntervalSince(mtpStart)

                let drafted = mtpIter.draftTokensGenerated
                let accepted = mtpIter.draftTokensAccepted
                let rate = drafted > 0 ? Double(accepted) / Double(drafted) : 0
                print(
                    "[MTP-IT] mtp n=\(nDraft): \(mtpTokens.count) tok in "
                        + String(format: "%.2f", mtpTime)
                        + "s (\(String(format: "%.1f", Double(mtpTokens.count) / mtpTime)) tok/s), "
                        + "acceptance \(accepted)/\(drafted) = \(String(format: "%.3f", rate)), "
                        + "speedup \(String(format: "%.2f", plainTime / Swift.max(mtpTime, 0.001)))x"
                )

                // Acceptance compares the target's BATCH-shape logits against
                // the drafts; batch matmul reduction order differs numerically
                // from S=1, so a borderline argmax tie can flip mid-sequence
                // and legitimately fork the continuation (llama.cpp MTP has
                // the same property). Require a long common prefix rather than
                // full equality.
                let n = Swift.min(plainTokens.count, mtpTokens.count)
                var common = 0
                while common < n && plainTokens[common] == mtpTokens[common] { common += 1 }
                print("[MTP-IT] mtp n=\(nDraft): common prefix \(common)/\(n)")
                #expect(common >= n / 2, "outputs diverge too early — wiring bug likely")
                #expect(rate > 0.2, "suspiciously low acceptance — wiring bug likely")
            }
        }

        @Test("MTP microbenchmark: step costs", .enabled(if: runIT))
        func testMTPMicrobench() async throws {
            let targetDir = try Self.locateSnapshot(
                repo: "models--mlx-community--gemma-4-E4B-it-qat-4bit")
            let assistantDir = try Self.locateSnapshot(
                repo: "models--mlx-community--gemma-4-E4B-it-assistant-bf16")
            let resolved = ResolvedModelConfiguration(
                modelDirectory: targetDir, tokenizerDirectory: targetDir,
                name: "gemma-4-E4B-it-qat-4bit", defaultPrompt: "",
                extraEOSTokens: [], eosTokenIds: Self.stopIds, toolCallFormat: nil)
            let context = try await VLMModelFactory.shared._load(
                configuration: resolved, tokenizerLoader: StubTokenizerLoader())
            guard let gemma = context.model as? Gemma4 else { return }
            let assistant = try Gemma4AssistantModel.load(directory: assistantDir)

            MLX.GPU.set(cacheLimit: 8 * 1024 * 1024 * 1024)
            let cache = gemma.newCache(parameters: GenerateParameters?.none)
            let prompt = MLXArray(Self.promptIds.map { Int32($0) }).expandedDimensions(axis: 0)
            _ = try gemma.prepare(LMInput(text: .init(tokens: prompt)), cache: cache, windowSize: 512)
            eval(cache.flatMap { ($0 as? BaseKVCache)?.state ?? [] })

            func bench(_ label: String, n: Int = 30, _ body: () -> Void) {
                body()  // warmup
                let start = Date()
                for _ in 0 ..< n { body() }
                let ms = Date().timeIntervalSince(start) / Double(n) * 1000
                print("[MTP-BENCH] \(label): \(String(format: "%.2f", ms)) ms")
            }

            let oneToken = MLXArray([Int32(2364)]).reshaped([1, 1])
            bench("target step S=1") {
                let (logits, _) = gemma.mtpTextStep(oneToken, cache: cache)
                eval(logits)
                for layer in cache {
                    if let r = layer as? RotatingKVCache { r.trimTail(1) } else { layer.trim(1) }
                }
            }

            for s in [2, 3, 4, 6, 8] {
                let tokens = MLXArray((0 ..< s).map { Int32(2364 + $0) }).reshaped([1, s])
                bench("target verify S=\(s)") {
                    let (logits, _) = gemma.mtpTextStep(tokens, cache: cache)
                    eval(logits)
                    for layer in cache {
                        if let r = layer as? RotatingKVCache {
                            r.trimTail(s)
                        } else {
                            layer.trim(s)
                        }
                    }
                }
            }

            let shared = gemma.mtpSharedCacheIndices
            let full = cache[shared.full] as! KVCacheSimple
            let fullKV = (
                keys: full.keys![.ellipsis, ..<full.offset, 0...],
                values: full.values![.ellipsis, ..<full.offset, 0...]
            )
            let slidingState = (cache[shared.sliding] as! RotatingKVCache).state
            let slidingKV = (keys: slidingState[0], values: slidingState[1])
            let h = MLXArray.zeros([1, 1, 2560], dtype: .bfloat16)
            bench("assistant step S=1") {
                let (logits, hNext) = assistant(
                    tokens: oneToken, hidden: h,
                    targetEmbedTokens: gemma.mtpEmbedTokens,
                    fullKV: fullKV, slidingKV: slidingKV,
                    positionOffset: full.offset)
                eval(logits, hNext)
            }

            bench("full MTP round (3 drafts + verify, lazy)") {
                var dt = oneToken
                var dh = h
                var draftArrs = [MLXArray]()
                for _ in 0 ..< 3 {
                    let (logits, hNext) = assistant(
                        tokens: dt.reshaped([1, 1]), hidden: dh,
                        targetEmbedTokens: gemma.mtpEmbedTokens,
                        fullKV: fullKV, slidingKV: slidingKV,
                        positionOffset: full.offset)
                    let tok = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
                    draftArrs.append(tok)
                    dt = tok
                    dh = hNext
                }
                let verify = concatenated([MLXArray([Int32(2364)])] + draftArrs).reshaped([1, 4])
                let (logits, hidden) = gemma.mtpTextStep(verify, cache: cache)
                let sampled = argMax(logits.squeezed(axis: 0), axis: -1)
                eval(sampled, hidden, concatenated(draftArrs))
                for layer in cache {
                    if let r = layer as? RotatingKVCache { r.trimTail(4) } else { layer.trim(4) }
                }
            }
        }

        static func locateSnapshot(repo: String) throws -> URL {
            let base = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub/\(repo)/snapshots")
            let snapshots = try FileManager.default.contentsOfDirectory(
                at: base, includingPropertiesForKeys: nil
            ).filter { !$0.lastPathComponent.hasPrefix(".") }
            guard let snapshot = snapshots.first else {
                throw NSError(
                    domain: "Gemma4MTPTests", code: 2,
                    userInfo: [
                        NSLocalizedDescriptionKey: "no snapshot under \(base.path)"
                    ])
            }
            return snapshot
        }
    }
}

// MARK: - Tokenizer stub (prompt is pre-tokenized; nothing here is exercised)

private struct StubTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct StubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        StubTokenizer()
    }
}

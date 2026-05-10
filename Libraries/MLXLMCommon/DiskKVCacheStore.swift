// Cross-session prefix-cache persistence for prefilled `[KVCache]` state.
//
// Routes through `MLXLMCommon.savePromptCache` / `loadPromptCache`
// (`Libraries/MLXLMCommon/KVCache.swift`) to serialize a prefilled
// `[KVCache]` to a `.safetensors` file in the OS caches directory,
// alongside the original token sequence stored as user metadata.
// On load, the saved tokens are matched against the incoming prompt
// with longest-common-prefix; the cache is then trimmed by the excess
// and returned for partial-prefix reuse.
//
// Equivalent to llama.cpp's `llama_state_save_file` /
// `llama_state_load_file` for the warm-system-prompt use case.

import CryptoKit
import Foundation

/// Stable on-disk prefix cache, keyed by model id (one file per model).
///
/// Each instance writes to its own subdirectory under the OS caches
/// directory; the directory and per-file name are derived from the
/// caller-supplied `namespace` so multiple consumers of mlx-swift-lm
/// in the same process won't collide.
public final class DiskKVCacheStore: @unchecked Sendable {

    // MARK: - Configuration

    /// Storage scope identifier used as the on-disk directory and
    /// filename prefix. Pick something stable per-app (e.g. "myapp").
    public let namespace: String

    /// Maximum total bytes across all entries before LRU eviction kicks in.
    public let maxCacheSizeBytes: Int

    /// Skip both save and partial-restore for prefixes shorter than this —
    /// short prompts don't benefit enough from cross-session caching to
    /// justify the I/O.
    public let minTokensForDiskCache: Int

    /// When true, prints save/load/eviction events to stdout. Off by default.
    public let verbose: Bool

    /// Format version for the metadata payload. Bump on incompatible changes.
    private static let formatVersion = 1

    // MARK: - State

    private let cacheDirectory: URL

    /// In-memory record of "what's currently on disk" so we can skip
    /// redundant saves. Without this, repeatedly calling `save` for the
    /// same prefix (e.g. multi-call synthesis) would write the same
    /// payload N times for no benefit.
    private let lastSavedLock = NSLock()
    private var lastSavedSignature: [String: String] = [:]   // modelId → SHA256

    // MARK: - Init

    public init(
        namespace: String,
        maxCacheSizeBytes: Int = 4 * 1_073_741_824,
        minTokensForDiskCache: Int = 200,
        verbose: Bool = false
    ) {
        precondition(!namespace.isEmpty, "DiskKVCacheStore namespace must be non-empty")
        precondition(
            namespace.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" },
            "DiskKVCacheStore namespace must be a filesystem-safe slug")
        self.namespace = namespace
        self.maxCacheSizeBytes = maxCacheSizeBytes
        self.minTokensForDiskCache = minTokensForDiskCache
        self.verbose = verbose

        let base = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask).first!
        self.cacheDirectory = base
            .appendingPathComponent("\(namespace)-kv-cache", isDirectory: true)
    }

    // MARK: - Public API

    /// Persist the prefilled KV state for `modelId`, alongside the token
    /// sequence that produced it. Best-effort; failures are logged and
    /// swallowed so the in-process cache is never blocked.
    /// Skips the write if the same token sequence was already persisted.
    public func save(modelId: String, tokens: [Int], cache: [KVCache]) {
        guard tokens.count >= minTokensForDiskCache else { return }

        let signature = signatureFor(tokens: tokens)
        lastSavedLock.lock()
        let alreadySaved = lastSavedSignature[modelId] == signature
        lastSavedLock.unlock()
        if alreadySaved {
            return
        }

        do {
            try ensureDirectory()
            let url = url(for: modelId)
            let metadata = makeMetadata(modelId: modelId, tokens: tokens)
            try savePromptCache(url: url, cache: cache, metadata: metadata)
            let size = (try? FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            log("💾 saved for \(modelId): \(size / 1_048_576) MB, \(tokens.count) tokens")
            lastSavedLock.lock()
            lastSavedSignature[modelId] = signature
            lastSavedLock.unlock()
            try? evictIfNeeded()
        } catch {
            log("save failed: \(error)")
        }
    }

    /// Try to restore from disk by longest-common-prefix matching the
    /// incoming `newTokens` against the saved token sequence. Returns
    /// `(cache, matchLen)` on success — `matchLen` is the number of
    /// leading tokens that the caller can skip prefilling. Returns nil
    /// on miss / validation failure / match below `minTokensForDiskCache`.
    public func load(modelId: String, newTokens: [Int]) -> ([KVCache], Int)? {
        let url = url(for: modelId)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let loadStart = CFAbsoluteTimeGetCurrent()
        do {
            let (cache, metadata) = try loadPromptCache(url: url)
            guard validate(metadata: metadata, expectedModelId: modelId) else {
                log("metadata mismatch, discarding")
                try? FileManager.default.removeItem(at: url)
                return nil
            }
            guard let storedTokens = parseTokens(from: metadata) else {
                log("token list malformed, discarding")
                try? FileManager.default.removeItem(at: url)
                return nil
            }
            let matchLen = longestCommonPrefix(storedTokens, newTokens)
            guard matchLen >= minTokensForDiskCache else { return nil }

            // Trim excess so the returned cache reflects only the matched
            // prefix. Each layer trims the same number of trailing tokens.
            let excess = storedTokens.count - matchLen
            if excess > 0 {
                for layer in cache { layer.trim(excess) }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - loadStart
            log("💾 restored for \(modelId): \(matchLen)/\(storedTokens.count) tokens reused (\(excess > 0 ? "partial" : "full") match), \(String(format: "%.3f", elapsed))s")
            // Bump mtime for LRU eviction.
            try? FileManager.default.setAttributes(
                [.modificationDate: Date()],
                ofItemAtPath: url.path
            )
            // Seed the in-memory signature so the immediately-following
            // save() recognises this exact prefix as already persisted
            // and skips the redundant write.
            let sig = signatureFor(tokens: storedTokens)
            lastSavedLock.lock()
            lastSavedSignature[modelId] = sig
            lastSavedLock.unlock()
            return (cache, matchLen)
        } catch {
            log("load failed: \(error)")
            try? FileManager.default.removeItem(at: url)
            return nil
        }
    }

    /// Manual purge — useful for tests or "reset cache" UI affordances.
    public func clearAll() throws {
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
        }
    }

    // MARK: - Internal helpers

    /// One file per model. Filename derived from a hash of the model id so
    /// it's filesystem-safe and bounded in length.
    private func url(for modelId: String) -> URL {
        let hash = SHA256.hash(data: Data(modelId.utf8))
            .prefix(8).map { String(format: "%02x", $0) }.joined()
        return cacheDirectory
            .appendingPathComponent("\(namespace)-\(hash).safetensors")
    }

    /// SHA256 of the token sequence — used to dedupe back-to-back saves.
    private func signatureFor(tokens: [Int]) -> String {
        var data = Data(capacity: tokens.count * 4)
        for token in tokens {
            var t = Int32(token)
            withUnsafeBytes(of: &t) { data.append(contentsOf: $0) }
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// User-metadata keys persisted alongside the .safetensors file. Routed
    /// through `loadPromptCache`'s "1.KEY" namespace.
    private static let metaFormatVersion = "mlx.kvcache.formatVersion"
    private static let metaModelId       = "mlx.kvcache.modelId"
    private static let metaTokens        = "mlx.kvcache.tokens"     // comma-sep int32

    private func makeMetadata(modelId: String, tokens: [Int]) -> [String: String] {
        [
            Self.metaFormatVersion: "\(Self.formatVersion)",
            Self.metaModelId: modelId,
            Self.metaTokens: serializeTokens(tokens),
        ]
    }

    private func validate(metadata: [String: String], expectedModelId: String) -> Bool {
        guard metadata[Self.metaFormatVersion] == "\(Self.formatVersion)" else { return false }
        guard metadata[Self.metaModelId] == expectedModelId else { return false }
        return true
    }

    private func parseTokens(from metadata: [String: String]) -> [Int]? {
        guard let raw = metadata[Self.metaTokens] else { return nil }
        var out: [Int] = []
        out.reserveCapacity(raw.count / 4)
        for piece in raw.split(separator: ",") {
            guard let n = Int(piece) else { return nil }
            out.append(n)
        }
        return out
    }

    private func serializeTokens(_ tokens: [Int]) -> String {
        // Comma-separated decimal — readable, ~6 bytes/token at typical vocab,
        // good enough since the body of a prompt cache is the safetensors
        // payload, not the metadata.
        var out = ""
        out.reserveCapacity(tokens.count * 6)
        for (i, t) in tokens.enumerated() {
            if i > 0 { out.append(",") }
            out.append(String(t))
        }
        return out
    }

    private func longestCommonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    private func ensureDirectory() throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: cacheDirectory.path) {
            try fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
            // Cache data is not user data — exclude from iCloud / Time Machine.
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var mutableURL = cacheDirectory
            try? mutableURL.setResourceValues(values)
        }
    }

    private func evictIfNeeded() throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
        ).filter { $0.pathExtension == "safetensors" }

        var totalSize = 0
        var stat: [(URL, Int, Date)] = []
        for url in entries {
            let v = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            let size = v.fileSize ?? 0
            let date = v.contentModificationDate ?? .distantPast
            stat.append((url, size, date))
            totalSize += size
        }
        guard totalSize > maxCacheSizeBytes else { return }

        // Oldest first.
        stat.sort { $0.2 < $1.2 }
        var freed = 0
        var evicted = 0
        for (url, size, _) in stat {
            guard totalSize > maxCacheSizeBytes else { break }
            try? fm.removeItem(at: url)
            totalSize -= size
            freed += size
            evicted += 1
        }
        if evicted > 0 {
            log("💾 evicted \(evicted) stale file(s) (\(freed / 1_048_576) MB freed)")
        }
    }

    private func log(_ message: @autoclosure () -> String) {
        if verbose {
            print("[\(namespace) KVCache] \(message())")
        }
    }
}

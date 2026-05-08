import XCTest
import MLX
import MLXLLM

final class TestQwenFP8: XCTestCase {
    func testGeneration() async throws {
        let modelDir = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/main")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir) else {
            print("Model not found")
            return
        }

        setenv("EXPERIMENTAL_SSD_STREAM", "1", 1)
        setenv("MLX_MOE_STACKED", "1", 1)

        let (model, tokenizer) = try await MLXLLM.load(hub: MLXLLM.HubConfiguration(id: "Qwen/Qwen3.6-35B-A3B-FP8"))
        let prompt = "Hello! What is 2 + 2?"
        let tokens = tokenizer.encode(text: prompt)
        var result = ""
        for token in MLXLLM.generate(prompt: MLXArray(tokens), model: model) {
            let t = tokenizer.decode(tokens: [token.token])
            result += t
            print(t, terminator: "")
            fflush(stdout)
            if result.count > 30 { break }
        }
        print("\nComplete")
    }
}

import Foundation
import MLX
import Testing

/// Correctness tests for the multi-row qmv kernel (`affine_qmv_fast_nb`) in
/// the mlx-swift fork — an OPT-IN experiment (`MLX_QMV_NB=1`) targeting
/// speculative-decoding verify batches on bandwidth-starved devices. When
/// enabled, the dispatch routes affine quantized matmuls with M in [2, 4],
/// N % 8 == 0, K % 512 == 0 to it; M == 1 and M > 4 stay on the original
/// paths and serve as the reference here (same quantization, same math).
///
/// The suite latches the env flag before the first quantized dispatch in
/// this process, so run it in its own `swift test --filter` invocation when
/// also doing performance measurements of the default path.
extension MLXTestingSuite {
    @Suite
    struct QmvNBKernelTests {
        static let enableNB: Void = {
            setenv("MLX_QMV_NB", "1", 1)
        }()

        private func ramp(_ shape: [Int], scale: Float, dtype: DType) -> MLXArray {
            let n = shape.reduce(1, *)
            return sin(MLXArray(0 ..< n).asType(.float32) * scale)
                .reshaped(shape).asType(dtype)
        }

        @Test(
            "multi-row qmv matches per-row qmv and dequantized reference",
            arguments: [DType.float16, .bfloat16], [4, 8])
        func testQmvNBCorrectness(dtype: DType, bits: Int) throws {
            _ = Self.enableNB
            let K = 1024
            let N = 64
            for groupSize in [32, 64] {
                let w = ramp([N, K], scale: 0.37, dtype: dtype)
                let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
                let wDeq = dequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize,
                    bits: bits, dtype: dtype)

                for m in 1 ... 6 {
                    let x = ramp([m, K], scale: 0.11, dtype: dtype)

                    let y = quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true,
                        groupSize: groupSize, bits: bits)

                    // Per-row single-vector calls take the original qmv path —
                    // identical math, different kernel.
                    var rows = [MLXArray]()
                    for r in 0 ..< m {
                        rows.append(
                            quantizedMM(
                                x[r ..< (r + 1), 0...], wq, scales: scales,
                                biases: biases, transpose: true,
                                groupSize: groupSize, bits: bits))
                    }
                    let yRef = concatenated(rows, axis: 0)
                    eval(y, yRef)

                    // The nb kernel and the (wide-load) qmv_fast path order
                    // their fp32 accumulation differently, so outputs agree
                    // only to reassociation noise + output rounding: measured
                    // ~2.5 ulp at |y|~24 (f16 0.04, bf16 0.19). A nibble-
                    // decode logic bug shows systematic >=1.0 deviations and
                    // still fails.
                    let atol: Double = dtype == .bfloat16 ? 0.5 : 0.08
                    let maxDiff = MLX.abs(
                        y.asType(.float32) - yRef.asType(.float32)
                    ).max().item(Float.self)
                    #expect(
                        allClose(y, yRef, rtol: 1e-2, atol: atol).item(Bool.self),
                        "kernel mismatch: dtype=\(dtype) bits=\(bits) gs=\(groupSize) M=\(m) maxAbsDiff=\(maxDiff)"
                    )

                    // Sanity vs a float32 dequantized matmul. Loose: the
                    // quantized kernels accumulate fp32 but round the output
                    // to dtype (bf16 ≈ 2-3 significant digits at |y| ~ 20),
                    // and this check fails identically on the ORIGINAL kernels
                    // (M=1,5,6) with tight tolerances.
                    let yDense = matmul(
                        x.asType(.float32), wDeq.asType(.float32).transposed())
                    #expect(
                        allClose(y.asType(.float32), yDense, rtol: 2e-2, atol: 0.5)
                            .item(Bool.self),
                        "dequantized mismatch: dtype=\(dtype) bits=\(bits) gs=\(groupSize) M=\(m)"
                    )
                }
            }
        }

        @Test(
            "raw shape benchmark (set SCRIBION_MTP_IT=1; MLX_QMV_NB=1 selects nb path)",
            .enabled(if: ProcessInfo.processInfo.environment["SCRIBION_MTP_IT"] == "1"))
        func benchRawShapes() throws {
            // E4B layer shapes (N=out, K=in): attention QKV-ish, FFN up, FFN
            // down, LM head.
            let shapes: [(n: Int, k: Int, label: String)] = [
                (2048, 2560, "attn  N=2048  K=2560"),
                (16384, 2560, "ffn-up N=16384 K=2560"),
                (2560, 16384, "ffn-dn N=2560  K=16384"),
                (262_144, 2560, "lmhead N=262k K=2560"),
            ]
            let path =
                ProcessInfo.processInfo.environment["MLX_QMV_NB"] != nil
                ? "nb" : "old"
            for (n, k, label) in shapes {
                let w = ramp([n, k], scale: 0.37, dtype: .bfloat16)
                let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
                eval(wq, scales)
                for m in [1, 2, 3, 4] {
                    let x = ramp([m, k], scale: 0.11, dtype: .bfloat16)
                    eval(x)
                    // warmup
                    eval(quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true,
                        groupSize: 64, bits: 4))
                    let iters = 50
                    let start = Date()
                    for _ in 0 ..< iters {
                        eval(quantizedMM(
                            x, wq, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: 4))
                    }
                    let us = Date().timeIntervalSince(start) / Double(iters) * 1e6
                    print(
                        "[QMV-BENCH] \(path) \(label) M=\(m): "
                            + String(format: "%8.1f", us) + " us")
                }
            }
        }

        @Test("3-D verify-shaped input [1, M, K] routes correctly")
        func testQmvNB3D() throws {
            _ = Self.enableNB
            let K = 2560
            let N = 256
            let (wq, scales, biases) = quantized(
                ramp([N, K], scale: 0.41, dtype: .bfloat16), groupSize: 64, bits: 4)

            for m in 2 ... 4 {
                let x = ramp([1, m, K], scale: 0.13, dtype: .bfloat16)
                let y = quantizedMM(
                    x, wq, scales: scales, biases: biases, transpose: true,
                    groupSize: 64, bits: 4)
                var rows = [MLXArray]()
                for r in 0 ..< m {
                    rows.append(
                        quantizedMM(
                            x[0..., r ..< (r + 1), 0...], wq, scales: scales,
                            biases: biases, transpose: true, groupSize: 64, bits: 4))
                }
                let yRef = concatenated(rows, axis: 1)
                eval(y, yRef)
                #expect(
                    allClose(y, yRef, rtol: 1e-2, atol: 0.5).item(Bool.self),
                    "3-D kernel mismatch at M=\(m)")
            }
        }
    }
}

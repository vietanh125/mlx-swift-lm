import Foundation
import MLX
import MLXNN

/// A Linear layer that dynamically decodes block-scaled FP8 weights on the fly
/// using a fused Metal GEMV kernel for decoding (batch = 1) and lazy MLX 
/// operations for prefill (batch > 1).
/// This avoids the 2x memory blowup of eagerly converting FP8 to bfloat16.
public class FP8Linear: Module, @unchecked Sendable {
    public let weight: MLXArray
    public let weightScaleInv: MLXArray
    public let bias: MLXArray?
    
    public let inputDims: Int
    public let outputDims: Int
    public let blockSize: Int
    
    private let customGemv: ([MLXArray]) -> [MLXArray]
    
    public init(weight: MLXArray, weightScaleInv: MLXArray, bias: MLXArray? = nil, blockSize: Int = 128) {
        self.weight = weight
        self.weightScaleInv = weightScaleInv
        self.bias = bias
        self.inputDims = weight.dim(1)
        self.outputDims = weight.dim(0)
        self.blockSize = blockSize
        
        // Compile the custom GEMV kernel for this specific layer's dimensions
        let metalSource = """
        #include <metal_stdlib>
        using namespace metal;

        inline float decode_fp8_e4m3(uint8_t byte) {
            if (byte == 0) return 0.0f;
            if (byte == 0x80) return -0.0f;
            uint s = (byte >> 7) & 1;
            uint e = (byte >> 3) & 0xF;
            uint m = byte & 0x7;
            float sign = s ? -1.0f : 1.0f;
            if (e == 0) {
                return sign * exp2(-6.0f) * (m / 8.0f);
            }
            if (e == 15 && m == 7) return sign * NAN;
            return sign * exp2(float(e) - 7.0f) * (1.0f + m / 8.0f);
        }

        kernel void fp8_gemv(
            device const bfloat *x [[buffer(0)]],
            device const uint8_t *w [[buffer(1)]],
            device const bfloat *scales [[buffer(2)]],
            device bfloat *out [[buffer(3)]],
            uint tg_idx [[threadgroup_position_in_grid]],
            uint ti_idx [[thread_position_in_threadgroup]],
            uint tg_size [[threads_per_threadgroup]]
        ) {
            int row = tg_idx;
            if (row >= OUT_DIM) return;
            
            int scale_cols = (IN_DIM + BS - 1) / BS;
            
            float sum = 0.0f;
            for (int col = ti_idx; col < IN_DIM; col += tg_size) {
                int scale_idx = (row / BS) * scale_cols + (col / BS);
                float scale_val = (float)scales[scale_idx];
                
                uint8_t w_byte = w[row * IN_DIM + col];
                float w_val = decode_fp8_e4m3(w_byte) * scale_val;
                float x_val = (float)x[col];
                
                sum += w_val * x_val;
            }
            
            threadgroup float shared_sum[1024];
            shared_sum[ti_idx] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                if (ti_idx < stride) {
                    shared_sum[ti_idx] += shared_sum[ti_idx + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (ti_idx == 0) {
                out[row] = (bfloat)shared_sum[0];
            }
        }
        """
        
        let inDim = weight.dim(1)
        let outDim = weight.dim(0)
        let bs = blockSize
        
        let actualSource = """
        #define IN_DIM \(inDim)
        #define OUT_DIM \(outDim)
        #define BS \(bs)
        
        """ + metalSource
        
        let kernel = MLXFast.metalKernel(
            name: "fp8_gemv",
            inputNames: ["x", "w", "scales"],
            outputNames: ["out"],
            source: actualSource
        )
        
        self.customGemv = CustomFunction {
            Forward { inputs in
                let x = inputs[0]
                let w = inputs[1]
                let scales = inputs[2]
                
                let outShape = [1, outDim]
                let result = kernel(
                    [x, w, scales],
                    grid: (outDim, 1, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [outShape],
                    outputDTypes: [x.dtype]
                )
                return result
            }
            VJP { primals, cotangents in
                return primals.map { MLXArray.zeros(like: $0) }
            }
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out: MLXArray
        
        // Use custom GEMV for single-token decoding to avoid graph overhead
        // x shape is typically [1, inDim] or [inDim] or [B, 1, inDim]
        let isDecoding = x.size == inputDims
        
        if isDecoding {
            let xFlat = x.reshaped([1, inputDims])
            out = customGemv([xFlat, weight, weightScaleInv])[0]
            out = out.reshaped(Array(x.shape.dropLast()) + [outputDims])
        } else {
            // For prefill (multi-token), use native MLX graph. 
            // It uses highly optimized MPS GEMM kernels. Memory is freed after the layer.
            let wFp = MLXFast.fromFp8(weight, dtype: x.dtype)
            let (m, n) = (wFp.dim(0), wFp.dim(1))
            let padB = (blockSize - m % blockSize) % blockSize
            let padS = (blockSize - n % blockSize) % blockSize
            
            var padded = MLX.padded(wFp, widths: [[0, padB], [0, padS]])
            padded = padded.reshaped([(m + padB) / blockSize, blockSize, (n + padS) / blockSize, blockSize])
            let scaled = padded * weightScaleInv[0..., .newAxis, 0..., .newAxis]
            let dequantized = scaled.reshaped([m + padB, n + padS])[0 ..< m, 0 ..< n]
            
            out = MLX.matmul(x, dequantized.T)
        }
        
        if let bias = bias {
            out = out + bias
        }
        return out
    }
}

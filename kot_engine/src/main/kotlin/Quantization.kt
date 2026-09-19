/*
 * Quantization.kt - INT8 per-row (per-output-channel) symmetric quantization.
 *
 * Same idea as llama.cpp's Q8_0: each row of a weight matrix (one output
 * neuron's incoming weights) gets its own float scale. Values are stored as
 * signed bytes in [-127, 127]; to recover the original weight, multiply by
 * that row's scale. This keeps quantization error localized per-row instead
 * of using one scale for a whole tensor, which matters a lot for quality
 * once you get down to 8 bits.
 *
 * Memory: 1 byte/weight + 4 bytes/row (scale) vs 4 bytes/weight for F32,
 * i.e. ~4x smaller, and 2x smaller than the BF16 the checkpoint ships as.
 * This is what makes a 1B-parameter model fit in ~1.2GB instead of ~5GB.
 */

import kotlin.math.abs
import kotlin.math.roundToInt

/** A weight matrix stored as row-quantized INT8 with one FP32 scale per row. */
class QuantizedMatrix(
    val data: ByteArray,      // outDim * inDim bytes, row-major
    val scales: FloatArray,   // one scale per row (length outDim)
    val outDim: Int,
    val inDim: Int
) {
    companion object {
        /** Quantize a row-major F32 matrix (outDim x inDim) into INT8 + per-row scales. */
        fun fromFloatArray(src: FloatArray, outDim: Int, inDim: Int): QuantizedMatrix {
            require(src.size == outDim * inDim) { "shape mismatch: ${src.size} vs $outDim x $inDim" }
            val data = ByteArray(outDim * inDim)
            val scales = FloatArray(outDim)
            for (row in 0 until outDim) {
                val base = row * inDim
                var maxAbs = 0.0f
                for (i in 0 until inDim) {
                    val v = abs(src[base + i])
                    if (v > maxAbs) maxAbs = v
                }
                // guard against an all-zero row (avoid divide-by-zero)
                val scale = if (maxAbs > 1e-12f) maxAbs / 127.0f else 1.0f
                scales[row] = scale
                val invScale = if (scale > 0f) 1.0f / scale else 0.0f
                for (i in 0 until inDim) {
                    val q = (src[base + i] * invScale).roundToInt().coerceIn(-127, 127)
                    data[base + i] = q.toByte()
                }
            }
            return QuantizedMatrix(data, scales, outDim, inDim)
        }
    }
}

/**
 * A weight tensor that is either kept as plain F32 or as an INT8-quantized
 * matrix. Every model engine should go through this instead of holding raw
 * FloatArrays directly, so the same forward-pass code works regardless of
 * which precision was chosen at load time.
 */
sealed class Weight {
    class Dense(val values: FloatArray) : Weight()
    class Quantized(val matrix: QuantizedMatrix) : Weight()

    companion object {
        fun of(values: FloatArray, outDim: Int, inDim: Int, mode: WeightFormat): Weight = when (mode) {
            WeightFormat.F32 -> Dense(values)
            WeightFormat.INT8 -> Quantized(QuantizedMatrix.fromFloatArray(values, outDim, inDim))
        }

        /** For 1-D tensors (norms, biases) — always kept as F32, they're tiny and precision-sensitive. */
        fun vector(values: FloatArray): Weight = Dense(values)
    }
}

enum class WeightFormat { F32, INT8 }

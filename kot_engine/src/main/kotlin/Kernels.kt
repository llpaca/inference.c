/*
 * Kernels.kt - core math kernels shared by every model engine.
 *
 * Mirrors the free functions in llama.c (rmsnorm, matmul via cblas_sgemv,
 * softmax, silu, rope). We don't have BLAS here, so matmul is a plain
 * row-major mat-vec loop, parallelized across output rows with a shared
 * thread pool -- this is the single biggest lever for CPU utilization since
 * matmul dominates the forward pass and is embarrassingly parallel over rows.
 *
 * Two matmul paths:
 *   - matmul()       : F32 weights, plain dot product
 *   - matmulQuantized(): INT8 weights (see Quantization.kt), dequantized
 *                        on the fly one row at a time -- the weight matrix
 *                        itself never exists as a full F32 copy in memory.
 */

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.CountDownLatch

object Kernels {

    /** Shared pool sized to the machine's core count; reused across every matmul call.
     *  Daemon threads so the JVM can exit normally once main() returns. */
    val threadPool: ExecutorService = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    ) { r -> Thread(r).apply { isDaemon = true } }
    private val nThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    fun rmsnorm(out: FloatArray, x: FloatArray, weight: FloatArray, dim: Int, eps: Float = 1e-5f) {
        var ss = 0.0
        for (i in 0 until dim) ss += x[i].toDouble() * x[i].toDouble()
        val scale = (1.0 / sqrt(ss / dim + eps)).toFloat()
        for (i in 0 until dim) out[i] = weight[i] * (scale * x[i])
    }

    /** Runs `body(rowStart, rowEndExclusive)` across the thread pool, splitting [0, outDim) into chunks. */
    private inline fun parallelRows(outDim: Int, crossinline body: (Int, Int) -> Unit) {
        if (outDim < 64 || nThreads == 1) {
            // not worth threading overhead for tiny outputs (e.g. norm-sized vectors)
            body(0, outDim)
            return
        }
        val chunks = nThreads.coerceAtMost(outDim)
        val chunkSize = (outDim + chunks - 1) / chunks
        val latch = CountDownLatch(chunks)
        for (c in 0 until chunks) {
            val start = c * chunkSize
            val end = minOf(start + chunkSize, outDim)
            if (start >= end) { latch.countDown(); continue }
            threadPool.execute {
                try { body(start, end) } finally { latch.countDown() }
            }
        }
        latch.await()
    }

    /**
     * out[outDim] = W[outDim x inDim] (row-major) . x[inDim]
     * Plain mat-vec, parallelized across output rows. beta=0 overwrites,
     * beta=1 accumulates into `out`.
     */
    fun matmul(out: FloatArray, x: FloatArray, w: FloatArray, outDim: Int, inDim: Int, accumulate: Boolean = false) {
        parallelRows(outDim) { rowStart, rowEnd ->
            for (o in rowStart until rowEnd) {
                var sum = 0.0f
                val rowBase = o * inDim
                var i = 0
                val limit = inDim - (inDim % 4)
                while (i < limit) {
                    sum += w[rowBase + i] * x[i] +
                           w[rowBase + i + 1] * x[i + 1] +
                           w[rowBase + i + 2] * x[i + 2] +
                           w[rowBase + i + 3] * x[i + 3]
                    i += 4
                }
                while (i < inDim) { sum += w[rowBase + i] * x[i]; i++ }
                out[o] = if (accumulate) out[o] + sum else sum
            }
        }
    }

    /**
     * Same as matmul() but the weight matrix is INT8-quantized (see
     * QuantizedMatrix): each row's bytes are dequantized (multiplied by that
     * row's scale) on the fly inside the dot product. No F32 copy of the
     * weight matrix is ever created -- this is what keeps memory low.
     */
    fun matmulQuantized(out: FloatArray, x: FloatArray, qw: QuantizedMatrix, accumulate: Boolean = false) {
        val inDim = qw.inDim
        val data = qw.data
        val scales = qw.scales
        parallelRows(qw.outDim) { rowStart, rowEnd ->
            for (o in rowStart until rowEnd) {
                val rowBase = o * inDim
                val scale = scales[o]
                var sum = 0.0f
                var i = 0
                val limit = inDim - (inDim % 4)
                while (i < limit) {
                    sum += data[rowBase + i] * x[i] +
                           data[rowBase + i + 1] * x[i + 1] +
                           data[rowBase + i + 2] * x[i + 2] +
                           data[rowBase + i + 3] * x[i + 3]
                    i += 4
                }
                while (i < inDim) { sum += data[rowBase + i] * x[i]; i++ }
                val result = sum * scale
                out[o] = if (accumulate) out[o] + result else result
            }
        }
    }

    /** Dispatches to the plain or quantized matmul based on which form the weight is in. */
    fun matmulDispatch(out: FloatArray, x: FloatArray, w: Weight, outDim: Int, inDim: Int, accumulate: Boolean = false) {
        when (w) {
            is Weight.Dense -> matmul(out, x, w.values, outDim, inDim, accumulate)
            is Weight.Quantized -> matmulQuantized(out, x, w.matrix, accumulate)
        }
    }

    fun softmax(x: FloatArray, n: Int) {
        var max = x[0]
        for (i in 1 until n) if (x[i] > max) max = x[i]
        var sum = 0.0f
        for (i in 0 until n) { x[i] = exp(x[i] - max); sum += x[i] }
        for (i in 0 until n) x[i] /= sum
    }

    fun silu(x: FloatArray, n: Int) {
        for (i in 0 until n) x[i] = x[i] / (1.0f + exp(-x[i]))
    }

    /**
     * Standard (non-scaled) RoPE, in-place on q and k for one position.
     * head_dim is split into [0, half) / [half, head_dim) rotation pairs
     * (the "neox"/half-split style Llama uses), matching llama.c's rope().
     */
    fun ropeInPlace(
        q: FloatArray, k: FloatArray, headDim: Int, nHeads: Int, nKvHeads: Int,
        pos: Int, invFreq: FloatArray
    ) {
        val half = headDim / 2
        for (h in 0 until nHeads) {
            val base = h * headDim
            for (i in 0 until half) {
                val angle = pos * invFreq[i]
                val c = cos(angle); val s = sin(angle)
                val q0 = q[base + i]; val q1 = q[base + i + half]
                q[base + i] = q0 * c - q1 * s
                q[base + i + half] = q0 * s + q1 * c
            }
        }
        for (h in 0 until nKvHeads) {
            val base = h * headDim
            for (i in 0 until half) {
                val angle = pos * invFreq[i]
                val c = cos(angle); val s = sin(angle)
                val k0 = k[base + i]; val k1 = k[base + i + half]
                k[base + i] = k0 * c - k1 * s
                k[base + i + half] = k0 * s + k1 * c
            }
        }
    }

    /**
     * Precompute inverse frequencies for RoPE, with optional Llama-3 style
     * frequency rescaling ("rope_type": "llama3" in config.json), which plain
     * llama.c does NOT implement but real Llama-3.1/3.2 checkpoints require
     * for correct long-context behavior.
     *
     * Reference: HF `_compute_llama3_parameters` in transformers/modeling_rope_utils.py
     */
    fun computeInvFreq(
        headDim: Int,
        theta: Float,
        ropeScaling: RopeScaling? = null
    ): FloatArray {
        val half = headDim / 2
        val invFreq = FloatArray(half) { i -> (1.0 / Math.pow(theta.toDouble(), (2.0 * i) / headDim)).toFloat() }
        if (ropeScaling == null) return invFreq

        val factor = ropeScaling.factor
        val lowFreqFactor = ropeScaling.lowFreqFactor
        val highFreqFactor = ropeScaling.highFreqFactor
        val oldContextLen = ropeScaling.originalMaxPositionEmbeddings.toFloat()

        val lowFreqWavelen = oldContextLen / lowFreqFactor
        val highFreqWavelen = oldContextLen / highFreqFactor

        for (i in invFreq.indices) {
            val freq = invFreq[i]
            val wavelen = (2.0 * Math.PI / freq).toFloat()
            invFreq[i] = when {
                wavelen < highFreqWavelen -> freq // high freq: unchanged
                wavelen > lowFreqWavelen -> freq / factor // low freq: scaled down
                else -> {
                    // smooth interpolation between the two regimes
                    val smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
                    (1 - smooth) * freq / factor + smooth * freq
                }
            }
        }
        return invFreq
    }
}

/** Mirrors config.json's "rope_scaling" block for Llama-3.x "llama3" rope_type. */
data class RopeScaling(
    val factor: Float,
    val lowFreqFactor: Float,
    val highFreqFactor: Float,
    val originalMaxPositionEmbeddings: Int
)

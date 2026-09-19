/*
 * LlamaEngine.kt - Kotlin inference engine for Llama 3.2 (1B / 3B, Instruct or base)
 *
 * Direct port of the architecture in inference.c's llama.c:
 *   - SafeTensors weight loading (F32 / F16 / BF16 -> F32)
 *   - RMSNorm, RoPE (with correct Llama-3 frequency rescaling), GQA
 *   - SwiGLU FFN
 *   - Greedy & temperature+top-p sampling
 *   - KV cache
 *
 * Usage (after `./build.sh`):
 *   java -jar inference.jar llama <model_dir> "<prompt>" [maxTokens] [temperature]
 *
 * model_dir must contain:
 *   model.safetensors (or model-00001-of-0000N.safetensors, first shard;
 *                       for sharded models pass the *directory* and this will
 *                       look for an index file too)
 *   tokenizer.json
 *   config.json
 */

import java.io.File
import kotlin.math.sqrt
import kotlin.random.Random

/** Mirrors config.json fields we actually need for the forward pass. */
data class LlamaConfig(
    val dim: Int,
    val nLayers: Int,
    val nHeads: Int,
    val nKvHeads: Int,
    val ffDim: Int,
    val vocabSize: Int,
    val maxSeqLen: Int,
    val ropeTheta: Float,
    val headDim: Int,
    val rmsNormEps: Float,
    val ropeScaling: RopeScaling?,
    val tieWordEmbeddings: Boolean
) {
    val gqaFactor: Int get() = nHeads / nKvHeads

    companion object {
        /** Known-good defaults, used when config.json is missing or partially specified. */
        fun llama32_1b() = LlamaConfig(
            dim = 2048, nLayers = 16, nHeads = 32, nKvHeads = 8, ffDim = 8192,
            vocabSize = 128256, maxSeqLen = 4096, ropeTheta = 500000f, headDim = 64,
            rmsNormEps = 1e-5f,
            ropeScaling = RopeScaling(32f, 1f, 4f, 8192),
            tieWordEmbeddings = true
        )

        fun llama32_3b() = LlamaConfig(
            dim = 3072, nLayers = 28, nHeads = 24, nKvHeads = 8, ffDim = 8192,
            vocabSize = 128256, maxSeqLen = 4096, ropeTheta = 500000f, headDim = 128,
            rmsNormEps = 1e-5f,
            ropeScaling = RopeScaling(32f, 1f, 4f, 8192),
            tieWordEmbeddings = true
        )

        /** Parse a real HuggingFace config.json, falling back to the given defaults for missing fields. */
        fun fromConfigJson(path: String, fallback: LlamaConfig, maxSeqLenCap: Int): LlamaConfig {
            if (!File(path).exists()) {
                println("[config] $path not found, using built-in defaults")
                return fallback
            }
            val text = File(path).readText()
            val (root, _) = MiniJson.parseObject(text, 0)

            fun i(key: String, def: Int): Int = (root[key] as? Double)?.toInt() ?: def
            fun f(key: String, def: Float): Float = (root[key] as? Double)?.toFloat() ?: def
            fun b(key: String, def: Boolean): Boolean = (root[key] as? Boolean) ?: def

            val dim = i("hidden_size", fallback.dim)
            val nHeads = i("num_attention_heads", fallback.nHeads)
            val headDim = i("head_dim", dim / nHeads)
            val nLayers = i("num_hidden_layers", fallback.nLayers)
            val nKvHeads = i("num_key_value_heads", fallback.nKvHeads)
            val ffDim = i("intermediate_size", fallback.ffDim)
            val vocabSize = i("vocab_size", fallback.vocabSize)
            val ropeTheta = f("rope_theta", fallback.ropeTheta)
            val rmsEps = f("rms_norm_eps", fallback.rmsNormEps)
            val tie = b("tie_word_embeddings", fallback.tieWordEmbeddings)
            val rawMaxPos = i("max_position_embeddings", fallback.maxSeqLen)
            val maxSeq = minOf(rawMaxPos, maxSeqLenCap)

            var ropeScaling = fallback.ropeScaling
            @Suppress("UNCHECKED_CAST")
            val rs = root["rope_scaling"] as? Map<String, Any?>
            if (rs != null) {
                val type = rs["rope_type"] as? String ?: rs["type"] as? String
                if (type == "llama3") {
                    ropeScaling = RopeScaling(
                        factor = (rs["factor"] as Double).toFloat(),
                        lowFreqFactor = (rs["low_freq_factor"] as Double).toFloat(),
                        highFreqFactor = (rs["high_freq_factor"] as Double).toFloat(),
                        originalMaxPositionEmbeddings = (rs["original_max_position_embeddings"] as Double).toInt()
                    )
                } else {
                    ropeScaling = null
                }
            }

            return LlamaConfig(
                dim = dim, nLayers = nLayers, nHeads = nHeads, nKvHeads = nKvHeads,
                ffDim = ffDim, vocabSize = vocabSize, maxSeqLen = maxSeq,
                ropeTheta = ropeTheta, headDim = headDim, rmsNormEps = rmsEps,
                ropeScaling = ropeScaling, tieWordEmbeddings = tie
            )
        }
    }
}

/** All transformer weights, resident as FloatArrays after being copied out of the mmap'd safetensors file. */
class LlamaWeights(private val st: SafeTensors, cfg: LlamaConfig, private val format: WeightFormat) {
    // Embedding and lm_head stay F32: they're gathered/looked-up by row rather than
    // used in a full matmul reduction, and the embedding table dominates disk size
    // less than you'd think relative to the win from quantizing every layer matrix.
    // (Quantizing them too is a reasonable follow-up if memory is still tight.)
    val embedTokens: FloatArray = st.getFloatArray("model.embed_tokens.weight")

    private fun loadLinear(name: String, outDim: Int, inDim: Int): Weight {
        val raw = st.getFloatArray(name)
        return Weight.of(raw, outDim, inDim, format)
    }

    val attnQ = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.self_attn.q_proj.weight", cfg.nHeads * cfg.headDim, cfg.dim) }
    val attnK = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.self_attn.k_proj.weight", cfg.nKvHeads * cfg.headDim, cfg.dim) }
    val attnV = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.self_attn.v_proj.weight", cfg.nKvHeads * cfg.headDim, cfg.dim) }
    val attnO = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.self_attn.o_proj.weight", cfg.dim, cfg.nHeads * cfg.headDim) }
    val ffGate = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.mlp.gate_proj.weight", cfg.ffDim, cfg.dim) }
    val ffUp = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.mlp.up_proj.weight", cfg.ffDim, cfg.dim) }
    val ffDown = Array(cfg.nLayers) { l -> loadLinear("model.layers.$l.mlp.down_proj.weight", cfg.dim, cfg.ffDim) }
    val attnNorm = Array(cfg.nLayers) { l -> st.getFloatArray("model.layers.$l.input_layernorm.weight") }
    val ffNorm = Array(cfg.nLayers) { l -> st.getFloatArray("model.layers.$l.post_attention_layernorm.weight") }
    val norm: FloatArray = st.getFloatArray("model.norm.weight")
    val lmHead: FloatArray = if (st.has("lm_head.weight")) {
        st.getFloatArray("lm_head.weight").also { println("[weights] lm_head: separate tensor") }
    } else {
        println("[weights] lm_head: tied to embed_tokens")
        embedTokens
    }
}

/** Scratch buffers reused across every forward() call, plus the KV cache. */
class LlamaRunState(cfg: LlamaConfig) {
    val qDim = cfg.nHeads * cfg.headDim
    val kvDim = cfg.nKvHeads * cfg.headDim

    val x = FloatArray(cfg.dim)
    val xb = FloatArray(cfg.dim)
    val q = FloatArray(qDim)
    val k = FloatArray(kvDim)
    val v = FloatArray(kvDim)
    val att = FloatArray(cfg.nHeads * cfg.maxSeqLen)
    val logits = FloatArray(cfg.vocabSize)
    val ffBuf = FloatArray(cfg.ffDim)
    val ffBuf2 = FloatArray(cfg.ffDim)
    val attnOut = FloatArray(cfg.dim)

    // kCache[layer][pos * kvDim + i]
    val kCache = Array(cfg.nLayers) { FloatArray(cfg.maxSeqLen * kvDim) }
    val vCache = Array(cfg.nLayers) { FloatArray(cfg.maxSeqLen * kvDim) }

    val invFreq = Kernels.computeInvFreq(cfg.headDim, cfg.ropeTheta, cfg.ropeScaling)
}

class LlamaEngine(private val cfg: LlamaConfig, private val w: LlamaWeights) {
    private val state = LlamaRunState(cfg)
    private val scale = 1.0f / sqrt(cfg.headDim.toFloat())

    /** Runs one token through all layers; returns the logits array (reused buffer, don't retain). */
    fun forward(token: Int, pos: Int): FloatArray {
        val dim = cfg.dim
        val s = state

        // 1. embedding lookup
        System.arraycopy(w.embedTokens, token * dim, s.x, 0, dim)

        for (l in 0 until cfg.nLayers) {
            // attention pre-norm
            Kernels.rmsnorm(s.xb, s.x, w.attnNorm[l], dim, cfg.rmsNormEps)

            // QKV projections
            Kernels.matmulDispatch(s.q, s.xb, w.attnQ[l], s.qDim, dim)
            Kernels.matmulDispatch(s.k, s.xb, w.attnK[l], s.kvDim, dim)
            Kernels.matmulDispatch(s.v, s.xb, w.attnV[l], s.kvDim, dim)

            // RoPE
            Kernels.ropeInPlace(s.q, s.k, cfg.headDim, cfg.nHeads, cfg.nKvHeads, pos, s.invFreq)

            // store into KV cache
            System.arraycopy(s.k, 0, s.kCache[l], pos * s.kvDim, s.kvDim)
            System.arraycopy(s.v, 0, s.vCache[l], pos * s.kvDim, s.kvDim)

            // grouped-query attention
            java.util.Arrays.fill(s.attnOut, 0f)
            for (h in 0 until cfg.nHeads) {
                val kvHead = h / cfg.gqaFactor
                val qBase = h * cfg.headDim
                val attBase = h * cfg.maxSeqLen

                for (t in 0..pos) {
                    val kBase = t * s.kvDim + kvHead * cfg.headDim
                    var dot = 0.0f
                    for (i in 0 until cfg.headDim) dot += s.q[qBase + i] * s.kCache[l][kBase + i]
                    s.att[attBase + t] = dot * scale
                }
                // softmax over [0, pos]
                softmaxRange(s.att, attBase, pos + 1)

                val outBase = h * cfg.headDim
                for (t in 0..pos) {
                    val vBase = t * s.kvDim + kvHead * cfg.headDim
                    val a = s.att[attBase + t]
                    for (i in 0 until cfg.headDim) s.attnOut[outBase + i] += a * s.vCache[l][vBase + i]
                }
            }

            // output projection + residual
            val projOut = FloatArray(dim)
            Kernels.matmulDispatch(projOut, s.attnOut, w.attnO[l], dim, s.qDim)
            for (i in 0 until dim) s.x[i] += projOut[i]

            // FFN pre-norm
            Kernels.rmsnorm(s.xb, s.x, w.ffNorm[l], dim, cfg.rmsNormEps)

            // SwiGLU FFN
            Kernels.matmulDispatch(s.ffBuf, s.xb, w.ffGate[l], cfg.ffDim, dim)
            Kernels.matmulDispatch(s.ffBuf2, s.xb, w.ffUp[l], cfg.ffDim, dim)
            Kernels.silu(s.ffBuf, cfg.ffDim)
            for (i in 0 until cfg.ffDim) s.ffBuf[i] *= s.ffBuf2[i]
            Kernels.matmulDispatch(projOut, s.ffBuf, w.ffDown[l], dim, cfg.ffDim)
            for (i in 0 until dim) s.x[i] += projOut[i]
        }

        // final norm + LM head
        Kernels.rmsnorm(s.x, s.x, w.norm, dim, cfg.rmsNormEps)
        Kernels.matmul(s.logits, s.x, w.lmHead, cfg.vocabSize, dim)
        return s.logits
    }

    private fun softmaxRange(x: FloatArray, offset: Int, n: Int) {
        var max = x[offset]
        for (i in 1 until n) if (x[offset + i] > max) max = x[offset + i]
        var sum = 0.0f
        for (i in 0 until n) { x[offset + i] = kotlin.math.exp(x[offset + i] - max); sum += x[offset + i] }
        for (i in 0 until n) x[offset + i] /= sum
    }

    val config: LlamaConfig get() = cfg
}

/** Greedy or temperature sampling, mirroring sample_argmax/sample_temperature in llama.c. */
object Sampler {
    fun argmax(logits: FloatArray): Int {
        var best = 0
        for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
        return best
    }

    fun temperature(logits: FloatArray, temp: Float, rng: Random): Int {
        val n = logits.size
        val scaled = FloatArray(n) { logits[it] / temp }
        Kernels.softmax(scaled, n)
        val r = rng.nextFloat()
        var cumsum = 0.0f
        for (i in 0 until n) {
            cumsum += scaled[i]
            if (r < cumsum) return i
        }
        return n - 1
    }
}

private fun nowSec(): Double = System.nanoTime() / 1e9

fun runLlama(args: List<String>) {
    if (args.size < 2) {
        System.err.println(
            """
            Usage: llama <model_dir> "<prompt>" [maxTokens] [temperature] [--size 1b|3b] [--weights f32|int8]

              model_dir    directory with model.safetensors, tokenizer.json, config.json
              maxTokens    default 200
              temperature  0 = greedy, >0 = sampling (default 0)
              --size       1b (default) or 3b — used only as a fallback if config.json is absent
              --weights    int8 (default) or f32 — int8 cuts memory ~4x vs f32, ~2x vs the
                           on-disk bf16 checkpoint, with a small quality tradeoff. Use f32
                           only if you specifically need maximum fidelity and have the RAM.
            """.trimIndent()
        )
        return
    }

    val modelDir = args[0]
    val prompt = args[1]
    val maxTokens = args.getOrNull(2)?.toIntOrNull() ?: 200
    val temperature = args.getOrNull(3)?.toFloatOrNull() ?: 0.0f
    val sizeFlagIdx = args.indexOf("--size")
    val sizeHint = if (sizeFlagIdx >= 0 && sizeFlagIdx + 1 < args.size) args[sizeFlagIdx + 1] else "1b"
    val weightsFlagIdx = args.indexOf("--weights")
    val weightsHint = if (weightsFlagIdx >= 0 && weightsFlagIdx + 1 < args.size) args[weightsFlagIdx + 1] else "int8"
    val weightFormat = when (weightsHint.lowercase()) {
        "f32", "fp32" -> WeightFormat.F32
        "int8", "i8" -> WeightFormat.INT8
        else -> error("Unknown --weights value '$weightsHint', expected f32 or int8")
    }

    println("+----------------------------------------+")
    println("|  LlamaEngine.kt  (Kotlin/JVM)           |")
    println("+----------------------------------------+")
    println("[runtime] cpu cores available: ${Runtime.getRuntime().availableProcessors()}")
    println("[runtime] weight format: $weightFormat")

    val fallback = if (sizeHint == "3b") LlamaConfig.llama32_3b() else LlamaConfig.llama32_1b()
    val cfg = LlamaConfig.fromConfigJson("$modelDir/config.json", fallback, maxSeqLenCap = 4096)
    println("[config] dim=${cfg.dim} layers=${cfg.nLayers} heads=${cfg.nHeads} kv_heads=${cfg.nKvHeads} " +
            "ff=${cfg.ffDim} vocab=${cfg.vocabSize} head_dim=${cfg.headDim} max_seq=${cfg.maxSeqLen}")

    val stPath = resolveSafetensorsPath(modelDir)
    println("[safetensors] loading $stPath")
    val loadStart = nowSec()
    val st = SafeTensors.open(stPath)
    println("[safetensors] header parsed: ${st.tensors.size} tensors (${"%.2f".format(nowSec() - loadStart)}s)")

    val weightLoadStart = nowSec()
    val weights = LlamaWeights(st, cfg, weightFormat)
    val rt = Runtime.getRuntime()
    System.gc()
    val usedMb = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)
    println("[weights] all weights materialized in ${"%.2f".format(nowSec() - weightLoadStart)}s " +
            "(JVM heap in use: ~${usedMb} MB)")

    val tokPath = "$modelDir/tokenizer.json"
    val tok = BpeTokenizer.load(tokPath)

    val engine = LlamaEngine(cfg, weights)

    val promptIds = ArrayList<Int>()
    promptIds.add(tok.bosId)
    promptIds.addAll(tok.encode(prompt))
    println("[encode] prompt tokens: ${promptIds.size}")
    print("\n--- Output ---\n")
    print(prompt)
    System.out.flush()

    val rng = Random(System.nanoTime())
    var nextToken = 0
    val total = minOf(promptIds.size + maxTokens, cfg.maxSeqLen)

    val prefillStart = nowSec()
    var firstTokenTime = 0.0
    var generated = 0

    for (pos in 0 until total) {
        val token = if (pos < promptIds.size) promptIds[pos] else nextToken
        val logits = engine.forward(token, pos)

        if (pos == promptIds.size - 1) firstTokenTime = nowSec()
        if (pos < promptIds.size - 1) continue

        nextToken = if (temperature <= 0.0f) Sampler.argmax(logits) else Sampler.temperature(logits, temperature, rng)

        if (tok.isEos(nextToken)) {
            print("\n[EOS]\n")
            break
        }

        print(tok.decodeToken(nextToken))
        System.out.flush()
        generated++
    }

    val end = nowSec()
    val prefillTime = firstTokenTime - prefillStart
    val genTime = end - firstTokenTime
    val tokPerSec = if (genTime > 0 && generated > 0) generated / genTime else 0.0

    println("\n\n========================================")
    println("  Prompt tokens : ${promptIds.size}")
    println("  Generated     : $generated tokens")
    println("  Prefill time  : ${"%.2f".format(prefillTime)} s (${"%.1f".format(if (prefillTime > 0) promptIds.size / prefillTime else 0.0)} tok/s)")
    println("  Generate time : ${"%.2f".format(genTime)} s")
    println("  Throughput    : ${"%.2f".format(tokPerSec)} tok/s")
    println("========================================")
}

private fun resolveSafetensorsPath(modelDir: String): String {
    val candidates = listOf(
        "model.safetensors",
        "model-00001-of-00001.safetensors",
        "model-00001-of-00002.safetensors"
    )
    for (c in candidates) {
        val f = File(modelDir, c)
        if (f.exists()) return f.path
    }
    val direct = File(modelDir)
    if (direct.isFile) return direct.path
    error(
        "Could not find model weights in $modelDir\n" +
        "Expected one of: ${candidates.joinToString(", ")}\n" +
        "Or pass the .safetensors file path directly."
    )
}

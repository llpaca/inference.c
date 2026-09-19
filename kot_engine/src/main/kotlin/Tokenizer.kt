/*
 * Tokenizer.kt - byte-level BPE tokenizer for HuggingFace `tokenizer.json` files
 *
 * Same approach as the GPT-2-style byte encoder/decoder + BPE merge loop found
 * in inference.c's tok_load/tok_encode, re-hosted on the JVM using
 * java.util.regex instead of a hand-rolled pretokenizer, and a HashMap instead
 * of the custom open-addressing table (the JVM's HashMap is plenty fast for
 * vocab_size ~128k / prompt-sized inputs).
 *
 * Supports what Llama 3.x tokenizer.json files use: a byte-level BPE `model`
 * block with `vocab` + `merges`, plus `added_tokens` for specials (BOS/EOS/etc).
 */

import java.io.File

/** Tiny JSON parser fallback: we avoid an external dependency by hand-parsing
 *  just enough of tokenizer.json (vocab map, merges list, added_tokens list). */
object MiniJson {
    fun parseObject(s: String, from: Int): Pair<Map<String, Any?>, Int> {
        var i = skipWs(s, from)
        require(s[i] == '{')
        i++
        val map = LinkedHashMap<String, Any?>()
        while (true) {
            i = skipWs(s, i)
            if (s[i] == '}') { i++; break }
            if (s[i] == ',') { i++; continue }
            val (key, afterKey) = parseString(s, i)
            i = skipWs(s, afterKey)
            require(s[i] == ':')
            i++
            i = skipWs(s, i)
            val (value, afterValue) = parseValue(s, i)
            map[key] = value
            i = afterValue
        }
        return map to i
    }

    fun parseValue(s: String, from: Int): Pair<Any?, Int> {
        val i = skipWs(s, from)
        return when {
            s[i] == '"' -> parseString(s, i)
            s[i] == '{' -> parseObject(s, i)
            s[i] == '[' -> parseArray(s, i)
            s.startsWith("true", i) -> true to i + 4
            s.startsWith("false", i) -> false to i + 5
            s.startsWith("null", i) -> null to i + 4
            else -> parseNumber(s, i)
        }
    }

    private fun parseArray(s: String, from: Int): Pair<List<Any?>, Int> {
        var i = from
        require(s[i] == '[')
        i++
        val list = ArrayList<Any?>()
        while (true) {
            i = skipWs(s, i)
            if (s[i] == ']') { i++; break }
            if (s[i] == ',') { i++; continue }
            val (v, after) = parseValue(s, i)
            list.add(v)
            i = after
        }
        return list to i
    }

    private fun parseNumber(s: String, from: Int): Pair<Double, Int> {
        var i = from
        val begin = i
        if (s[i] == '-') i++
        while (i < s.length && (s[i].isDigit() || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) i++
        return s.substring(begin, i).toDouble() to i
    }

    fun parseString(s: String, from: Int): Pair<String, Int> {
        require(s[from] == '"')
        var i = from + 1
        val sb = StringBuilder()
        while (s[i] != '"') {
            if (s[i] == '\\') {
                i++
                when (s[i]) {
                    'n' -> sb.append('\n')
                    't' -> sb.append('\t')
                    'r' -> sb.append('\r')
                    '"' -> sb.append('"')
                    '\\' -> sb.append('\\')
                    '/' -> sb.append('/')
                    'b' -> sb.append('\b')
                    'u' -> {
                        val hex = s.substring(i + 1, i + 5)
                        sb.append(hex.toInt(16).toChar())
                        i += 4
                    }
                    else -> sb.append(s[i])
                }
            } else sb.append(s[i])
            i++
        }
        return sb.toString() to (i + 1)
    }

    private fun skipWs(s: String, from: Int): Int {
        var i = from
        while (i < s.length && s[i].isWhitespace()) i++
        return i
    }
}

class BpeTokenizer private constructor(
    private val tokenToId: Map<String, Int>,
    private val idToToken: Array<String?>,
    private val merges: Map<Pair<String, String>, Int>,
    val bosId: Int,
    val eosIds: Set<Int>,
    private val byteEncoder: Map<Int, Char>,
    private val byteDecoder: Map<Char, Int>,
    private val specialIds: Set<Int>,
    private val specialByString: Map<String, Int>
) {
    companion object {
        /** GPT-2 style byte<->unicode mapping so every raw byte maps to a printable char. */
        private fun buildByteEncoder(): Pair<Map<Int, Char>, Map<Char, Int>> {
            val bs = ArrayList<Int>()
            bs.addAll((33..126))
            bs.addAll((161..172))
            bs.addAll((174..255))
            val cs = ArrayList<Int>(bs)
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            val enc = HashMap<Int, Char>()
            val dec = HashMap<Char, Int>()
            for (idx in bs.indices) {
                enc[bs[idx]] = cs[idx].toChar()
                dec[cs[idx].toChar()] = bs[idx]
            }
            return enc to dec
        }

        fun load(path: String): BpeTokenizer {
            val text = File(path).readText(Charsets.UTF_8)
            val (root, _) = MiniJson.parseObject(text, 0)

            @Suppress("UNCHECKED_CAST")
            val model = root["model"] as Map<String, Any?>
            @Suppress("UNCHECKED_CAST")
            val vocabMap = model["vocab"] as Map<String, Any?>
            @Suppress("UNCHECKED_CAST")
            val mergesList = (model["merges"] as? List<Any?>) ?: emptyList()

            val tokenToId = HashMap<String, Int>()
            var maxId = 0
            for ((tok, id) in vocabMap) {
                val idInt = (id as Double).toInt()
                tokenToId[tok] = idInt
                if (idInt > maxId) maxId = idInt
            }

            val merges = HashMap<Pair<String, String>, Int>()
            for ((rank, m) in mergesList.withIndex()) {
                // merges entries can be "a b" strings or ["a","b"] arrays depending on tokenizer version
                val parts: Pair<String, String> = when (m) {
                    is String -> {
                        val sp = m.indexOf(' ')
                        m.substring(0, sp) to m.substring(sp + 1)
                    }
                    is List<*> -> (m[0] as String) to (m[1] as String)
                    else -> error("unrecognized merges entry: $m")
                }
                merges[parts] = rank
            }

            @Suppress("UNCHECKED_CAST")
            val addedTokens = (root["added_tokens"] as? List<Any?>) ?: emptyList()
            var bos = -1
            val eos = HashSet<Int>()
            val specialIds = HashSet<Int>()
            val specialByString = HashMap<String, Int>()
            for (at in addedTokens) {
                @Suppress("UNCHECKED_CAST")
                val m = at as Map<String, Any?>
                val id = (m["id"] as Double).toInt()
                val content = m["content"] as String
                tokenToId[content] = id
                if (id > maxId) maxId = id
                specialIds.add(id)
                specialByString[content] = id
                if (content == "<|begin_of_text|>") bos = id
                if (content == "<|end_of_text|>" || content == "<|eot_id|>" || content == "<|eom_id|>") eos.add(id)
            }
            if (bos == -1) bos = tokenToId["<|begin_of_text|>"] ?: 128000
            if (eos.isEmpty()) {
                tokenToId["<|end_of_text|>"]?.let { eos.add(it) }
                tokenToId["<|eot_id|>"]?.let { eos.add(it) }
                if (eos.isEmpty()) eos.add(128009)
            }

            val idToToken = arrayOfNulls<String>(maxId + 1)
            for ((tok, id) in tokenToId) idToToken[id] = tok

            val (enc, dec) = buildByteEncoder()

            return BpeTokenizer(tokenToId, idToToken, merges, bos, eos, enc, dec, specialIds, specialByString)
        }
    }

    /** Encode raw UTF-8 text into token ids using byte-level BPE. */
    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>()
        // Split on special tokens first so they never get merged into surrounding text.
        var remaining = text
        if (specialByString.isNotEmpty()) {
            val pattern = specialByString.keys.sortedByDescending { it.length }
                .joinToString("|") { Regex.escape(it) }
            val re = Regex(pattern)
            var lastEnd = 0
            for (match in re.findAll(remaining)) {
                if (match.range.first > lastEnd) {
                    ids.addAll(encodeChunk(remaining.substring(lastEnd, match.range.first)))
                }
                ids.add(specialByString[match.value]!!)
                lastEnd = match.range.last + 1
            }
            if (lastEnd < remaining.length) ids.addAll(encodeChunk(remaining.substring(lastEnd)))
        } else {
            ids.addAll(encodeChunk(remaining))
        }
        return ids
    }

    // GPT-2 pretokenizer regex (approximation good enough for Latin text / code / punctuation).
    private val pretokenRe = Regex(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    )

    private fun encodeChunk(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()
        val out = ArrayList<Int>()
        for (piece in pretokenRe.findAll(text).map { it.value }) {
            val bytes = piece.toByteArray(Charsets.UTF_8)
            var symbols = bytes.map { b -> byteEncoder[b.toInt() and 0xFF]!!.toString() }.toMutableList()
            symbols = bpeMerge(symbols)
            for (sym in symbols) {
                val id = tokenToId[sym]
                if (id != null) out.add(id)
                else {
                    // fall back to per-byte-char lookups if a merged symbol somehow isn't in vocab
                    for (ch in sym) out.add(tokenToId[ch.toString()] ?: 0)
                }
            }
        }
        return out
    }

    private fun bpeMerge(inputSymbols: MutableList<String>): MutableList<String> {
        var symbols = inputSymbols
        if (symbols.size < 2) return symbols
        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (idx in 0 until symbols.size - 1) {
                val pair = symbols[idx] to symbols[idx + 1]
                val rank = merges[pair] ?: continue
                if (rank < bestRank) { bestRank = rank; bestIdx = idx }
            }
            if (bestIdx == -1) break
            val merged = symbols[bestIdx] + symbols[bestIdx + 1]
            val next = ArrayList<String>(symbols.size - 1)
            next.addAll(symbols.subList(0, bestIdx))
            next.add(merged)
            next.addAll(symbols.subList(bestIdx + 2, symbols.size))
            symbols = next
        }
        return symbols
    }

    /** Decode a single token id back to its raw UTF-8 string piece. */
    fun decodeToken(id: Int): String {
        val tok = idToToken.getOrNull(id) ?: return ""
        if (id in specialIds) return "" // don't print specials during streaming generation
        val bytes = ByteArray(tok.length)
        var n = 0
        for (ch in tok) {
            val b = byteDecoder[ch]
            if (b != null) bytes[n++] = b.toByte()
        }
        return String(bytes, 0, n, Charsets.UTF_8)
    }

    fun isEos(id: Int): Boolean = id in eosIds
}

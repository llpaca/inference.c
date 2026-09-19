/*
 * SafeTensors.kt - minimal, dependency-free .safetensors reader
 *
 * Mirrors the parsing approach used in inference.c (llama.c):
 *   - mmap the file
 *   - read the little-endian u64 header length
 *   - hand-parse the JSON header (no external JSON lib)
 *   - expose each tensor as a lazily-materialized FloatArray (F32/F16/BF16 -> F32)
 *
 * This is shared infrastructure used by every *_engine.kt file, the same way
 * llama.c's SafeTensors struct + st_* functions are shared by every model
 * variant compiled from that source tree.
 */

import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File

enum class STDtype { F32, BF16, F16, I32, I64, U8, BOOL, UNKNOWN }

data class STTensorInfo(
    val name: String,
    val dtype: STDtype,
    val shape: IntArray,
    val offsetStart: Long,
    val offsetEnd: Long
) {
    val numel: Long get() = shape.fold(1L) { a, b -> a * b }
}

/**
 * Open + parsed .safetensors file.
 *
 * Tensor bytes are read on demand via RandomAccessFile.seek()+readFully() rather
 * than a single memory-mapped buffer: FileChannel.map() takes a 32-bit size, so
 * it cannot map files (or even single tensor blobs) larger than ~2GB in one
 * call, which real multi-GB model checkpoints exceed. Random-access reads have
 * no such limit and are plenty fast for a one-time weight load.
 */
class SafeTensors private constructor(
    private val raf: RandomAccessFile,
    private val dataStart: Long,
    val tensors: Map<String, STTensorInfo>
) {
    companion object {
        private fun dtypeOf(s: String): STDtype = when (s) {
            "F32" -> STDtype.F32
            "BF16" -> STDtype.BF16
            "F16" -> STDtype.F16
            "I32" -> STDtype.I32
            "I64" -> STDtype.I64
            "U8" -> STDtype.U8
            "BOOL" -> STDtype.BOOL
            else -> STDtype.UNKNOWN
        }

        fun open(path: String): SafeTensors {
            val file = File(path)
            require(file.exists()) { "safetensors file not found: $path" }
            val raf = RandomAccessFile(file, "r")
            val len = raf.length()

            val headerLenBytes = ByteArray(8)
            raf.seek(0)
            raf.readFully(headerLenBytes)
            val headerLen = ByteBuffer.wrap(headerLenBytes).order(ByteOrder.LITTLE_ENDIAN).long
            require(headerLen in 1..len - 8) { "bad safetensors header length: $headerLen" }

            val headerBytes = ByteArray(headerLen.toInt())
            raf.seek(8)
            raf.readFully(headerBytes)
            val headerJson = String(headerBytes, Charsets.UTF_8)

            val tensors = parseHeader(headerJson)
            val dataStart = 8L + headerLen

            return SafeTensors(raf, dataStart, tensors)
        }

        /** Minimal hand-rolled JSON parser tuned exactly for the safetensors header shape. */
        private fun parseHeader(json: String): Map<String, STTensorInfo> {
            val result = LinkedHashMap<String, STTensorInfo>()
            var i = skipWs(json, 0)
            require(json[i] == '{') { "bad safetensors header" }
            i++
            while (true) {
                i = skipWs(json, i)
                if (json[i] == '}') { i++; break }
                if (json[i] == ',') { i++; continue }

                val (name, afterName) = readString(json, i)
                i = skipWs(json, afterName)
                require(json[i] == ':') { "expected ':' after key $name" }
                i++
                i = skipWs(json, i)

                if (name == "__metadata__") {
                    i = skipValue(json, i)
                    continue
                }

                val (entry, afterEntry) = parseTensorEntry(json, i, name)
                result[name] = entry
                i = afterEntry
            }
            return result
        }

        private fun parseTensorEntry(json: String, start: Int, name: String): Pair<STTensorInfo, Int> {
            var i = skipWs(json, start)
            require(json[i] == '{')
            i++
            var dtype = STDtype.UNKNOWN
            var shape = IntArray(0)
            var offStart = 0L
            var offEnd = 0L
            while (true) {
                i = skipWs(json, i)
                if (json[i] == '}') { i++; break }
                if (json[i] == ',') { i++; continue }
                val (key, afterKey) = readString(json, i)
                i = skipWs(json, afterKey)
                require(json[i] == ':')
                i++
                i = skipWs(json, i)
                when (key) {
                    "dtype" -> {
                        val (v, after) = readString(json, i)
                        dtype = dtypeOf(v)
                        i = after
                    }
                    "shape" -> {
                        require(json[i] == '[')
                        i++
                        val dims = ArrayList<Int>()
                        while (true) {
                            i = skipWs(json, i)
                            if (json[i] == ']') { i++; break }
                            if (json[i] == ',') { i++; continue }
                            val (num, after) = readNumber(json, i)
                            dims.add(num.toInt())
                            i = after
                        }
                        shape = dims.toIntArray()
                    }
                    "data_offsets" -> {
                        require(json[i] == '[')
                        i++
                        i = skipWs(json, i)
                        val (a, afterA) = readNumber(json, i)
                        offStart = a.toLong()
                        i = skipWs(json, afterA)
                        if (json[i] == ',') i++
                        i = skipWs(json, i)
                        val (b, afterB) = readNumber(json, i)
                        offEnd = b.toLong()
                        i = skipWs(json, afterB)
                        require(json[i] == ']')
                        i++
                    }
                    else -> {
                        i = skipValue(json, i)
                    }
                }
            }
            return STTensorInfo(name, dtype, shape, offStart, offEnd) to i
        }

        private fun skipWs(s: String, start: Int): Int {
            var i = start
            while (i < s.length && s[i].let { it == ' ' || it == '\n' || it == '\r' || it == '\t' }) i++
            return i
        }

        private fun readString(s: String, start: Int): Pair<String, Int> {
            require(s[start] == '"') { "expected string at $start" }
            var i = start + 1
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

        private fun readNumber(s: String, start: Int): Pair<Long, Int> {
            var i = start
            val begin = i
            if (s[i] == '-') i++
            while (i < s.length && s[i].isDigit()) i++
            return s.substring(begin, i).toLong() to i
        }

        /** Skip a complete JSON value (string, object, array, or bare literal/number). */
        private fun skipValue(s: String, start: Int): Int {
            var i = skipWs(s, start)
            return when {
                s[i] == '"' -> readString(s, i).second
                s[i] == '{' || s[i] == '[' -> {
                    val open = s[i]
                    val close = if (open == '{') '}' else ']'
                    var depth = 1
                    i++
                    while (i < s.length && depth > 0) {
                        when (s[i]) {
                            '\\' -> i++
                            '"' -> { i = readString(s, i).second; continue }
                            open -> depth++
                            close -> depth--
                        }
                        i++
                    }
                    i
                }
                else -> {
                    while (i < s.length && s[i] != ',' && s[i] != '}' && s[i] != ']') i++
                    i
                }
            }
        }
    }

    /** Materialize a tensor's raw bytes into a Float32 array, converting from its native dtype. */
    @Synchronized
    fun getFloatArray(name: String): FloatArray {
        val t = tensors[name] ?: error("Missing tensor: $name")
        require(t.numel <= Int.MAX_VALUE) { "Tensor $name has ${t.numel} elements, exceeds max array size ${Int.MAX_VALUE}" }
        val n = t.numel.toInt()
        val out = FloatArray(n)
        raf.seek(dataStart + t.offsetStart)

        // Read in bounded chunks rather than allocating one ByteArray for the whole
        // tensor: a single tensor's byte length can itself exceed Int.MAX_VALUE for
        // very large layers (e.g. big embedding/lm_head matrices), which would
        // overflow a plain `ByteArray(byteLen.toInt())` allocation.
        val bytesPerElem = when (t.dtype) {
            STDtype.F32 -> 4
            STDtype.BF16, STDtype.F16 -> 2
            else -> error("Unsupported dtype for tensor $name: ${t.dtype}")
        }
        val chunkElems = 1 shl 20 // 1M elements per chunk (<= 4MB per read)
        val chunkBuf = ByteArray(chunkElems * bytesPerElem)
        var elemsDone = 0
        while (elemsDone < n) {
            val thisChunkElems = minOf(chunkElems, n - elemsDone)
            val thisChunkBytes = thisChunkElems * bytesPerElem
            raf.readFully(chunkBuf, 0, thisChunkBytes)
            val bb = ByteBuffer.wrap(chunkBuf, 0, thisChunkBytes).order(ByteOrder.LITTLE_ENDIAN)
            when (t.dtype) {
                STDtype.F32 -> bb.asFloatBuffer().get(out, elemsDone, thisChunkElems)
                STDtype.BF16 -> for (idx in 0 until thisChunkElems) {
                    val bits = bb.getShort(idx * 2).toInt() and 0xFFFF
                    out[elemsDone + idx] = java.lang.Float.intBitsToFloat(bits shl 16)
                }
                STDtype.F16 -> for (idx in 0 until thisChunkElems) {
                    val bits = bb.getShort(idx * 2).toInt() and 0xFFFF
                    out[elemsDone + idx] = f16ToF32(bits)
                }
                else -> error("unreachable")
            }
            elemsDone += thisChunkElems
        }
        return out
    }

    fun has(name: String): Boolean = tensors.containsKey(name)

    private fun f16ToF32(bits: Int): Float {
        val sign = (bits shr 15) and 0x1
        var exp = (bits shr 10) and 0x1f
        var mantissa = bits and 0x3ff
        val out: Int
        if (exp == 0) {
            if (mantissa == 0) {
                out = sign shl 31
            } else {
                exp = 1
                while (mantissa and 0x400 == 0) { mantissa = mantissa shl 1; exp-- }
                mantissa = mantissa and 0x3ff
                out = (sign shl 31) or ((exp + 127 - 15) shl 23) or (mantissa shl 13)
            }
        } else if (exp == 31) {
            out = (sign shl 31) or (0xff shl 23) or (mantissa shl 13)
        } else {
            out = (sign shl 31) or ((exp + 127 - 15) shl 23) or (mantissa shl 13)
        }
        return java.lang.Float.intBitsToFloat(out)
    }
}

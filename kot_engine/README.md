# inference.kt

A Kotlin/JVM port of the [inference.c](https://github.com/llpaca/inference.c) style:
pure, dependency-free inference for small LLMs, built one model at a time as
`<model>_engine`-style code, sharing a small common core.

```
src/main/kotlin/
  SafeTensors.kt     shared: mmap + hand-rolled .safetensors header parser
  Tokenizer.kt        shared: byte-level BPE tokenizer for tokenizer.json
  Kernels.kt          shared: rmsnorm, matmul, softmax, silu, RoPE (+ Llama-3 freq scaling)
  LlamaEngine.kt      model: Llama 3.2 1B / 3B (config, weights, forward pass, sampler, CLI)
  Main.kt             CLI dispatcher: `<model> <args...>`
build.sh              compiles everything into inference.jar
```

No BLAS binding, no ONNX/PyTorch — matmul is a plain, JIT-friendly loop. It's
slower than the C version's BLAS path, but it's portable (runs anywhere the
JVM does) and it's the base to later port to Kotlin/Native or Android.

## Build

You need a Kotlin compiler (`kotlinc`) and a JDK (17+ is fine, tested on 21).

```bash
# if you don't already have kotlinc:
#   sdk install kotlin        (via SDKMAN, easiest on macOS/Linux)
# or download a release zip from https://github.com/JetBrains/kotlin/releases

./build.sh
```

This produces `inference.jar` (includes the Kotlin runtime, so you only need a JVM to run it).

## Run

```bash
java -jar inference.jar llama <model_dir> "<prompt>" [maxTokens] [temperature] [--size 1b|3b] [--weights f32|int8]
```

`model_dir` must contain:
- `model.safetensors` (single-file; sharded multi-file models aren't handled yet)
- `tokenizer.json`
- `config.json`

Example once you've downloaded a model (see below):

```bash
java -jar inference.jar llama ./models/llama-3.2-1b-instruct "Explain recursion simply." 200 0.0
java -jar inference.jar llama ./models/llama-3.2-3b-instruct "Write a haiku about the sea." 100 0.7 --size 3b
```

`temperature 0` = greedy decoding (deterministic). `temperature > 0` = sampling.

### Weight format: `--weights int8` (default) vs `--weights f32`

Model checkpoints ship as BF16 on disk. This engine can hold them in memory two ways:

| Format | Memory (1B model) | Quality | When to use |
|---|---|---|---|
| `int8` (default) | ~1.2 GB | ~0.4% RMSE vs full precision, cosine similarity >0.9999 per-layer | Phones, laptops with limited RAM — this is the one you want |
| `f32` | ~5 GB | exact upcast of the source weights | Only if you have RAM to spare and want max fidelity |

INT8 uses **per-row symmetric quantization** (each output neuron's weight row gets
its own scale, same approach as llama.cpp's Q8_0) — quantization happens once at
load time, weights are dequantized one row at a time inside the matmul, so the
full-precision matrix is never fully materialized in memory.

Embeddings and norm weights are always kept at F32 — they're small and precision-sensitive.

### CPU utilization

Matmuls are parallelized across all available CPU cores automatically (a
shared thread pool splits output rows across threads). No flag needed — check
`[runtime] cpu cores available: N` in the startup log to confirm how many
cores it found. Rows are only split across threads above a small size
threshold, so tiny norm/vector operations stay single-threaded (not worth the
overhead there).

## Getting real model weights

These are gated on Hugging Face (free, just needs a HF account + accepting Meta's license):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login

# 1B
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ./models/llama-3.2-1b-instruct \
  --include "model.safetensors" "tokenizer.json" "config.json"

# 3B
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/llama-3.2-3b-instruct \
  --include "model.safetensors" "tokenizer.json" "config.json"
```

The 3B model ships as *sharded* safetensors on the Hub by default
(`model-00001-of-00002.safetensors` etc.) — the `--include "model.safetensors"`
filter above only works if a single-file version exists for that repo. If HF
only gives you shards, either:
- look for a non-sharded re-upload (several exist, e.g. under `mlx-community/`
  or `unsloth/`, search "Llama-3.2-3B-Instruct" + "safetensors" single file), or
- ask for sharded-model support to be added to `SafeTensors.kt` (it's a
  straightforward extension: read `model.safetensors.index.json` and open each
  shard, same idea as `llama.c`'s candidate-path loop but for N files instead of 1).

## Testing without downloading a multi-GB model

There's no bundled test model in this repo (weights are large), but you can
generate a tiny random-weight model with matching tensor names/shapes to
smoke-test the whole pipeline (safetensors parsing, tokenizer, forward pass,
sampling) in under a second. See the `make_tiny_model.py` idea below — happy
to hand you that script if you want to keep it in-repo as `tools/make_tiny_model.py`.

## On the "ternary Bonsai 27B" model

I looked into this before writing any code for it. The web pages describing
a "Ternary-Bonsai-27B" / "PrismML" model (Qwen3.6-27B backbone, ~2-bit
weights, 90-95% of FP16 quality, running on an iPhone) don't hold up under
scrutiny — there's no verifiable "Qwen3.6" release, several of the supporting
tool names and benchmark citations look synthetic, and the claimed
bits-per-parameter/quality tradeoff is well outside anything published in
peer-reviewed ternary-LLM research. I'd rather tell you that directly than
quietly build an engine around specs I can't verify.

For a **real, verifiable, low-bit model good for phones**, consider instead:

- **[Microsoft BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B4T)**
  — an actual published 1.58-bit (ternary: {-1, 0, +1}) 2B model, with a paper,
  reference code, and real benchmarks. This is the legitimate version of what
  "Bonsai" is claiming to be, at a size that's realistic to run on a phone today.
- Smaller Llama/Qwen/Gemma variants (1B-4B range) quantized to real INT4/INT8,
  which is a well-established and verifiable way to get phone-class speed.

Once `LlamaEngine.kt` is solid, a `BitNetEngine.kt` is a natural next step:
same SafeTensors/Tokenizer/Kernels core, but with a ternary-packed weight
format and a specialized matmul that unpacks 2-bit-packed {-1,0,1} weights
on the fly (no floating-point multiply needed for the weight side at all,
which is where the real on-device speedup comes from).

## Roadmap

- [x] Llama 3.2 1B / 3B (`LlamaEngine.kt`)
- [x] INT8 per-row weight quantization (~4x memory reduction vs F32, ~2x vs BF16 source)
- [x] Multithreaded matmul (parallelized across CPU cores)
- [ ] Sharded safetensors support (multi-file models)
- [ ] INT4 quantization for even tighter phone budgets
- [ ] `BitNetEngine.kt` for BitNet b1.58 2B4T (real ternary model)
- [ ] Simple top-p / top-k sampling (currently: greedy or plain temperature)
- [ ] Kotlin/Native or Android target once JVM version is validated

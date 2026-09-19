"""
Builds a plain-text corpus for training the custom BPE tokenizer
(tokenizer.c), pulling from the same math/chess/reasoning sources used
by the model's training pipeline, so the vocab reflects what the model
will actually see -- in roughly the same proportions.

Usage:
    python build_tokenizer_corpus.py --out corpus.txt
    python build_tokenizer_corpus.py --out corpus.txt --extra mathstack-qa/data.txt
    python build_tokenizer_corpus.py --out corpus.txt --no-dedup   # skip dedup pass

Then train the tokenizer as before:
    ./tokenizer_fast corpus.txt tokenizer.bin 16384 8
"""

import argparse
import os
import re
from datasets import load_dataset


def normalize_for_dedup(text):
    """Same normalization as the model script's dedup step -- collapses
    numbers/punctuation so near-identical augmented examples (MetaMathQA
    in particular rephrases the same problem many times) count as one."""
    t = re.sub(r'\d+', '', text.lower())
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    return t.strip()


def collect_examples(name, texts, bucket, seen, dedup):
    n_seen, n_kept = 0, 0
    for t in texts:
        t = t.strip()
        if not t:
            continue
        n_seen += 1
        if dedup:
            key = normalize_for_dedup(t)
            if key in seen:
                continue
            seen.add(key)
        bucket.append(t.replace("\n", " \\n "))
        n_kept += 1
    print(f"  {name}: {n_seen:,} examples -> {n_kept:,} kept" +
          (f" ({n_seen - n_kept:,} deduped)" if dedup else ""))


def build_corpus(out_path, extra_files, dedup=True):
    lines = []
    seen = set()

    # ===== MATH (full sets -- these are the bulk of the model's real corpus) =====
    print("Loading GSM8K...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    collect_examples("gsm8k",
                      (f"Question: {x['question']}\nAnswer: {x['answer']}" for x in gsm8k),
                      lines, seen, dedup)

    print("Loading MetaMathQA...")
    metamath = load_dataset("meta-math/MetaMathQA", split="train")
    collect_examples("metamathqa",
                      (f"Question: {x['query']}\nAnswer: {x['response']}" for x in metamath),
                      lines, seen, dedup)

    print("Loading MATH (DigitalLearningGmbH/MATH-lighteval)...")
    math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    collect_examples("math",
                      (f"Question: {x.get('problem', '')}\nAnswer: {x.get('solution', '')}" for x in math_ds),
                      lines, seen, dedup)

    print("Loading MiniF2F...")
    minif2f = load_dataset("Tonic/MiniF2F", split="train")
    def minif2f_text(x):
        conv = x.get("conversation", [])
        if len(conv) >= 2:
            return f"Question: {conv[0].get('content','')}\nAnswer: {conv[-1].get('content','')}"
        return ""
    collect_examples("minif2f", (minif2f_text(x) for x in minif2f), lines, seen, dedup)

    # ===== CHESS (capped to match the model script's proportions) =====
    print("Loading Lichess chess puzzles (30k sample)...")
    chess = load_dataset("Lichess/chess-puzzles", split="train[:30000]")
    collect_examples("chess",
                      (f"Chess Position (FEN): {x['FEN']}\nBest move sequence: {x['Moves']}" for x in chess),
                      lines, seen, dedup)

    # ===== REASONING (capped to match the model script's proportions) =====
    print("Loading ARC-Challenge (20k sample)...")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train[:20000]")
    collect_examples("arc",
                      (f"Question: {x['question']}\nChoices: {', '.join(x['choices']['text'])}\nAnswer: {x['answerKey']}"
                       for x in arc), lines, seen, dedup)

    print("Loading CommonsenseQA (20k sample)...")
    csqa = load_dataset("tau/commonsense_qa", split="train[:20000]")
    collect_examples("commonsense_qa",
                      (f"Question: {x['question']}\nChoices: {', '.join(x['choices']['text'])}\nAnswer: {x['answerKey']}"
                       for x in csqa), lines, seen, dedup)

    print("Loading OpenBookQA (20k sample)...")
    obqa = load_dataset("allenai/openbookqa", split="train[:20000]")
    collect_examples("openbookqa",
                      (f"Question: {x['question_stem']}\nChoices: {', '.join(x['choices']['text'])}\nAnswer: {x['answerKey']}"
                       for x in obqa), lines, seen, dedup)

    # ===== EXTRA LOCAL FILES (e.g. your mathstack-qa/data.txt) =====
    for path in extra_files:
        print(f"Adding local file: {path}")
        n_seen, n_kept = 0, 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_seen += 1
                if dedup:
                    key = normalize_for_dedup(line)
                    if key in seen:
                        continue
                    seen.add(key)
                lines.append(line)
                n_kept += 1
        print(f"  {path}: {n_seen:,} lines -> {n_kept:,} kept")

    # ===== WRITE OUT + SIZE REPORT =====
    with open(out_path, "w", encoding="utf-8") as out_f:
        for line in lines:
            out_f.write(line + "\n")

    size_bytes = os.path.getsize(out_path)
    size_mb = size_bytes / (1024 * 1024)
    approx_chars = sum(len(l) for l in lines)

    print(f"\nDone. Corpus written to: {out_path}")
    print(f"  Lines: {len(lines):,}")
    print(f"  Size: {size_mb:.1f} MB ({size_bytes:,} bytes)")
    print(f"  Approx. characters: {approx_chars:,}")

    # Rough rule of thumb for byte-level BPE: you want the corpus to be
    # at least ~1000x your target vocab size in bytes, so every merge
    # candidate has enough occurrences to rank reliably. Below that, late
    # merges (highest vocab ids) start getting picked from noisy, low-count
    # pairs -- and your trainer's own "No more mergeable pairs" message
    # is the hard failure mode if it runs out entirely.
    for target_vocab in (9216, 16384):
        min_recommended_mb = (target_vocab * 1000) / (1024 * 1024)
        verdict = "OK" if size_mb >= min_recommended_mb else "MIGHT BE TOO SMALL"
        print(f"  For vocab={target_vocab}: recommend >= {min_recommended_mb:.1f} MB -> {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="corpus.txt", help="Output corpus path")
    parser.add_argument("--extra", nargs="*", default=[],
                         help="Extra local text files to append (e.g. mathstack-qa/data.txt)")
    parser.add_argument("--no-dedup", action="store_true",
                         help="Skip the near-duplicate filter (keeps MetaMathQA's rephrasing duplicates etc.)")
    args = parser.parse_args()
    build_corpus(args.out, args.extra, dedup=not args.no_dedup)
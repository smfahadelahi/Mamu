"""Build a tokenizer and tokenised training shards from local text files.

Usage:
  python -m data.prepare train-tokenizer --files corpus/*.txt --out tokenizer/ --vocab-size 32000
  python -m data.prepare tokenize --files corpus/*.txt --tokenizer tokenizer/tokenizer.model --out shards/

The tokenizer is SentencePiece BPE with byte fallback, the same family as LLaMA.
llama.cpp converts this format without changes. Byte-level BPE with a custom
pre-tokenizer would need an entry added to llama.cpp's convert script.

Only local files are read. Documents are separated by blank lines. The filters are
deliberately simple: they drop very short, mostly non-text, and exact-duplicate
documents.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import tempfile
from collections.abc import Iterator

import numpy as np
import sentencepiece as spm

# Ids follow the LLaMA convention so exported configs use bos=1, eos=2.
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, -1
MODEL_NAME = "tokenizer.model"


def iter_documents(paths: list[str], sep: str = "\n\n") -> Iterator[str]:
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for doc in f.read().split(sep):
                doc = doc.strip()
                if doc:
                    yield doc


def keep_document(doc: str, min_chars: int = 200, min_ascii_ratio: float = 0.85) -> bool:
    if len(doc) < min_chars:
        return False
    ascii_ratio = sum(c.isascii() for c in doc) / len(doc)
    # Keep non-English text for multilingual use, but drop mostly-symbol noise.
    alpha = sum(c.isalpha() for c in doc) / len(doc)
    return ascii_ratio >= min_ascii_ratio or alpha >= 0.6


def train_tokenizer(paths: list[str], out_dir: str, vocab_size: int = 32_000) -> str:
    """Train SentencePiece BPE on filtered documents. Returns the model path."""
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "tokenizer")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        for doc in iter_documents(paths):
            if keep_document(doc):
                # One document per line; SentencePiece treats newlines as sentence breaks.
                tmp.write(" ".join(doc.split()) + "\n")
        input_path = tmp.name
    try:
        spm.SentencePieceTrainer.train(
            input=input_path,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            byte_fallback=True,          # any UTF-8 byte is representable
            character_coverage=0.9995,
            unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            split_digits=True,
            add_dummy_prefix=True,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            minloglevel=2,
        )
    finally:
        os.unlink(input_path)
    return model_prefix + ".model"


def tokenize_to_shards(
    paths: list[str],
    tokenizer_path: str,
    out_dir: str,
    shard_tokens: int = 100_000_000,
    val_fraction: float = 0.005,
) -> dict:
    """Write uint16 .bin shards for train and val. Returns summary statistics."""
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    if sp.vocab_size() > 65_535:
        raise ValueError("vocab does not fit in uint16 shards")
    os.makedirs(out_dir, exist_ok=True)

    seen: set[str] = set()
    buf: list[int] = []
    shard_idx = {"train": 0, "val": 0}
    counts = {"train": 0, "val": 0, "docs": 0, "dropped_dupes": 0, "dropped_quality": 0}

    def flush(split: str) -> None:
        if not buf:
            return
        np.asarray(buf, dtype=np.uint16).tofile(
            os.path.join(out_dir, f"{split}_{shard_idx[split]:05d}.bin"))
        shard_idx[split] += 1
        counts[split] += len(buf)
        buf.clear()

    for doc in iter_documents(paths):
        if not keep_document(doc):
            counts["dropped_quality"] += 1
            continue
        h = hashlib.sha1(doc.encode("utf-8")).hexdigest()
        if h in seen:
            counts["dropped_dupes"] += 1
            continue
        seen.add(h)
        # Deterministic split by document hash, so reruns give the same split.
        split = "val" if int(h[:8], 16) / 0xFFFFFFFF < val_fraction else "train"
        buf.extend(sp.encode(doc) + [EOS_ID])  # EOS separates documents
        counts["docs"] += 1
        if len(buf) >= shard_tokens:
            flush(split)
    flush("train")
    flush("val")
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-tokenizer")
    t.add_argument("--files", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--vocab-size", type=int, default=32_000)
    z = sub.add_parser("tokenize")
    z.add_argument("--files", nargs="+", required=True)
    z.add_argument("--tokenizer", required=True)
    z.add_argument("--out", required=True)
    args = p.parse_args()
    files = sorted({f for pat in args.files for f in glob.glob(pat)})
    if not files:
        raise SystemExit("no input files matched")
    if args.cmd == "train-tokenizer":
        print("tokenizer written to", train_tokenizer(files, args.out, args.vocab_size))
    else:
        print(tokenize_to_shards(files, args.tokenizer, args.out))


if __name__ == "__main__":
    main()

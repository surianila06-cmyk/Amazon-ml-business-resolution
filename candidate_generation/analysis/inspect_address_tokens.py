"""
Stage 2 (address blocking) prep: inspect normalized business_address token
document-frequency on a sample of Source-2/Source-3, to empirically justify
which tokens are too common to use as address blocking keys.

v2: a first version sampled every 15th row (~688k addresses) and hand-picked
the top 40 tokens as an explicit stopword list. That was NOT tight enough:
compound place names made of two individually-common-but-not-top-40 words
(e.g. "high" + "point" for "High Point, NC") still let unrelated businesses
in the same city share >=2 tokens, so a full-scale run trended toward tens
of GB of candidates instead of shrinking. This version samples more
densely (stride=5, ~2.05M addresses) and reports a much longer tail (up to
rank 200) so a data-driven DOCUMENT-FREQUENCY THRESHOLD can replace the
hand-picked list -- catching tokens like "high"/"point" that individually
sit outside a top-40 cut but are still far too common to be distinctive.

Read-only inspection script; does not modify any dataset file, does not
build an index, does not touch ground truth.

Usage:
    python candidate_generation/analysis/inspect_address_tokens.py
"""
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preprocessing import normalize_business_address  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

# <repo root>/dataset/train, derived from this script's location (candidate_generation/analysis/)
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset" / "train")
STRIDE = 5  # every 5th row -> ~1M+1M addresses sampled, spread across each file

_TOKEN_SPLIT_RE = re.compile(r"[,\s]+")


def address_tokens(normalized_address):
    if not normalized_address:
        return []
    return [t for t in _TOKEN_SPLIT_RE.split(normalized_address) if t]


def main():
    word_doc_freq = Counter()
    numeric_token_lengths = Counter()
    n_addresses = 0
    n_empty = 0

    for fname in ("train_source2.tsv", "train_source3.tsv"):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            idx_addr = header.index("business_address")
            for i, row in enumerate(reader):
                if i % STRIDE != 0:
                    continue
                norm = normalize_business_address(row[idx_addr])
                n_addresses += 1
                if not norm:
                    n_empty += 1
                    continue
                tokens = set(address_tokens(norm))  # doc frequency: count each token once per address
                for tok in tokens:
                    if any(ch.isdigit() for ch in tok):
                        digits_only_len = sum(ch.isdigit() for ch in tok)
                        numeric_token_lengths[digits_only_len] += 1
                    else:
                        word_doc_freq[tok] += 1
        print(f"Sampled from {fname} (stride={STRIDE}), running total addresses: {n_addresses:,}")

    print(f"\nTotal sampled addresses: {n_addresses:,} ({n_empty:,} empty/null, "
          f"{n_empty / n_addresses * 100:.2f}%)")

    denom = n_addresses - n_empty
    print(f"\nTop 200 WORD tokens by document frequency (fraction of non-empty "
          f"sampled addresses containing the token):")
    for tok, count in word_doc_freq.most_common(200):
        print(f"  {tok!r:20s} {count:>8,}  ({count / denom * 100:6.4f}%)")

    print(f"\nHow many distinct word tokens exceed each candidate threshold "
          f"(this sizes the resulting stopword list):")
    for threshold_pct in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        n_over = sum(1 for c in word_doc_freq.values() if c / denom * 100 > threshold_pct)
        print(f"  > {threshold_pct:>4.2f}%: {n_over:,} tokens")

    print(f"\nNUMERIC token digit-length distribution (how many digits in the "
          f"token, counted once per address where it appears):")
    for length in sorted(numeric_token_lengths):
        print(f"  {length:2d} digits: {numeric_token_lengths[length]:>8,} occurrences")


if __name__ == "__main__":
    main()

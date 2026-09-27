"""
Stage 2 (address blocking) prep: dump the FULL word-token document-frequency
table for the same stride-5 Source-2/Source-3 sample used by
inspect_address_tokens.py (which printed only the top 200). The expanded
ADDRESS_WORD_STOPWORDS list is derived from this table by a DF threshold.

Output: candidate_generation/analysis/address_word_token_df.tsv
  columns: token, doc_count, doc_freq_pct  (denominator = non-empty sampled
  addresses), sorted by doc_count desc, tokens with doc_count >= 100 only.

Read-only w.r.t. datasets; no ground truth used.
"""
import csv
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.analysis.inspect_address_tokens import (  # noqa: E402
    DATA_DIR, STRIDE, address_tokens,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "address_word_token_df.tsv")
MIN_COUNT_TO_WRITE = 100


def main():
    word_doc_freq = Counter()
    n_addresses = n_empty = 0
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            idx_addr = next(reader).index("business_address")
            for i, row in enumerate(reader):
                if i % STRIDE != 0:
                    continue
                norm = normalize_business_address(row[idx_addr])
                n_addresses += 1
                if not norm:
                    n_empty += 1
                    continue
                for tok in set(address_tokens(norm)):
                    if not any(ch.isdigit() for ch in tok):
                        word_doc_freq[tok] += 1
    denom = n_addresses - n_empty
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# sampled_addresses={n_addresses} empty={n_empty} denom_nonempty={denom} stride={STRIDE}\n")
        f.write("token\tdoc_count\tdoc_freq_pct\n")
        for tok, c in word_doc_freq.most_common():
            if c < MIN_COUNT_TO_WRITE:
                break
            f.write(f"{tok}\t{c}\t{c / denom * 100:.6f}\n")
    print(f"sampled={n_addresses:,} empty={n_empty:,} denom={denom:,} distinct_words={len(word_doc_freq):,}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

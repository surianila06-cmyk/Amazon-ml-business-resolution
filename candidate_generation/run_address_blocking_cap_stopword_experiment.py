#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation
Stage 2 (address blocking) BOUNDED experiment #2: posting-list cap x
frequency-derived stopword list, at MIN_SHARED_TOKENS=2.

Follows run_address_blocking_bounded_experiment.py (min=2 vs min=3), which
kept min=2 as primary (min=3 lost ~28 points of pair recall). This script
asks whether a lower per-token posting cap and/or a larger, DF-derived
stopword list gives a better recall / candidate-volume tradeoff.

Stopword lists are derived from candidate_generation/analysis/
address_word_token_df.tsv (dump_address_token_df.py: full word-token
document-frequency table over the same stride-5 S2+S3 sample as
inspect_address_tokens.py, 1,995,322 non-empty addresses). A list for
threshold T = current ADDRESS_WORD_STOPWORDS UNION {word tokens with
len >= 2 and doc_freq_pct > T}. Single-char and numeric tokens are already
excluded by is_keepable_address_token, so they are not counted.

Exact emulation, one index:
  - The S2/S3 indexes are built ONCE, exactly as in the previous bounded
    experiment (cap 5,000, current ADDRESS_WORD_STOPWORDS), so the baseline
    reproduces the previous min=2 numbers.
  - Every expanded stopword list is a superset of the current list, so a
    larger list is emulated by dropping extra query tokens.
  - The index keeps FIRST-SEEN ids up to the cap, so cap N < 5,000 is
    exactly postings[:N].
  Hence each configuration is identical to rebuilding the index with that
  cap/list, without holding more than one pair of indexes in memory.

Same 20,000 stride-sampled S1 entities as the previous experiment (same
sample_source1). Candidate files are NOT written (they were ~600MB each);
'output size' is the exact byte count of the TSV that would be written in
the same format. Ground truth is used only for scoring.

Does NOT modify utils/preprocessing.py (normalize_business_address). Does NOT
launch a full 2.2M-entity run.

Usage:
    python candidate_generation/run_address_blocking_cap_stopword_experiment.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv
"""

import argparse
import os
import statistics
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import (  # noqa: E402
    ADDRESS_WORD_STOPWORDS,
    address_blocking_keys,
    build_address_token_index,
)
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    load_ground_truth_for_ids,
    sample_source1,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

DF_TABLE_PATH = os.path.join(_REPO_ROOT, "candidate_generation", "analysis", "address_word_token_df.tsv")
INDEX_CAP = 5000
CAPS = (1000, 2000, 5000)
# None = current ADDRESS_WORD_STOPWORDS (baseline); otherwise DF threshold in %.
DF_THRESHOLDS_PCT = (None, 0.5, 0.2, 0.1)
MIN_SHARED_TOKENS = 2


def load_df_table(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("token\t"):
                continue
            tok, count, pct = line.rstrip("\n").split("\t")
            rows.append((tok, int(count), float(pct)))
    return rows


def stopwords_for_threshold(df_rows, threshold_pct):
    if threshold_pct is None:
        return ADDRESS_WORD_STOPWORDS
    return ADDRESS_WORD_STOPWORDS | frozenset(
        tok for tok, _, pct in df_rows if len(tok) >= 2 and pct > threshold_pct)


def evaluate_config(sample_keys, s2_index, s3_index, stopwords, cap, truth):
    t0 = time.time()
    per_entity_counts = []
    total_pairs = zero_candidates = out_bytes = 0
    eval_entities_with_truth = entities_covered = total_true_pairs = recovered_true_pairs = 0

    out_bytes += len("source1_entity_id\tcandidate_entity_ids\n".encode("utf-8"))
    for eid, base_keys in sample_keys:
        keys = [t for t in base_keys if t not in stopwords]
        s2_counts = Counter()
        s3_counts = Counter()
        for tok in keys:
            s2_counts.update(s2_index.get(tok, ())[:cap])
            s3_counts.update(s3_index.get(tok, ())[:cap])

        candidates = ({c for c, ct in s2_counts.items() if ct >= MIN_SHARED_TOKENS}
                      | {c for c, ct in s3_counts.items() if ct >= MIN_SHARED_TOKENS})
        n = len(candidates)
        per_entity_counts.append(n)
        total_pairs += n
        if n == 0:
            zero_candidates += 1
        out_bytes += len(f"{eid}\t{','.join(sorted(candidates))}\n".encode("utf-8"))

        truth_ids = truth.get(eid)
        if truth_ids:
            eval_entities_with_truth += 1
            tp = len(candidates & truth_ids)
            total_true_pairs += len(truth_ids)
            recovered_true_pairs += tp
            if tp:
                entities_covered += 1

    n_entities = len(sample_keys)
    return {
        "n_entities": n_entities,
        "total_pairs": total_pairs,
        "mean": total_pairs / n_entities,
        "median": statistics.median(per_entity_counts),
        "max": max(per_entity_counts),
        "zero_pct": zero_candidates / n_entities * 100,
        "out_size_bytes": out_bytes,
        "elapsed_s": time.time() - t0,
        "eval_entities_with_truth": eval_entities_with_truth,
        "entities_covered": entities_covered,
        "entity_coverage_pct": entities_covered / eval_entities_with_truth * 100,
        "total_true_pairs": total_true_pairs,
        "recovered_true_pairs": recovered_true_pairs,
        "pair_recall_pct": recovered_true_pairs / total_true_pairs * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--sample-size", type=int, default=20000)
    args = parser.parse_args()

    df_rows = load_df_table(DF_TABLE_PATH)
    stopword_sets = {th: stopwords_for_threshold(df_rows, th) for th in DF_THRESHOLDS_PCT}
    for th, sw in stopword_sets.items():
        label = "current list" if th is None else f"DF > {th}%"
        print(f"Stopword set [{label}]: {len(sw)} tokens")

    print(f"\nSampling {args.sample_size:,} S1 entities from {args.source1} (stride-sampled) ...")
    sample_rows, total_s1, stride = sample_source1(args.source1, args.sample_size)
    print(f"  total S1 rows: {total_s1:,}, stride: {stride}, sampled: {len(sample_rows):,}")
    # Only keys are needed downstream; drop raw addresses.
    sample_keys = [(eid, address_blocking_keys(normalize_business_address(addr)))
                   for eid, addr in sample_rows]
    del sample_rows

    truth = load_ground_truth_for_ids(args.ground_truth, {eid for eid, _ in sample_keys})
    print(f"  loaded {len(truth):,} ground-truth rows")

    t0 = time.time()
    print(f"\nBuilding FULL address-token index over {args.source2} (cap {INDEX_CAP:,}) ...")
    s2_index = build_address_token_index(args.source2, max_posting_size=INDEX_CAP)
    print(f"Building FULL address-token index over {args.source3} (cap {INDEX_CAP:,}) ...")
    s3_index = build_address_token_index(args.source3, max_posting_size=INDEX_CAP)
    print(f"  index build elapsed: {time.time() - t0:.1f}s")

    # Keep only the postings the sample can ever touch; release the rest.
    needed = {t for _, keys in sample_keys for t in keys}
    s2_index = {t: s2_index[t] for t in needed if t in s2_index}
    s3_index = {t: s3_index[t] for t in needed if t in s3_index}
    print(f"  pruned indexes to the {len(needed):,} tokens used by the sample")

    results = []
    for th in DF_THRESHOLDS_PCT:
        for cap in CAPS:
            label = f"{'base35' if th is None else f'DF>{th}%'} ({len(stopword_sets[th])}) / cap {cap}"
            print(f"\nEvaluating {label} ...")
            r = evaluate_config(sample_keys, s2_index, s3_index, stopword_sets[th], cap, truth)
            r["label"] = label
            results.append(r)
            print(f"  pairs={r['total_pairs']:,} mean={r['mean']:.1f} zero={r['zero_pct']:.3f}% "
                  f"entity_cov={r['entity_coverage_pct']:.4f}% pair_recall={r['pair_recall_pct']:.4f}% "
                  f"({r['elapsed_s']:.1f}s)")

    print(f"\n{'=' * 78}\nCOMPARISON (MIN_SHARED_TOKENS=2, sample {len(sample_keys):,} of {total_s1:,} S1)\n{'=' * 78}")
    cols = ["label", "total_pairs", "mean", "median", "max", "zero_pct", "out_size_bytes", "elapsed_s",
            "eval_entities_with_truth", "entities_covered", "entity_coverage_pct",
            "total_true_pairs", "recovered_true_pairs", "pair_recall_pct"]
    print("\t".join(cols))
    for r in results:
        print("\t".join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols))


if __name__ == "__main__":
    main()

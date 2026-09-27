#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation
Stage 2 (address blocking) BOUNDED experiment: compare MIN_SHARED_TOKENS=2
vs MIN_SHARED_TOKENS=3 on a manageable SAMPLE of Source-1 entities.

Context: a full 2.2M-entity run with MIN_SHARED_TOKENS=2 was killed after
~24 minutes, having already written 2.3GB of candidates for only ~3% of
entities, projecting to ~65-70GB / ~12+ hours to complete. This script
measures whether raising the threshold to 3 controls candidate-set size
without destroying recall, WITHOUT launching another full-scale run.

Design:
  - The Source-2/Source-3 address-token index is built ONCE, from the FULL
    files (blocking retrieval needs the real target corpus to be
    meaningful) -- this is the one unavoidable, fixed cost.
  - Only the Source-1 QUERY side is sampled (stride-sampled, spread evenly
    across the file, not just the head) to keep the experiment fast and
    safe.
  - Both threshold settings are evaluated against the SAME sample and the
    SAME index, so the comparison isolates the effect of min_shared_tokens
    alone. The stopword list (ADDRESS_WORD_STOPWORDS) is left unchanged in
    this script -- any recommendation to expand it is reported separately,
    from candidate_generation/analysis/address_token_inspection_output.txt.
  - Ground truth is loaded only for the sampled S1 IDs (not the full 2.2M
    rows), then used purely for scoring -- never for candidate generation.

Does NOT modify exact_name_blocking.py or utils/preprocessing.py. Does NOT
launch a full 2.2M-entity run.

Usage:
    python candidate_generation/run_address_blocking_bounded_experiment.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --sample-size 20000
"""

import argparse
import csv
import os
import statistics
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import (  # noqa: E402
    address_blocking_keys,
    build_address_token_index,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass


def count_rows(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for _ in f:
            n += 1
    return n


def sample_source1(path, sample_size):
    """Stride-sample `sample_size` (entity_id, business_address) rows,
    spread evenly across the whole file."""
    total = count_rows(path)
    stride = max(1, total // sample_size)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_addr = header.index("business_address")
        for i, row in enumerate(reader):
            if i % stride == 0:
                rows.append((row[idx_id], row[idx_addr]))
                if len(rows) >= sample_size:
                    break
    return rows, total, stride


def load_ground_truth_for_ids(path, wanted_ids):
    truth = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if row[0] in wanted_ids:
                truth[row[0]] = frozenset(row[1].split(",")) if row[1].strip() else frozenset()
    return truth


def evaluate_setting(sample_rows, s2_index, s3_index, min_shared_tokens, truth, out_path):
    t0 = time.time()
    per_entity_counts = []
    total_pairs = 0
    zero_candidates = 0

    eval_entities_with_truth = 0
    entities_covered = 0
    total_true_pairs = 0
    recovered_true_pairs = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, addr in sample_rows:
            norm = normalize_business_address(addr)
            keys = address_blocking_keys(norm)

            s2_counts = Counter()
            s3_counts = Counter()
            for tok in keys:
                for cid in s2_index.get(tok, ()):
                    s2_counts[cid] += 1
                for cid in s3_index.get(tok, ()):
                    s3_counts[cid] += 1

            candidates = ({c for c, ct in s2_counts.items() if ct >= min_shared_tokens}
                          | {c for c, ct in s3_counts.items() if ct >= min_shared_tokens})

            n = len(candidates)
            per_entity_counts.append(n)
            total_pairs += n
            if n == 0:
                zero_candidates += 1

            f_out.write(f"{eid}\t{','.join(sorted(candidates))}\n")

            truth_ids = truth.get(eid)
            if truth_ids:
                eval_entities_with_truth += 1
                tp = len(candidates & truth_ids)
                total_true_pairs += len(truth_ids)
                recovered_true_pairs += tp
                if tp > 0:
                    entities_covered += 1

    elapsed = time.time() - t0
    out_size = os.path.getsize(out_path)
    n_entities = len(sample_rows)

    return {
        "min_shared_tokens": min_shared_tokens,
        "n_entities": n_entities,
        "total_pairs": total_pairs,
        "mean": total_pairs / n_entities,
        "median": statistics.median(per_entity_counts),
        "max": max(per_entity_counts),
        "zero_pct": zero_candidates / n_entities * 100,
        "out_size_bytes": out_size,
        "elapsed_s": elapsed,
        "eval_entities_with_truth": eval_entities_with_truth,
        "entities_covered": entities_covered,
        "entity_coverage_pct": (entities_covered / eval_entities_with_truth * 100
                                 if eval_entities_with_truth else None),
        "total_true_pairs": total_true_pairs,
        "recovered_true_pairs": recovered_true_pairs,
        "pair_recall_pct": (recovered_true_pairs / total_true_pairs * 100
                             if total_true_pairs else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--out-dir", default="candidate_generation/output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Sampling {args.sample_size:,} S1 entities from {args.source1} (stride-sampled) ...")
    sample_rows, total_s1, stride = sample_source1(args.source1, args.sample_size)
    print(f"  total S1 rows: {total_s1:,}, stride: {stride}, sampled: {len(sample_rows):,}")

    wanted_ids = {eid for eid, _ in sample_rows}
    print(f"\nLoading ground truth for the {len(wanted_ids):,} sampled S1 entities only ...")
    truth = load_ground_truth_for_ids(args.ground_truth, wanted_ids)
    print(f"  loaded {len(truth):,} ground-truth rows")

    print(f"\nBuilding FULL address-token index over {args.source2} (Source-2) ...")
    t0 = time.time()
    s2_index = build_address_token_index(args.source2)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"\nBuilding FULL address-token index over {args.source3} (Source-3) ...")
    s3_index = build_address_token_index(args.source3)
    print(f"  elapsed: {time.time() - t1:.1f}s")

    results = {}
    for mst in (2, 3):
        out_path = os.path.join(args.out_dir, f"bounded_experiment_min{mst}_candidates.tsv")
        print(f"\nEvaluating MIN_SHARED_TOKENS={mst} on the {len(sample_rows):,}-entity sample ...")
        results[mst] = evaluate_setting(sample_rows, s2_index, s3_index, mst, truth, out_path)
        print(f"  done: {results[mst]['elapsed_s']:.2f}s, output: {out_path}")

    print(f"\n{'=' * 78}")
    print(f"BOUNDED COMPARISON: MIN_SHARED_TOKENS=2 vs 3  "
          f"(sample: {len(sample_rows):,} of {total_s1:,} S1 entities)")
    print("=" * 78)

    rows = [
        ("S1 entities sampled", "n_entities", "{:,}"),
        ("Total candidate pairs", "total_pairs", "{:,}"),
        ("Mean candidates/S1", "mean", "{:.4f}"),
        ("Median candidates/S1", "median", "{:.1f}"),
        ("Max candidates/S1", "max", "{:,}"),
        ("Zero-candidate %", "zero_pct", "{:.3f}%"),
        ("Output size (bytes)", "out_size_bytes", "{:,}"),
        ("Runtime (s)", "elapsed_s", "{:.2f}"),
        ("Entities with >=1 true match (eval)", "eval_entities_with_truth", "{:,}"),
        ("Entities where >=1 true match recovered", "entities_covered", "{:,}"),
        ("Entity-level coverage", "entity_coverage_pct", "{:.4f}%"),
        ("Total true pairs (in sample)", "total_true_pairs", "{:,}"),
        ("Recovered true pairs", "recovered_true_pairs", "{:,}"),
        ("Pair-level recall", "pair_recall_pct", "{:.4f}%"),
    ]
    header = f"{'Metric':42s} {'min=2':>18s} {'min=3':>18s}"
    print(header)
    print("-" * len(header))
    for label, key, fmt in rows:
        v2 = results[2][key]
        v3 = results[3][key]
        s2 = fmt.format(v2) if v2 is not None else "n/a"
        s3 = fmt.format(v3) if v3 is not None else "n/a"
        print(f"{label:42s} {s2:>18s} {s3:>18s}")


if __name__ == "__main__":
    main()

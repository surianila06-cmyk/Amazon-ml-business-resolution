#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation
Evaluation: score a candidate_pairs.tsv-format file against
train_ground_truth.tsv.

Ground truth is used ONLY by this script, for scoring. It is never read
during candidate generation (see exact_name_blocking.py /
run_exact_name_blocking.py).

Generic across blocking stages: any file following the project's
candidate_pairs.tsv contract (header 'source1_entity_id\\tcandidate_entity_ids',
comma-separated S2-/S3- IDs, one row per Source-1 entity) can be scored here.

Reports, separately for S2-only candidates, S3-only candidates, and the
combined S2+S3 candidate set:
  - total candidate pairs, mean/median/max candidates per S1 entity
  - S1 entities with zero candidates
  - entity-level coverage: fraction of S1 entities with >=1 true match (in
    that scope) for which at least one true match was retrieved
  - pair-level recall: recovered true pairs / total true pairs
  - false candidate pairs (candidates that are not true matches)
  - candidate reduction ratio vs. brute-force S1 x S2/S3 pairing (only if
    --source2/--source3 are given, to count target-file rows)

Usage:
    python candidate_generation/evaluate_candidates.py \\
        --candidates candidate_generation/output/train_candidate_pairs_exact_name.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv
"""

import argparse
import csv
import statistics
import sys

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

SCOPES = ("s2", "s3", "all")
PREFIX = {"s2": "S2-", "s3": "S3-"}


def load_ground_truth(path):
    truth = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            s1_id, matched_raw = row[0], row[1]
            truth[s1_id] = frozenset(matched_raw.split(",")) if matched_raw.strip() else frozenset()
    return truth


def count_rows(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for _ in f:
            n += 1
    return n


def scoped(ids, scope):
    if scope == "all":
        return ids
    prefix = PREFIX[scope]
    return {i for i in ids if i.startswith(prefix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv-format file to score")
    parser.add_argument("--ground-truth", required=True, help="train_ground_truth.tsv path")
    parser.add_argument("--source2", default=None, help="Optional, for candidate reduction ratio")
    parser.add_argument("--source3", default=None, help="Optional, for candidate reduction ratio")
    args = parser.parse_args()

    print(f"Loading ground truth from {args.ground_truth} ...")
    truth = load_ground_truth(args.ground_truth)
    print(f"  {len(truth):,} ground-truth rows loaded.\n")

    num_s2 = count_rows(args.source2) if args.source2 else None
    num_s3 = count_rows(args.source3) if args.source3 else None
    if num_s2 is not None:
        print(f"Source-2 row count (for reduction ratio): {num_s2:,}")
    if num_s3 is not None:
        print(f"Source-3 row count (for reduction ratio): {num_s3:,}")

    total_pairs = {s: 0 for s in SCOPES}
    counts = {s: [] for s in SCOPES}
    zero_candidates = {s: 0 for s in SCOPES}
    entities_with_truth = {s: 0 for s in SCOPES}
    entities_covered = {s: 0 for s in SCOPES}
    total_true_pairs = {s: 0 for s in SCOPES}
    recovered_true_pairs = {s: 0 for s in SCOPES}
    false_pairs = {s: 0 for s in SCOPES}

    n_entities = 0
    missing_from_truth = 0

    print(f"\nScoring {args.candidates} ...")
    with open(args.candidates, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            s1_id, cand_raw = row[0], row[1]
            cand_ids = set(cand_raw.split(",")) if cand_raw.strip() else set()
            truth_ids = truth.get(s1_id)
            if truth_ids is None:
                missing_from_truth += 1
                truth_ids = frozenset()

            for scope in SCOPES:
                sc_cand = scoped(cand_ids, scope)
                sc_truth = scoped(truth_ids, scope)

                n_cand = len(sc_cand)
                total_pairs[scope] += n_cand
                counts[scope].append(n_cand)
                if n_cand == 0:
                    zero_candidates[scope] += 1

                false_pairs[scope] += len(sc_cand - sc_truth)

                if sc_truth:
                    entities_with_truth[scope] += 1
                    tp = len(sc_cand & sc_truth)
                    total_true_pairs[scope] += len(sc_truth)
                    recovered_true_pairs[scope] += tp
                    if tp > 0:
                        entities_covered[scope] += 1

            n_entities += 1
            if n_entities % 500_000 == 0:
                print(f"  scored {n_entities:,} entities...")

    print(f"  done: scored {n_entities:,} entities.")
    if missing_from_truth:
        print(f"  WARNING: {missing_from_truth:,} candidate rows had no matching "
              f"ground-truth entry (unexpected -- check candidates file / entity_id values).")

    print(f"\n{'=' * 78}")
    print(f"{'TOTAL S1 ENTITIES:':40s} {n_entities:,}")
    print("=" * 78)

    reduction_denominator = {
        "s2": n_entities * num_s2 if num_s2 is not None else None,
        "s3": n_entities * num_s3 if num_s3 is not None else None,
        "all": n_entities * (num_s2 + num_s3) if (num_s2 is not None and num_s3 is not None) else None,
    }

    for scope in SCOPES:
        label = {"s2": "S2 CANDIDATES", "s3": "S3 CANDIDATES", "all": "COMBINED S2+S3 CANDIDATES"}[scope]
        print(f"\n--- {label} ---")
        c = counts[scope]
        print(f"  Total candidate pairs: {total_pairs[scope]:,}")
        print(f"  Mean candidates per S1 entity: {total_pairs[scope] / n_entities:.4f}")
        print(f"  Median candidates per S1 entity: {statistics.median(c):.1f}")
        print(f"  Max candidates for one S1 entity: {max(c):,}")
        print(f"  S1 entities with zero candidates: {zero_candidates[scope]:,} "
              f"({zero_candidates[scope] / n_entities * 100:.3f}%)")

        ewt = entities_with_truth[scope]
        print(f"  S1 entities with >=1 true match (in this scope): {ewt:,}")
        if ewt:
            coverage = entities_covered[scope] / ewt
            print(f"  Entity-level candidate coverage (>=1 true match retrieved): "
                  f"{coverage:.4%} ({entities_covered[scope]:,} / {ewt:,})")
        else:
            print("  Entity-level candidate coverage: n/a (no true matches in this scope)")

        ttp = total_true_pairs[scope]
        print(f"  Total true pairs (this scope): {ttp:,}")
        print(f"  Recovered true pairs: {recovered_true_pairs[scope]:,}")
        if ttp:
            recall = recovered_true_pairs[scope] / ttp
            print(f"  Pair-level recall: {recall:.4%}")
        else:
            print("  Pair-level recall: n/a (no true pairs in this scope)")

        print(f"  False candidate pairs (not true matches): {false_pairs[scope]:,}")

        denom = reduction_denominator[scope]
        if denom:
            rr = 1 - (total_pairs[scope] / denom)
            print(f"  Candidate reduction ratio (vs. brute-force {n_entities:,} x "
                  f"{denom // n_entities:,}): {rr:.6%}")
        else:
            print("  Candidate reduction ratio: n/a (pass --source2/--source3 to enable)")


if __name__ == "__main__":
    main()

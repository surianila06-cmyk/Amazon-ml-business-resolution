#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation / Blocking (CLI)

Builds a name-token inverted index over Source-2/3, generates a candidate
set of S2/S3 IDs for every Source-1 entity, writes candidate_pairs.tsv, and
-- when a ground-truth file is available (the train split) -- evaluates
candidate-generation recall against it.

See `blocking.py` for the blocking algorithm and its rationale (rarest-token
blocking with a posting-list cap, optional country pre-filter -- enabled by
default after an empirical check found 100% country agreement across
153,045 sampled true pairs from train_ground_truth.tsv).

Does NOT modify any dataset file. Does NOT implement matching/scoring --
only candidate generation and (optionally) recall measurement against
ground truth.

Usage (train split, with recall evaluation):
    python candidate_generation/generate_candidates.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --out candidate_generation/output/train_candidate_pairs.tsv

Usage (test split, no ground truth available -- candidate generation only):
    python candidate_generation/generate_candidates.py \\
        --source1 dataset/test/test_source1.tsv \\
        --source2 dataset/test/test_source2.tsv \\
        --source3 dataset/test/test_source3.tsv \\
        --out candidate_generation/output/test_candidate_pairs.tsv
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.blocking import (  # noqa: E402
    DEFAULT_MAX_POSTING_SIZE,
    DEFAULT_MAX_QUERY_TOKENS,
    build_name_token_index,
    candidates_for_name,
)

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass


def load_ground_truth(path):
    """Stream train_ground_truth.tsv into {source1_entity_id: frozenset(matched_ids)}."""
    truth = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)  # header
        for row in reader:
            s1_id, matched_raw = row[0], row[1]
            truth[s1_id] = frozenset(matched_raw.split(",")) if matched_raw.strip() else frozenset()
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--out", required=True, help="Output candidate_pairs.tsv path")
    parser.add_argument("--ground-truth", default=None,
                         help="train_ground_truth.tsv path; if given, recall is evaluated")
    parser.add_argument("--max-posting-size", type=int, default=DEFAULT_MAX_POSTING_SIZE)
    parser.add_argument("--max-query-tokens", type=int, default=DEFAULT_MAX_QUERY_TOKENS)
    parser.add_argument("--no-country-filter", action="store_true",
                         help="Disable the country pre-filter (on by default)")
    args = parser.parse_args()

    country_filter = not args.no_country_filter
    t0 = time.time()

    ground_truth = None
    if args.ground_truth:
        print(f"Loading ground truth from {args.ground_truth} ...")
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"  loaded {len(ground_truth):,} ground-truth rows "
              f"({time.time() - t0:.1f}s elapsed)")

    print(f"\nBuilding name-token index over source2/source3 "
          f"(max_posting_size={args.max_posting_size:,}) ...")
    t1 = time.time()
    index, id_country, index_stats = build_name_token_index(
        [("source2", args.source2), ("source3", args.source3)],
        max_posting_size=args.max_posting_size,
    )
    print(f"Index build time: {time.time() - t1:.1f}s")

    print(f"\nGenerating candidates for {args.source1} "
          f"(max_query_tokens={args.max_query_tokens}, country_filter={country_filter}) ...")
    t2 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_entities = 0
    total_candidates = 0
    max_candidates = 0
    zero_candidate_entities = 0
    candidate_size_counter = Counter()  # bucketed for a rough distribution

    # Recall accounting (only meaningful for entities with ground truth)
    eval_entities_with_truth = 0
    sum_true_matches = 0
    sum_true_positives = 0
    entities_fully_covered = 0
    entities_partially_covered = 0
    entities_zero_covered = 0
    per_entity_recall_sum = 0.0

    with open(args.source1, "r", encoding="utf-8", newline="") as f_in, \
         open(args.out, "w", encoding="utf-8", newline="\n") as f_out:
        reader = csv.reader(f_in, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")
        idx_country = header.index("country")

        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in reader:
            eid = row[idx_id]
            name = row[idx_name]
            country = row[idx_country]

            candidates = candidates_for_name(name, index, args.max_query_tokens)
            if country_filter and country:
                candidates = {c for c in candidates if id_country.get(c) == country}

            n_entities += 1
            n_cand = len(candidates)
            total_candidates += n_cand
            max_candidates = max(max_candidates, n_cand)
            if n_cand == 0:
                zero_candidate_entities += 1
            bucket = min(n_cand // 50, 20)  # 0-49, 50-99, ..., 1000+
            candidate_size_counter[bucket] += 1

            f_out.write(f"{eid}\t{','.join(sorted(candidates))}\n")

            if ground_truth is not None:
                truth = ground_truth.get(eid)
                if truth:  # non-empty: entity has real matches
                    eval_entities_with_truth += 1
                    tp = len(candidates & truth)
                    sum_true_matches += len(truth)
                    sum_true_positives += tp
                    per_entity_recall_sum += tp / len(truth)
                    if tp == len(truth):
                        entities_fully_covered += 1
                    elif tp > 0:
                        entities_partially_covered += 1
                    else:
                        entities_zero_covered += 1

            if n_entities % 200_000 == 0:
                elapsed = time.time() - t2
                print(f"  processed {n_entities:,} S1 entities "
                      f"({elapsed:.1f}s elapsed, {n_entities / max(elapsed, 1e-9):.0f} rows/s)")

    print(f"\nCandidate generation time: {time.time() - t2:.1f}s")
    print(f"Total time: {time.time() - t0:.1f}s")

    print(f"\n{'=' * 70}")
    print("CANDIDATE SET STATISTICS")
    print("=" * 70)
    print(f"S1 entities processed: {n_entities:,}")
    print(f"Total candidate pairs generated: {total_candidates:,}")
    print(f"Mean candidates per entity: {total_candidates / n_entities:.2f}")
    print(f"Max candidates for one entity: {max_candidates:,}")
    print(f"Entities with zero candidates: {zero_candidate_entities:,} "
          f"({zero_candidate_entities / n_entities * 100:.3f}%)")
    print("Candidate-set-size distribution (bucketed by 50, last bucket is 1000+):")
    for bucket in sorted(candidate_size_counter):
        lo = bucket * 50
        label = f"{lo}-{lo + 49}" if bucket < 20 else "1000+"
        print(f"  {label}: {candidate_size_counter[bucket]:,} entities")

    if ground_truth is not None:
        print(f"\n{'=' * 70}")
        print("RECALL EVALUATION (against ground truth)")
        print("=" * 70)
        print(f"S1 entities with >=1 true match (eval denominator): {eval_entities_with_truth:,}")
        print(f"Total true matched pairs (S2/S3 records): {sum_true_matches:,}")
        print(f"True matched pairs recovered by candidates: {sum_true_positives:,}")
        if sum_true_matches:
            micro_recall = sum_true_positives / sum_true_matches
            print(f"MICRO (pair-level) recall: {micro_recall:.4%}")
        if eval_entities_with_truth:
            macro_recall = per_entity_recall_sum / eval_entities_with_truth
            print(f"MACRO (entity-level average) recall: {macro_recall:.4%}")
            print(f"Entities with ALL true matches recovered: {entities_fully_covered:,} "
                  f"({entities_fully_covered / eval_entities_with_truth * 100:.3f}%)")
            print(f"Entities with SOME true matches recovered: {entities_partially_covered:,} "
                  f"({entities_partially_covered / eval_entities_with_truth * 100:.3f}%)")
            print(f"Entities with ZERO true matches recovered: {entities_zero_covered:,} "
                  f"({entities_zero_covered / eval_entities_with_truth * 100:.3f}%)")

    print(f"\nOutput written to: {args.out}")


if __name__ == "__main__":
    main()

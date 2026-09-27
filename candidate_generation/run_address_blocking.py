#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation, Stage 2 (CLI)
ADDRESS-BASED BLOCKING ONLY (independent experiment).

Builds address-token indexes for Source-2 and Source-3 separately, then for
every Source-1 entity retrieves address-token matches from each and writes
the combined candidate set to a candidate_pairs.tsv-format file.

This is an EXPERIMENTAL output, not the project's final candidate_pairs.tsv
submission artifact -- it measures address blocking's standalone
contribution, before any combination with Stage 1 (exact-name blocking).

Ground truth is never read by this script -- it only touches
source1/source2/source3. Use evaluate_candidates.py separately to score the
output against train_ground_truth.tsv.

Usage:
    python candidate_generation/run_address_blocking.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --out candidate_generation/output/train_candidate_pairs_address_experiment.tsv
"""

import argparse
import csv
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import (  # noqa: E402
    DEFAULT_MIN_SHARED_TOKENS,
    build_address_token_index,
    generate_address_candidates,
)

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--out", required=True, help="Output candidate_pairs.tsv-format path (experimental)")
    parser.add_argument("--min-shared-tokens", type=int, default=DEFAULT_MIN_SHARED_TOKENS,
                         help="Minimum number of shared address tokens required for a candidate "
                              "(AND semantics; default: %(default)s)")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Building address-token index over {args.source2} (Source-2) ...")
    s2_index = build_address_token_index(args.source2)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"\nBuilding address-token index over {args.source3} (Source-3) ...")
    s3_index = build_address_token_index(args.source3)
    print(f"  elapsed: {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"\nGenerating address-token candidates for {args.source1} "
          f"(min_shared_tokens={args.min_shared_tokens}) ...")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_entities = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, s2_cand, s3_cand in generate_address_candidates(
                args.source1, s2_index, s3_index, min_shared_tokens=args.min_shared_tokens):
            combined = sorted(s2_cand) + sorted(s3_cand)
            f_out.write(f"{eid}\t{','.join(combined)}\n")
            n_entities += 1

    print(f"  elapsed: {time.time() - t2:.1f}s")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"S1 entities processed: {n_entities:,}")
    print(f"Output written to: {args.out}")


if __name__ == "__main__":
    main()

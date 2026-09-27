#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation, Stage 1 (CLI)
EXACT NORMALIZED-NAME BLOCKING ONLY.

Builds a normalized_business_name -> [entity_id] index for Source-2 and one
for Source-3 (kept separate, per the project's blocking contract), then for
every Source-1 entity retrieves exact-normalized-name matches from each and
writes the combined candidate set to a candidate_pairs.tsv-format file.

Ground truth is never read by this script -- it only touches
source1/source2/source3. Use evaluate_candidates.py separately to score the
output against train_ground_truth.tsv.

Usage:
    python candidate_generation/run_exact_name_blocking.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --out candidate_generation/output/train_candidate_pairs_exact_name.tsv
"""

import argparse
import csv
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.exact_name_blocking import (  # noqa: E402
    build_exact_name_index,
    generate_exact_match_candidates,
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
    parser.add_argument("--out", required=True, help="Output candidate_pairs.tsv path")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Building exact-name index over {args.source2} (Source-2) ...")
    s2_index = build_exact_name_index(args.source2)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"\nBuilding exact-name index over {args.source3} (Source-3) ...")
    s3_index = build_exact_name_index(args.source3)
    print(f"  elapsed: {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"\nGenerating exact-name candidates for {args.source1} ...")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_entities = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, s2_ids, s3_ids in generate_exact_match_candidates(args.source1, s2_index, s3_index):
            combined = sorted(s2_ids) + sorted(s3_ids)
            f_out.write(f"{eid}\t{','.join(combined)}\n")
            n_entities += 1

    print(f"  elapsed: {time.time() - t2:.1f}s")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"S1 entities processed: {n_entities:,}")
    print(f"Output written to: {args.out}")


if __name__ == "__main__":
    main()

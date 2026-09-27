"""
Ground-truth inspection for the Amazon ML Challenge 2026 Business Entity
Resolution problem (Role 2: Candidate Generation / Blocking).

This script ONLY inspects train_ground_truth.tsv (and, for cross-reference,
streams entity_id values out of train_source1.tsv). It does not modify any
raw dataset file, does not build candidate pairs, and does not implement
any blocking/matching logic.

Memory approach:
    - train_ground_truth.tsv is read in a single streaming pass
      (csv.reader over a text file handle, never read() into memory).
    - train_source1.tsv is read in a second streaming pass, collecting
      ONLY entity_id values into a set (not full rows) for cross-reference.
    - The two large source2/source3 files are NOT opened by this script.
    - No full DataFrame / full-file-in-memory loading is used.

Usage:
    python inspect_ground_truth.py
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

# <repo root>/dataset/train, derived from this script's location (candidate_generation/analysis/)
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset" / "train")
GT_PATH = os.path.join(DATA_DIR, "train_ground_truth.tsv")
SOURCE1_PATH = os.path.join(DATA_DIR, "train_source1.tsv")

EXPECTED_HEADER = ["source1_entity_id", "matched_entity_ids"]
DELIMITER = "\t"

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def detect_delimiter(path):
    with open(path, "rb") as f:
        raw_first_line = f.readline()
    line_str = raw_first_line.decode("utf-8", errors="replace").rstrip("\r\n")
    candidates = ["\t", ",", ";", "|"]
    counts = {c: line_str.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best, counts, line_str


def collect_source1_ids(path):
    """Stream train_source1.tsv, returning the set of its entity_id values.
    Only the entity_id column is retained (not full rows)."""
    ids = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx = header.index("entity_id")
        for row in reader:
            ids.add(row[idx])
    return ids


def main():
    print("Amazon ML Challenge 2026 - Business Entity Resolution")
    print("Role 2 (Candidate Generation / Blocking) - GROUND TRUTH INSPECTION ONLY")
    print(f"Ground truth file: {GT_PATH}")
    print("NOTE: no files are being modified; this is a read-only pass.\n")

    exists = os.path.isfile(GT_PATH)
    print(f"File exists at expected location: {exists}")
    if not exists:
        print("Aborting: ground truth file not found.")
        return
    size_mb = os.path.getsize(GT_PATH) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    delim, delim_counts, raw_header_line = detect_delimiter(GT_PATH)
    print(f"Detected delimiter: {repr(delim)} (char counts in header: {delim_counts})")
    print(f"Raw header line: {raw_header_line!r}")

    # ---- Single streaming pass over ground truth ----
    row_count = 0
    structural_mismatch_rows = 0
    header = None

    gt_source1_ids_seen = set()
    duplicate_source1_id_rows = 0

    zero_match_count = 0
    single_match_count = 0
    multi_match_count = 0
    total_matched_records = 0
    max_matches = 0
    match_count_distribution = Counter()

    matched_prefix_counter = Counter()
    non_s2_s3_examples = []

    rows_with_duplicate_matched_ids = 0
    duplicate_matched_id_examples = []

    rows_with_whitespace_issues = 0
    whitespace_examples = []

    source1_prefix_ok_rows = 0
    source1_prefix_bad_examples = []

    with open(GT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delim, quoting=csv.QUOTE_NONE)
        header = next(reader)
        header_ok = header == EXPECTED_HEADER
        col_index = {c: i for i, c in enumerate(header)}
        idx_s1 = col_index.get("source1_entity_id", 0)
        idx_matched = col_index.get("matched_entity_ids", 1)

        for row in reader:
            row_count += 1
            if len(row) != len(header):
                structural_mismatch_rows += 1
                continue

            s1_id = row[idx_s1]
            matched_raw = row[idx_matched]

            if s1_id in gt_source1_ids_seen:
                duplicate_source1_id_rows += 1
            else:
                gt_source1_ids_seen.add(s1_id)

            if s1_id.startswith("S1-"):
                source1_prefix_ok_rows += 1
            elif len(source1_prefix_bad_examples) < 10:
                source1_prefix_bad_examples.append(s1_id)

            # whitespace check: does raw field differ from a stripped/rejoined version?
            if matched_raw != matched_raw.strip():
                rows_with_whitespace_issues += 1
                if len(whitespace_examples) < 5:
                    whitespace_examples.append((s1_id, repr(matched_raw)))

            if matched_raw.strip() == "":
                match_ids = []
            else:
                match_ids = matched_raw.split(",")
                # detect embedded whitespace around individual ids
                for mid in match_ids:
                    if mid != mid.strip():
                        rows_with_whitespace_issues += 1
                        if len(whitespace_examples) < 5:
                            whitespace_examples.append((s1_id, repr(matched_raw)))
                        break

            n_matches = len(match_ids)
            total_matched_records += n_matches
            max_matches = max(max_matches, n_matches)
            match_count_distribution[n_matches] += 1

            if n_matches == 0:
                zero_match_count += 1
            elif n_matches == 1:
                single_match_count += 1
            else:
                multi_match_count += 1

            if len(match_ids) != len(set(match_ids)) and match_ids:
                rows_with_duplicate_matched_ids += 1
                if len(duplicate_matched_id_examples) < 5:
                    duplicate_matched_id_examples.append((s1_id, matched_raw))

            for mid in match_ids:
                prefix = mid.split("-", 1)[0] if "-" in mid else mid
                matched_prefix_counter[prefix] += 1
                if prefix not in ("S2", "S3") and len(non_s2_s3_examples) < 10:
                    non_s2_s3_examples.append((s1_id, mid))

    print(f"\nHeader matches expected {EXPECTED_HEADER}: {header_ok}")
    print(f"Actual header: {header}")
    print(f"Row count (data rows, excluding header): {row_count:,}")
    print(f"Structural mismatch rows (column count != header count): {structural_mismatch_rows:,}")

    print(f"\nDuplicate source1_entity_id rows (full-file exact check): {duplicate_source1_id_rows:,}")
    print(f"Unique source1_entity_id count in ground truth: {len(gt_source1_ids_seen):,}")

    print(f"\nsource1_entity_id rows with 'S1-' prefix: {source1_prefix_ok_rows:,} / {row_count:,}")
    if source1_prefix_bad_examples:
        print(f"  Examples WITHOUT 'S1-' prefix: {source1_prefix_bad_examples}")

    print(f"\nRows with zero matches (empty matched_entity_ids): {zero_match_count:,} "
          f"({zero_match_count / row_count * 100:.3f}%)")
    print(f"Rows with exactly one match: {single_match_count:,} "
          f"({single_match_count / row_count * 100:.3f}%)")
    print(f"Rows with multiple matches: {multi_match_count:,} "
          f"({multi_match_count / row_count * 100:.3f}%)")
    print(f"Total true matched S2/S3 records (sum across all rows): {total_matched_records:,}")
    print(f"Max matches for a single S1 entity: {max_matches:,}")
    print(f"Mean matches per S1 entity: {total_matched_records / row_count:.3f}")

    print("\nDistribution of number of matches per S1 entity (match_count -> num_rows), "
          "showing up to 25 smallest counts, then a summary of the tail:")
    sorted_counts = sorted(match_count_distribution.items())
    for k, v in sorted_counts[:25]:
        print(f"  {k} matches: {v:,} rows")
    if len(sorted_counts) > 25:
        tail_rows = sum(v for k, v in sorted_counts[25:])
        tail_range = (sorted_counts[25][0], sorted_counts[-1][0])
        print(f"  ... {len(sorted_counts) - 25} more distinct match-counts "
              f"(range {tail_range[0]}-{tail_range[1]}), totaling {tail_rows:,} rows")

    print(f"\nmatched_entity_ids prefix breakdown (all matched IDs, all rows): "
          f"{matched_prefix_counter.most_common()}")
    if non_s2_s3_examples:
        print(f"Examples of matched IDs NOT prefixed S2/S3: {non_s2_s3_examples}")
    else:
        print("All matched IDs use only S2-/S3- prefixes. No anomalies found.")

    print(f"\nRows with duplicate matched IDs within the same row: {rows_with_duplicate_matched_ids:,}")
    if duplicate_matched_id_examples:
        print("Examples:")
        for s1_id, raw in duplicate_matched_id_examples:
            print(f"  - {s1_id}: {raw}")

    print(f"\nRows with whitespace issues (leading/trailing/around-comma spaces): "
          f"{rows_with_whitespace_issues:,}")
    if whitespace_examples:
        print("Examples:")
        for s1_id, raw in whitespace_examples:
            print(f"  - {s1_id}: {raw}")

    # ---- Cross-reference with train_source1.tsv ----
    print(f"\n{'=' * 70}")
    print("Cross-referencing with train_source1.tsv entity_id values")
    print("=" * 70)
    source1_ids = collect_source1_ids(SOURCE1_PATH)
    print(f"Unique entity_id count in train_source1.tsv: {len(source1_ids):,}")

    s1_missing_from_gt = source1_ids - gt_source1_ids_seen
    gt_ids_not_in_s1 = gt_source1_ids_seen - source1_ids

    print(f"S1 entities present in train_source1.tsv but with NO ground-truth row: "
          f"{len(s1_missing_from_gt):,}")
    if s1_missing_from_gt:
        sample = list(s1_missing_from_gt)[:10]
        print(f"  Example missing IDs: {sample}")

    print(f"Ground-truth rows whose source1_entity_id does NOT appear in "
          f"train_source1.tsv: {len(gt_ids_not_in_s1):,}")
    if gt_ids_not_in_s1:
        sample = list(gt_ids_not_in_s1)[:10]
        print(f"  Example extraneous IDs: {sample}")

    print(f"\nEvery train_source1.tsv entity has exactly one ground-truth row: "
          f"{len(s1_missing_from_gt) == 0 and len(gt_ids_not_in_s1) == 0 and duplicate_source1_id_rows == 0}")


if __name__ == "__main__":
    main()

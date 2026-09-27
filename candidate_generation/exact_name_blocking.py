"""
ML Challenge 2026 -- Role 2: Candidate Generation / Blocking
STAGE 1 ONLY: exact normalized-business_name blocking.

For every Source-1 entity, retrieve every Source-2 / Source-3 record whose
normalized business_name is EXACTLY equal (string equality after
`utils.preprocessing.normalize_business_name`). No fuzzy matching, no
token/rare-token selection, no address blocking, no country filtering, and
no thresholds -- exact-name blocking has no threshold to tune.

This module builds two SEPARATE lookup indexes (one for Source-2, one for
Source-3), each mapping normalized_business_name -> [entity_id, ...], then
looks each Source-1 query up in both. Ground truth is never read here --
this module only ever sees source1/source2/source3, never
train_ground_truth.tsv (see evaluate_candidates.py for the evaluation-only
use of ground truth).

Memory approach:
    - Each source file is streamed once (csv.reader over a text handle).
    - Source-2 and Source-3 are indexed one at a time (never both files
      open/read simultaneously) via `build_exact_name_index`, though their
      resulting index dicts do need to coexist afterwards so Source-1
      queries can check both -- that in-memory pair of dicts is the
      unavoidable cost of blocking (a lookup structure must be resident).
    - Source-1 is streamed row by row; candidates are generated and yielded
      immediately, never buffered into a big intermediate list.
"""

import csv
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preprocessing import normalize_business_name  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def build_exact_name_index(path, progress_every=1_000_000, log=print):
    """Stream a source-style TSV (entity_id, business_name, business_address,
    country) and return {normalized_business_name: [entity_id, ...]}.

    Records whose normalized name is empty (missing/null-like business_name)
    are excluded from the index -- an empty string is not a meaningful
    blocking key and would otherwise let every empty-name query collide with
    every other empty-name record.
    """
    index = defaultdict(list)
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")

        for row in reader:
            norm = normalize_business_name(row[idx_name])
            if norm:
                index[norm].append(row[idx_id])
            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  indexed {n:,} rows from {os.path.basename(path)}...")

    log(f"  done indexing {os.path.basename(path)}: {n:,} rows, "
        f"{len(index):,} distinct normalized names.")
    return dict(index)


def generate_exact_match_candidates(source1_path, s2_index, s3_index,
                                     progress_every=200_000, log=print):
    """Stream `source1_path`; for each row yield
    (entity_id, s2_candidate_ids, s3_candidate_ids) -- the exact-match
    lookups of that row's normalized business_name in `s2_index` /
    `s3_index`, each a (possibly empty) list of original entity_id strings
    (e.g. 'S2-681193310', 'S3-11291185') with prefixes untouched.
    """
    with open(source1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")

        n = 0
        for row in reader:
            eid = row[idx_id]
            norm = normalize_business_name(row[idx_name])

            s2_ids = s2_index.get(norm, []) if norm else []
            s3_ids = s3_index.get(norm, []) if norm else []

            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  generated candidates for {n:,} S1 entities...")

            yield eid, s2_ids, s3_ids

        log(f"  done: generated candidates for {n:,} S1 entities.")

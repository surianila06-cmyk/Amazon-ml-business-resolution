#!/usr/bin/env python3
"""
Small synthetic unit test for exact_name_blocking.py, run BEFORE using it on
the full training data. Builds tiny hand-crafted source1/source2/source3
TSVs in a temp directory, exercises build_exact_name_index /
generate_exact_match_candidates directly, and asserts the expected exact
normalized-name matches (and non-matches).

Usage:
    python candidate_generation/tests/test_exact_name_blocking.py
"""

import csv
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.exact_name_blocking import (  # noqa: E402
    build_exact_name_index,
    generate_exact_match_candidates,
)

HEADER = ["entity_id", "business_name", "business_address", "country"]


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        for row in rows:
            w.writerow(row)


def _silent_log(*args, **kwargs):
    pass


class TestExactNameBlocking(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.s1_path = os.path.join(self.tmpdir.name, "source1.tsv")
        self.s2_path = os.path.join(self.tmpdir.name, "source2.tsv")
        self.s3_path = os.path.join(self.tmpdir.name, "source3.tsv")

        _write_tsv(self.s2_path, [
            # exact-name (after normalization) match candidate for S1-1
            ["S2-1", "Global Tech Inc", "1 Main St, City, ST", "US"],
            # another record with the SAME normalized name as S2-1
            # -- exact blocking must return BOTH, not dedupe to one
            ["S2-2", "GLOBAL TECH, INC.", "2 Other St, City, ST", "US"],
            # unrelated name -- should never appear as a candidate for S1-1
            ["S2-3", "Totally Different Co", "9 Nowhere Ave", "US"],
            # empty business_name -- must be excluded from the index
            ["S2-4", "", "3 Blank St", "US"],
        ])
        _write_tsv(self.s3_path, [
            # legal-suffix + punctuation normalization should make this match
            # S1-2's "XYZ Private Limited"
            ["S3-1", "XYZ Pvt. Ltd.", "10 Some Rd", "India"],
            ["S3-2", "Unrelated Traders", "11 Some Rd", "India"],
        ])
        _write_tsv(self.s1_path, [
            # normalizes to "global tech inc" (commas/periods/whitespace
            # cleaned up) -- should exact-match S2-1 and S2-2, not S2-3
            ["S1-1", "Global   Tech,  Inc.", "5 Query St, City, ST", "US"],
            # normalizes to "xyz pvt ltd" via legal-suffix normalization --
            # should exact-match S3-1 only
            ["S1-2", "XYZ Private Limited", "20 Query Rd", "India"],
            # no plausible match in either source
            ["S1-3", "Nonexistent Business Name Zzz", "30 Query Ln", "US"],
            # null-like business_name -- must yield zero candidates, not
            # match S2-4's empty-name record
            ["S1-4", "N/A", "40 Query Blvd", "US"],
        ])

        self.s2_index = build_exact_name_index(self.s2_path, log=_silent_log)
        self.s3_index = build_exact_name_index(self.s3_path, log=_silent_log)
        self.results = {
            eid: (sorted(s2_ids), sorted(s3_ids))
            for eid, s2_ids, s3_ids in generate_exact_match_candidates(
                self.s1_path, self.s2_index, self.s3_index, log=_silent_log
            )
        }

    def test_index_excludes_empty_names(self):
        self.assertNotIn("", self.s2_index)

    def test_exact_match_after_punctuation_and_whitespace_normalization(self):
        s2_ids, s3_ids = self.results["S1-1"]
        self.assertEqual(s2_ids, ["S2-1", "S2-2"])
        self.assertEqual(s3_ids, [])

    def test_exact_match_after_legal_suffix_normalization(self):
        s2_ids, s3_ids = self.results["S1-2"]
        self.assertEqual(s2_ids, [])
        self.assertEqual(s3_ids, ["S3-1"])

    def test_no_match_returns_empty_lists(self):
        s2_ids, s3_ids = self.results["S1-3"]
        self.assertEqual(s2_ids, [])
        self.assertEqual(s3_ids, [])

    def test_null_like_query_name_yields_no_candidates(self):
        s2_ids, s3_ids = self.results["S1-4"]
        self.assertEqual(s2_ids, [])
        self.assertEqual(s3_ids, [])

    def test_original_id_prefixes_are_preserved(self):
        s2_ids, s3_ids = self.results["S1-1"]
        self.assertTrue(all(i.startswith("S2-") for i in s2_ids))
        s2_ids2, s3_ids2 = self.results["S1-2"]
        self.assertTrue(all(i.startswith("S3-") for i in s3_ids2))

    def test_every_s1_entity_present_even_with_zero_candidates(self):
        self.assertEqual(set(self.results.keys()), {"S1-1", "S1-2", "S1-3", "S1-4"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

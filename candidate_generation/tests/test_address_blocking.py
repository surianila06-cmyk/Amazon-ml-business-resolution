#!/usr/bin/env python3
"""
Small synthetic unit test for address_blocking.py, run BEFORE using it on
the full training data. Builds tiny hand-crafted source1/source2/source3
TSVs in a temp directory and asserts the expected address-token-blocking
behavior: numeric/word key filtering, missing-address handling, and that
overly common ("stopword") tokens never produce candidates on their own.

Usage:
    python candidate_generation/tests/test_address_blocking.py
"""

import csv
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import (  # noqa: E402
    address_blocking_keys,
    build_address_token_index,
    generate_address_candidates,
    is_keepable_address_token,
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


class TestTokenFiltering(unittest.TestCase):
    """Unit tests for the token-level filtering rules in isolation."""

    def test_numeric_token_below_min_digits_is_dropped(self):
        self.assertFalse(is_keepable_address_token("12"))   # 2 digits
        self.assertFalse(is_keepable_address_token("7"))    # 1 digit

    def test_numeric_token_at_or_above_min_digits_is_kept(self):
        self.assertTrue(is_keepable_address_token("123"))       # 3-digit house number
        self.assertTrue(is_keepable_address_token("560001"))    # 6-digit PIN code
        self.assertTrue(is_keepable_address_token("a-212"))     # hyphenated, 3 digits

    def test_stopword_is_dropped(self):
        for tok in ("rd", "st", "delhi", "null", "no", "unit"):
            self.assertFalse(is_keepable_address_token(tok), f"{tok!r} should be a stopword")

    def test_single_char_word_token_is_dropped(self):
        self.assertFalse(is_keepable_address_token("a"))
        self.assertFalse(is_keepable_address_token("c"))

    def test_distinctive_word_token_is_kept(self):
        self.assertTrue(is_keepable_address_token("westchester"))

    def test_address_blocking_keys_on_empty_address(self):
        self.assertEqual(address_blocking_keys(""), [])


class TestAddressBlockingCandidates(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.s1_path = os.path.join(self.tmpdir.name, "source1.tsv")
        self.s2_path = os.path.join(self.tmpdir.name, "source2.tsv")
        self.s3_path = os.path.join(self.tmpdir.name, "source3.tsv")

        _write_tsv(self.s2_path, [
            # shares house number 1795 (>=3 digits) with S1-1's address
            ["S2-1", "Some Shop", "1795 Westchester Drive, High Point, NC", "US"],
            # shares only stopword tokens ('rd'/'st'/'new'/'city') with S1-1
            # after normalization -- must NOT be retrieved via those alone
            ["S2-2", "Other Shop", "1 New City Rd", "US"],
            # unrelated address entirely
            ["S2-3", "Random Biz", "999999 Faraway Blvd, Nowhere", "US"],
            # missing address -- must never be indexed / never a candidate
            ["S2-4", "No Address Biz", "", "US"],
            # shares exactly ONE qualifying token with S1-1 ('1795') but not
            # 'westchester'/'high'/'point' -- under AND semantics
            # (min_shared_tokens=2) this must NOT be a candidate, even
            # though a union/OR design would have wrongly included it
            ["S2-5", "Another Shop", "1795 Random Boulevard, Nowhere", "US"],
        ])
        _write_tsv(self.s3_path, [
            # shares PIN code 560001 with S1-2's address
            ["S3-1", "Some Firm Pvt Ltd", "12 MG Road, Bangalore, 560001", "India"],
            ["S3-2", "Unrelated Firm", "45 Other Road, Chennai, 600001", "India"],
        ])
        _write_tsv(self.s1_path, [
            ["S1-1", "Query Shop", "1795 Westchester Drive, High Point, NC", "US"],
            ["S1-2", "Query Firm", "77 MG Road, Bangalore, 560001", "India"],
            # only short numeric + stopword tokens -- should get zero candidates
            ["S1-3", "Query Nothing", "1 New City Rd", "US"],
            # missing address entirely
            ["S1-4", "Query Missing Address", "", "US"],
        ])

        self.s2_index = build_address_token_index(self.s2_path, log=_silent_log)
        self.s3_index = build_address_token_index(self.s3_path, log=_silent_log)
        self.results = {
            eid: (s2_cand, s3_cand)
            for eid, s2_cand, s3_cand in generate_address_candidates(
                self.s1_path, self.s2_index, self.s3_index, log=_silent_log
            )
        }

    def test_shared_house_number_produces_candidate(self):
        s2_cand, s3_cand = self.results["S1-1"]
        self.assertIn("S2-1", s2_cand)
        # S2-3's address shares no qualifying token with S1-1
        self.assertNotIn("S2-3", s2_cand)

    def test_single_shared_token_is_not_enough_under_and_semantics(self):
        # Regression test for the union/OR-based blowup: S2-5 shares only
        # '1795' with S1-1 -- one shared token must not be a candidate.
        s2_cand, _ = self.results["S1-1"]
        self.assertNotIn("S2-5", s2_cand)

    def test_shared_pin_code_produces_candidate(self):
        s2_cand, s3_cand = self.results["S1-2"]
        self.assertIn("S3-1", s3_cand)
        self.assertNotIn("S3-2", s3_cand)

    def test_stopword_only_overlap_produces_no_candidates(self):
        # S1-3's address ("1 new city rd") shares only stopword/short-numeric
        # tokens with S2-2's ("1 new city rd") -- must yield zero candidates
        s2_cand, s3_cand = self.results["S1-3"]
        self.assertEqual(s2_cand, set())
        self.assertEqual(s3_cand, set())

    def test_missing_query_address_yields_no_candidates(self):
        s2_cand, s3_cand = self.results["S1-4"]
        self.assertEqual(s2_cand, set())
        self.assertEqual(s3_cand, set())

    def test_missing_target_address_never_indexed(self):
        # S2-4 has no address at all -- it must not appear under any token
        all_candidates = set()
        for s2_cand, _ in self.results.values():
            all_candidates |= s2_cand
        self.assertNotIn("S2-4", all_candidates)

    def test_original_id_prefixes_preserved(self):
        s2_cand, s3_cand = self.results["S1-1"]
        self.assertTrue(all(i.startswith("S2-") for i in s2_cand))

    def test_every_s1_entity_present(self):
        self.assertEqual(set(self.results.keys()), {"S1-1", "S1-2", "S1-3", "S1-4"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

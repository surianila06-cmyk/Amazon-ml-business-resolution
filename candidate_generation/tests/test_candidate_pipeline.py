#!/usr/bin/env python3
"""
Synthetic unit tests for the production candidate pipeline
(candidate_pipeline.py + run_candidate_pipeline.py): each channel's locked
rules, DISTINCT-token address semantics, per-source posting cap, top-k
ordering and tie-breaks, same-country restriction confined to the rare-name
channel, union/identity handling, output format, determinism and resume.

Usage:
    python candidate_generation/tests/test_candidate_pipeline.py
"""

import csv
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.candidate_pipeline import (  # noqa: E402
    PipelineConfig,
    build_target_index,
    collect_s1_vocabulary,
    generate_candidates,
    iter_source_rows,
    load_address_stopwords,
    select_rare_token,
)
from candidate_generation.run_candidate_pipeline import build_parser, run  # noqa: E402

HEADER = ["entity_id", "business_name", "business_address", "country"]
STOPWORDS = load_address_stopwords()


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)


def _silent(*args, **kwargs):
    pass


class PipelineCase(unittest.TestCase):
    """Builds the index from in-test rows and returns {s1_id: S1Candidates}."""

    def run_channels(self, s1, s2, s3, cfg=PipelineConfig()):
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, f"s{i}.tsv") for i in (1, 2, 3)]
            for p, rows in zip(paths, (s1, s2, s3)):
                _write_tsv(p, rows)
            vocab = collect_s1_vocabulary(iter_source_rows(paths[0]), STOPWORDS)
            idx = build_target_index(paths[1:], vocab, STOPWORDS, cfg, log=_silent)
            self.idx = idx
            return {c.entity_id: c for c in generate_candidates(iter_source_rows(paths[0]), idx, STOPWORDS, cfg)}


class TestExactName(PipelineCase):
    def test_match_after_normalization(self):
        out = self.run_channels([["S1-1", "Acme Widgets, Inc.", "", "US"]],
                                [["S2-1", "ACME WIDGETS INC", "", "US"]], [])
        self.assertEqual(out["S1-1"].exact, {"S2-1"})

    def test_empty_name_never_matches(self):
        out = self.run_channels([["S1-1", "", "", "US"]], [["S2-1", "", "", "US"]], [])
        self.assertEqual(out["S1-1"].exact, frozenset())


class TestAddressChannel(PipelineCase):
    def test_two_distinct_shared_tokens_retrieve(self):
        out = self.run_channels([["S1-1", "a1", "4521 Quizzle Fernwick", "US"]],
                                [["S2-1", "b1", "4521 Quizzle Brambleton", "US"]], [])
        self.assertEqual(out["S1-1"].address, {"S2-1"})

    def test_repeated_single_token_is_not_enough(self):
        # Only ONE distinct shared token (4521), repeated on both sides.
        out = self.run_channels([["S1-1", "a1", "4521 4521 Brambleton", "US"]],
                                [["S2-1", "b1", "4521 4521 Otherplace", "US"]], [])
        self.assertEqual(out["S1-1"].address, frozenset())

    def test_stopword_overlap_does_not_count(self):
        out = self.run_channels([["S1-1", "a1", "4521 Delhi Mumbai", "India"]],
                                [["S2-1", "b1", "9999 Delhi Mumbai", "India"]], [])
        self.assertEqual(out["S1-1"].address, frozenset())

    def test_posting_cap_is_per_source_and_first_seen(self):
        cfg = PipelineConfig(posting_cap=2)
        s2 = [[f"S2-{i}", f"n{i}", "7777 Zentrix", "US"] for i in (1, 2, 3)]
        out = self.run_channels([["S1-1", "q", "7777 Zentrix", "US"]], s2,
                                [["S3-1", "m", "7777 Zentrix", "US"]], cfg)
        self.assertEqual(out["S1-1"].address, {"S2-1", "S2-2", "S3-1"})

    def test_top_k_orders_by_jaccard_then_entity_id(self):
        cfg = PipelineConfig(top_k=2)
        s2 = [["S2-3", "x", "1111 Aaaq Bbbq", "US"],   # J = 1
              ["S2-2", "x", "1111 Aaaq Zzzq", "US"],   # J = 0.5
              ["S2-1", "x", "1111 Aaaq Yyyq", "US"]]   # J = 0.5, smaller id wins the tie
        out = self.run_channels([["S1-1", "q", "1111 Aaaq Bbbq", "US"]], s2, [], cfg)
        self.assertEqual(out["S1-1"].address, {"S2-3", "S2-1"})

    def test_equal_jaccard_prefers_more_shared_tokens(self):
        cfg = PipelineConfig(top_k=1)
        s2 = [["S2-1", "x", "Alfq Brvq", "US"],                                    # 2/4 = 0.5
              ["S2-9", "x", "Alfq Brvq Chrq Dltq Ecq Fxq Glq Htq", "US"]]           # 4/8 = 0.5
        out = self.run_channels([["S1-1", "q", "Alfq Brvq Chrq Dltq", "US"]], s2, [], cfg)
        self.assertEqual(out["S1-1"].address, {"S2-9"})

    def test_missing_address_yields_nothing(self):
        out = self.run_channels([["S1-1", "q", "", "US"]], [["S2-1", "x", "", "US"]], [])
        self.assertEqual(out["S1-1"].address, frozenset())


class TestRareNameChannel(PipelineCase):
    def test_same_country_applies_only_to_rare_channel(self):
        s2 = [["S2-10", "Zyxcorp Trading", "", "US"],
              ["S2-11", "Zyxcorp Services", "", "India"]]
        s3 = [["S3-12", "Zyxcorp", "", "India"],                  # exact name, other country
              ["S3-13", "unrelated", "5555 Qwertyx Road", "India"]]  # address match, other country
        out = self.run_channels([["S1-1", "Zyxcorp", "5555 Qwertyx Lane", "US"]], s2, s3)
        c = out["S1-1"]
        self.assertEqual(c.rare, {"S2-10"})
        self.assertEqual(c.exact, {"S3-12"})
        self.assertEqual(c.address, {"S3-13"})
        self.assertEqual(c.union, ["S2-10", "S3-12", "S3-13"])

    def test_rarest_eligible_token_with_df_cap_and_tiebreak(self):
        cfg = PipelineConfig(rare_max_df=2)
        s2 = [["S2-1", "Alphaq one", "", "US"], ["S2-2", "Alphaq two", "", "US"], ["S2-3", "Alphaq x", "", "US"],
              ["S2-4", "Gammaq", "", "US"], ["S2-5", "Gammaq", "", "US"],
              ["S2-6", "Betaq", "", "US"], ["S2-7", "Betaq", "", "US"]]
        out = self.run_channels([["S1-1", "Alphaq Betaq Gammaq", "", "US"]], s2, [], cfg)
        self.assertEqual(select_rare_token(self.idx, {"alphaq", "betaq", "gammaq"}, cfg), (2, "betaq"))
        self.assertNotIn("alphaq", self.idx.name_postings)
        self.assertEqual(out["S1-1"].rare, {"S2-6", "S2-7"})

    def test_no_eligible_token_or_empty_country(self):
        out = self.run_channels([["S1-1", "Solo", "", ""], ["S1-2", "Nowhere", "", "US"]],
                                [["S2-1", "Solo", "", ""]], [])
        self.assertEqual(out["S1-1"].rare, frozenset())   # empty S1 country
        self.assertEqual(out["S1-2"].rare, frozenset())   # token DF 0


class TestUnionAndIdentity(PipelineCase):
    def test_union_dedups_by_entity_id_and_keeps_s2_s3_distinct(self):
        s2 = [["S2-5", "Kappa Store", "8080 Wobblex Quarn", "US"]]
        s3 = [["S3-5", "Kappa Store", "1 x", "US"]]
        c = self.run_channels([["S1-1", "Kappa Store", "8080 Wobblex Quarn", "US"]], s2, s3)["S1-1"]
        self.assertIn("S2-5", c.exact & c.address)
        self.assertEqual(c.union, ["S2-5", "S3-5"])


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.paths = [os.path.join(d, f"s{i}.tsv") for i in (1, 2, 3)]
        _write_tsv(self.paths[0], [["S1-1", "Kappa Store", "8080 Wobblex Quarn", "US"],
                                   ["S1-2", "Nothing Matches", "", "US"],
                                   ["S1-3", "Zyxcorp", "5555 Qwertyx Lane", "US"]])
        _write_tsv(self.paths[1], [["S2-5", "Kappa Store", "8080 Wobblex Quarn", "US"],
                                   ["S2-10", "Zyxcorp Trading", "", "US"]])
        _write_tsv(self.paths[2], [["S3-5", "Kappa Store", "", "US"],
                                   ["S3-13", "other", "5555 Qwertyx Road", "India"]])

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, out, *extra):
        return build_parser().parse_args(["--source1", self.paths[0], "--source2", self.paths[1],
                                          "--source3", self.paths[2], "--out", out, *extra])

    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_output_format_every_s1_present(self):
        out = os.path.join(self.tmp.name, "pilot.tsv")
        run(self._args(out), log=_silent)
        lines = self._read(out).splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
        self.assertEqual(lines[1:], ["S1-1\tS2-5,S3-5", "S1-2\t", "S1-3\tS2-10,S3-13"])

    def test_deterministic_across_runs(self):
        a, b = (os.path.join(self.tmp.name, n) for n in ("a.tsv", "b.tsv"))
        run(self._args(a), log=_silent)
        run(self._args(b), log=_silent)
        self.assertEqual(self._read(a), self._read(b))

    def test_resume_after_interruption_matches_full_run(self):
        full, resumed = (os.path.join(self.tmp.name, n) for n in ("full.tsv", "resumed.tsv"))
        run(self._args(full, "--chunk-size", "1"), log=_silent)
        self.assertIsNone(run(self._args(resumed, "--chunk-size", "1"), log=_silent, _stop_after_chunks=1))
        self.assertFalse(os.path.exists(resumed))
        run(self._args(resumed, "--chunk-size", "1", "--resume"), log=_silent)
        self.assertEqual(self._read(full), self._read(resumed))

    def test_refuses_final_candidate_pairs_without_flag(self):
        with self.assertRaises(SystemExit):
            run(self._args(os.path.join(self.tmp.name, "candidate_pairs.tsv")), log=_silent)


if __name__ == "__main__":
    unittest.main(verbosity=2)

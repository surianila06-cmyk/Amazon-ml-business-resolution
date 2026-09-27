#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation
Complementarity of Stage 1 (exact-name blocking) and Stage 2 (address
blocking) on the SAME 20,000 stride-sampled S1 entities used by
run_address_blocking_bounded_experiment.py and
run_address_blocking_cap_stopword_experiment.py.

Stage 1: REUSES the existing full-run output
  candidate_generation/output/train_candidate_pairs_exact_name.tsv
  (streamed, filtered to the sampled S1 ids; not regenerated).
Stage 2: the currently selected address configuration, generated for the
  sample only: MIN_SHARED_TOKENS=2, posting cap 5,000, stopwords =
  ADDRESS_WORD_STOPWORDS UNION {word tokens, len >= 2, DF > 0.5%} (161
  tokens) from analysis/address_word_token_df.tsv. The S2/S3 indexes hold
  only the tokens the sample can query; each token's posting list is still
  built in file order with the same first-seen cap, so candidates are
  identical to a full-index run. address_blocking.py is NOT modified.

Ground truth is used only for scoring. No candidate_pairs.tsv is written;
the only output is analysis/exact_vs_address_union_20k.txt.

Usage (from repo root):
    python candidate_generation/analysis/exact_vs_address_union_20k.py
"""
import csv
import os
import statistics
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import address_blocking_keys  # noqa: E402
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    load_ground_truth_for_ids,
    sample_source1,
)
from candidate_generation.run_address_blocking_cap_stopword_experiment import (  # noqa: E402
    DF_TABLE_PATH,
    load_df_table,
    stopwords_for_threshold,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

DATA = os.path.join(_REPO_ROOT, "dataset", "train")
SOURCE1 = os.path.join(DATA, "train_source1.tsv")
SOURCE2 = os.path.join(DATA, "train_source2.tsv")
SOURCE3 = os.path.join(DATA, "train_source3.tsv")
GROUND_TRUTH = os.path.join(DATA, "train_ground_truth.tsv")
EXACT_NAME_OUTPUT = os.path.join(_REPO_ROOT, "candidate_generation", "output",
                                 "train_candidate_pairs_exact_name.tsv")
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exact_vs_address_union_20k.txt")

SAMPLE_SIZE = 20000
DF_THRESHOLD_PCT = 0.5
POSTING_CAP = 5000
MIN_SHARED_TOKENS = 2


def load_exact_name_candidates(path, wanted_ids):
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if row[0] in wanted_ids:
                out[row[0]] = frozenset(row[1].split(",")) if len(row) > 1 and row[1] else frozenset()
    return out


def build_pruned_address_index(path, needed_tokens, cap):
    """Same semantics as address_blocking.build_address_token_index (file
    order, first-seen ids, cap), restricted to `needed_tokens`."""
    index = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_addr = header.index("business_address")
        for row in reader:
            for tok in address_blocking_keys(normalize_business_address(row[idx_addr])):
                if tok not in needed_tokens:
                    continue
                lst = index.setdefault(tok, [])
                if len(lst) < cap:
                    lst.append(row[idx_id])
    return index


def address_candidates(keys, s2_index, s3_index):
    s2_counts = Counter()
    s3_counts = Counter()
    for tok in keys:
        s2_counts.update(s2_index.get(tok, ()))
        s3_counts.update(s3_index.get(tok, ()))
    return frozenset({c for c, ct in s2_counts.items() if ct >= MIN_SHARED_TOKENS}
                     | {c for c, ct in s3_counts.items() if ct >= MIN_SHARED_TOKENS})


def volume_stats(sizes):
    return {"pairs": sum(sizes), "mean": sum(sizes) / len(sizes),
            "median": statistics.median(sizes), "max": max(sizes),
            "zero_pct": sum(1 for s in sizes if s == 0) / len(sizes) * 100}


def main():
    t_start = time.time()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    sample_rows, total_s1, stride = sample_source1(SOURCE1, SAMPLE_SIZE)
    sample_ids = [eid for eid, _ in sample_rows]
    wanted = set(sample_ids)
    log(f"Sample: {len(sample_ids):,} of {total_s1:,} S1 entities (stride {stride}), "
        f"same sampler as previous bounded experiments.")

    stopwords = stopwords_for_threshold(load_df_table(DF_TABLE_PATH), DF_THRESHOLD_PCT)
    sample_keys = {eid: [t for t in address_blocking_keys(normalize_business_address(addr))
                         if t not in stopwords]
                   for eid, addr in sample_rows}
    del sample_rows
    needed = {t for keys in sample_keys.values() for t in keys}
    log(f"Stage 2 config: min_shared_tokens={MIN_SHARED_TOKENS}, cap={POSTING_CAP:,}, "
        f"stopwords={len(stopwords)} (current list + DF > {DF_THRESHOLD_PCT}%); "
        f"{len(needed):,} distinct query tokens.")

    truth = load_ground_truth_for_ids(GROUND_TRUTH, wanted)
    exact = load_exact_name_candidates(EXACT_NAME_OUTPUT, wanted)
    log(f"Loaded ground truth for {len(truth):,} and Stage 1 candidates for {len(exact):,} sampled S1 ids "
        f"(reused {os.path.basename(EXACT_NAME_OUTPUT)}).")
    missing = wanted - exact.keys()
    if missing:
        raise SystemExit(f"{len(missing)} sampled ids missing from Stage 1 output")

    t0 = time.time()
    s2_index = build_pruned_address_index(SOURCE2, needed, POSTING_CAP)
    s3_index = build_pruned_address_index(SOURCE3, needed, POSTING_CAP)
    log(f"Built pruned S2/S3 address indexes in {time.time() - t0:.1f}s.")

    sizes = {"exact": [], "address": [], "union": []}
    covered = {"exact": 0, "address": 0, "union": 0}
    entities_with_truth = total_true = 0
    both = only_exact = only_addr = neither = 0

    for eid in sample_ids:
        e = exact[eid]
        a = address_candidates(sample_keys[eid], s2_index, s3_index)
        u = e | a
        sizes["exact"].append(len(e))
        sizes["address"].append(len(a))
        sizes["union"].append(len(u))

        t = truth.get(eid)
        if not t:
            continue
        entities_with_truth += 1
        total_true += len(t)
        te, ta = t & e, t & a
        both += len(te & ta)
        only_exact += len(te - ta)
        only_addr += len(ta - te)
        neither += len(t - te - ta)
        covered["exact"] += bool(te)
        covered["address"] += bool(ta)
        covered["union"] += bool(te or ta)

    rec = {"exact": both + only_exact, "address": both + only_addr, "union": both + only_exact + only_addr}
    vol = {k: volume_stats(v) for k, v in sizes.items()}

    def pct(n, d):
        return n / d * 100

    log("")
    log("=" * 86)
    log(f"EXACT-NAME vs ADDRESS vs UNION  (sample {len(sample_ids):,} S1; "
        f"{entities_with_truth:,} with >=1 true match; {total_true:,} true pairs)")
    log("=" * 86)
    log(f"{'Metric':44s}{'Exact-name':>14s}{'Address':>14s}{'Union':>14s}")
    log("-" * 86)
    log("CANDIDATE VOLUME")
    log(f"{'  Candidate pairs':44s}" + "".join(f"{vol[k]['pairs']:>14,}" for k in vol))
    log(f"{'  Mean candidates/S1':44s}" + "".join(f"{vol[k]['mean']:>14.2f}" for k in vol))
    log(f"{'  Median candidates/S1':44s}" + "".join(f"{vol[k]['median']:>14.1f}" for k in vol))
    log(f"{'  Max candidates/S1':44s}" + "".join(f"{vol[k]['max']:>14,}" for k in vol))
    log(f"{'  Zero-candidate %':44s}" + "".join(f"{vol[k]['zero_pct']:>13.3f}%" for k in vol))
    log("PAIR-LEVEL RECALL (true S1-candidate pairs recovered / all true pairs)")
    log(f"{'  Recovered true pairs':44s}" + "".join(f"{rec[k]:>14,}" for k in rec))
    log(f"{'  Pair-level recall':44s}" + "".join(f"{pct(rec[k], total_true):>13.4f}%" for k in rec))
    log("ENTITY-LEVEL COVERAGE (S1 entities with >=1 true match recovered / entities with >=1 true match)")
    log(f"{'  Entities covered':44s}" + "".join(f"{covered[k]:>14,}" for k in covered))
    log(f"{'  Entity-level coverage':44s}" + "".join(f"{pct(covered[k], entities_with_truth):>13.4f}%" for k in covered))
    log("")
    log(f"TRUE-PAIR OVERLAP (of {total_true:,} true pairs)")
    log(f"  Recovered by BOTH:            {both:>10,}  ({pct(both, total_true):.4f}%)")
    log(f"  Recovered ONLY by exact-name: {only_exact:>10,}  ({pct(only_exact, total_true):.4f}%)")
    log(f"  Recovered ONLY by address:    {only_addr:>10,}  ({pct(only_addr, total_true):.4f}%)")
    log(f"  Recovered by NEITHER:         {neither:>10,}  ({pct(neither, total_true):.4f}%)")
    log("")
    log("INCREMENTAL CONTRIBUTION OF ADDRESS BLOCKING OVER EXACT-NAME")
    log(f"  Address-only recovered true pairs:            {only_addr:,}")
    log(f"  Pair recall:  union - exact = {pct(rec['union'], total_true):.4f}% - "
        f"{pct(rec['exact'], total_true):.4f}% = +{pct(rec['union'] - rec['exact'], total_true):.4f} pts")
    log(f"  Entity coverage: union - exact = {pct(covered['union'], entities_with_truth):.4f}% - "
        f"{pct(covered['exact'], entities_with_truth):.4f}% = "
        f"+{pct(covered['union'] - covered['exact'], entities_with_truth):.4f} pts "
        f"(+{covered['union'] - covered['exact']:,} entities)")
    log(f"  Extra candidate pairs added by address over exact: {vol['union']['pairs'] - vol['exact']['pairs']:,}")
    log(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

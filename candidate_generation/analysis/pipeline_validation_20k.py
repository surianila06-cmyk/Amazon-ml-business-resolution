#!/usr/bin/env python3
"""
20k validation of the PRODUCTION candidate pipeline (run_candidate_pipeline.py
--sample-size 20000 --channel-diagnostics), with DISTINCT-token address
semantics, against the previous bounded experiment (repeated-token
semantics; rare_name_df_threshold_comparison_20k.txt, config A: DF <= 200).

Checks and reports:
  - pilot_candidate_pairs.tsv is exactly the union of the per-channel id
    lists (i.e. the file handed to Role 3 is the final union), one row per
    sampled S1, same S1 sample as the experiments;
  - per-channel and union volume / recall / entity coverage / unique
    contributions (ground truth used for scoring only);
  - comparison with the previous experiment, and an explanation of the
    address-channel difference: previous top-200 lists (saved
    top200_address_candidates_20k.tsv) vs new ones, with DISTINCT shared
    qualifying tokens recomputed for every changed pair.

Output: candidate_generation/analysis/pipeline_validation_20k.txt

Usage:
    python candidate_generation/analysis/pipeline_validation_20k.py --cache-dir <scratchpad cache>
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.candidate_pipeline import (  # noqa: E402
    address_query_tokens, iter_source1_selection, iter_source_rows, load_address_stopwords,
)
from candidate_generation.run_address_blocking_bounded_experiment import load_ground_truth_for_ids  # noqa: E402
from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

DATA = os.path.join(_REPO_ROOT, "dataset", "train")
OUT_DIR = os.path.join(_REPO_ROOT, "candidate_generation", "output")
PILOT = os.path.join(OUT_DIR, "pilot_candidate_pairs.tsv")
CHANNELS = os.path.join(OUT_DIR, "pilot_candidate_pairs.channels.tsv")
SUMMARY = os.path.join(OUT_DIR, "pilot_candidate_pairs.summary.json")
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_validation_20k.txt")

# Previous bounded experiment (repeated-token address semantics), from
# exact_vs_address_union_20k.txt / address_filter_union_comparison_20k.txt /
# rare_name_df_threshold_comparison_20k.txt (config A, DF <= 200).
PREVIOUS = {
    "exact": {"pairs": 212_167, "tp": 15_684},
    "address": {"pairs": 2_478_702, "tp": 54_099},
    "rare": {"pairs": 478_551, "tp": 19_014},
    "union": {"pairs": 3_132_683, "tp": 60_437, "cov": 18_389},
}


def split_ids(s):
    return frozenset(s.split(",")) if s else frozenset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    # ---- load pilot outputs -----------------------------------------------------
    pilot = []
    with open(PILOT, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        assert next(r) == ["source1_entity_id", "candidate_entity_ids"]
        for row in r:
            pilot.append((row[0], row[1]))
    chans = {}
    with open(CHANNELS, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            chans[row[0]] = tuple(split_ids(x) for x in row[1:4])
    with open(SUMMARY) as f:
        summary = json.load(f)

    # ---- integrity: pilot == union of channels, same S1 sample ---------------------
    s1_sample = [row for row in iter_source1_selection(os.path.join(DATA, "train_source1.tsv"), sample_size=20000)]
    s1_ids = [r[0] for r in s1_sample]
    prev_top200 = {}
    with open(os.path.join(args.cache_dir, "top200_address_candidates_20k.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            e, ids = line.rstrip("\n").split("\t")
            prev_top200[e] = split_ids(ids)
    checks = {
        "pilot rows == sampled S1 ids, same order": [e for e, _ in pilot] == s1_ids,
        "sample identical to previous experiments": set(s1_ids) == set(prev_top200),
        "no duplicate S1 rows": len({e for e, _ in pilot}) == len(pilot),
        "each pilot row == sorted(exact | address | rare)": all(
            ids == ",".join(sorted(chans[e][0] | chans[e][1] | chans[e][2])) for e, ids in pilot),
        "candidate ids unique within each row": all(len(set(ids.split(","))) == len(ids.split(",")) for _, ids in pilot if ids),
        "only S2-/S3- ids": all(i.startswith(("S2-", "S3-")) for _, ids in pilot for i in split_ids(ids)),
    }

    truth = load_ground_truth_for_ids(os.path.join(DATA, "train_ground_truth.tsv"), set(s1_ids))
    total_true = sum(len(t) for t in truth.values() if t)
    ent_truth = sum(1 for t in truth.values() if t)

    def metrics(get):
        sizes, tp, cov = [], 0, 0
        for e in s1_ids:
            s = get(e)
            sizes.append(len(s))
            t = truth.get(e)
            if t:
                h = len(s & t)
                tp += h
                cov += h > 0
        pairs = sum(sizes)
        return {"pairs": pairs, "mean": pairs / len(sizes), "median": statistics.median(sizes), "max": max(sizes),
                "zero": sum(1 for x in sizes if x == 0) / len(sizes) * 100, "tp": tp,
                "recall": tp / total_true * 100, "cov": cov, "cov_pct": cov / ent_truth * 100}

    m = {
        "exact": metrics(lambda e: chans[e][0]),
        "address": metrics(lambda e: chans[e][1]),
        "rare": metrics(lambda e: chans[e][2]),
        "union": metrics(lambda e: chans[e][0] | chans[e][1] | chans[e][2]),
    }
    pilot_map = dict(pilot)
    m["pilot_file"] = metrics(lambda e: split_ids(pilot_map[e]))

    only = {"exact": [0, 0], "address": [0, 0], "rare": [0, 0]}  # [pairs, true pairs]
    for e in s1_ids:
        ex, ad, ra = chans[e]
        t = truth.get(e) or frozenset()
        for k, own, others in (("exact", ex, ad | ra), ("address", ad, ex | ra), ("rare", ra, ex | ad)):
            d = own - others
            only[k][0] += len(d)
            only[k][1] += len(d & t)

    # ---- address-channel difference vs previous top-200 --------------------------------
    removed, added = {}, {}
    for e in s1_ids:
        new, old = chans[e][1], prev_top200[e]
        if old - new:
            removed[e] = old - new
        if new - old:
            added[e] = new - old
    need = set().union(*removed.values(), *added.values()) if (removed or added) else set()
    stopwords = load_address_stopwords()
    tgt_tokens = {}
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        for eid, _, addr, _ in iter_source_rows(os.path.join(DATA, fname)):
            if eid in need:
                tgt_tokens[eid] = address_query_tokens(normalize_business_address(addr), stopwords)
    s1_tokens = {r[0]: address_query_tokens(normalize_business_address(r[2]), stopwords) for r in s1_sample}
    prev_status = {}
    with open(os.path.join(args.cache_dir, "true_pair_status.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            e, tid, _, in_ret, _, _ = line.rstrip("\n").split("\t")
            prev_status[(e, tid)] = in_ret == "1"

    def shared(e, c):
        return len(s1_tokens[e] & tgt_tokens[c])

    rem_shared = Counter(min(shared(e, c), 3) for e, cs in removed.items() for c in cs)
    add_shared = Counter(min(shared(e, c), 3) for e, cs in added.items() for c in cs)
    rem_true = sum(len(cs & (truth.get(e) or frozenset())) for e, cs in removed.items())
    add_true = sum(len(cs & (truth.get(e) or frozenset())) for e, cs in added.items())
    add_true_prev_pool = sum(1 for e, cs in added.items() for c in cs & (truth.get(e) or frozenset())
                             if prev_status.get((e, c)))
    rem_true_shared = Counter(min(shared(e, c), 3) for e, cs in removed.items()
                              for c in cs & (truth.get(e) or frozenset()))
    s1_affected = len(set(removed) | set(added))

    # ---- report ------------------------------------------------------------------------
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    W = "=" * 104
    out(W)
    out("PRODUCTION PIPELINE -- 20k VALIDATION (distinct-token address semantics)")
    out(W)
    out(f"Run: run_candidate_pipeline.py --sample-size 20000 (stride-110 sample, same as all experiments); "
        f"config {summary['config']}; stopwords {summary['stopwords']}.")
    out(f"Ground truth: {ent_truth:,} S1 with >=1 true match, {total_true:,} true pairs (scoring only).")
    out("\nINTEGRITY CHECKS")
    for k, v in checks.items():
        out(f"  [{'PASS' if v else 'FAIL'}] {k}")
    out(f"  [{'PASS' if m['pilot_file'] == m['union'] else 'FAIL'}] metrics of pilot file == metrics of channel union")

    out("\nRUNTIME / MEMORY (from pilot_candidate_pairs.summary.json)")
    rt = summary["runtime_s"]
    out(f"  S1 vocabulary pass {rt['s1_vocabulary']:.0f}s; S2/S3 index pass {rt['target_index']:.0f}s; "
        f"candidate generation {rt['candidate_generation']:.0f}s; total {rt['total']:.0f}s")
    out(f"  peak memory (peak working set): {summary['peak_memory_mb']:,.0f} MB; pilot file "
        f"{summary['out_bytes']:,} bytes")

    rows = [("candidate pairs", "pairs", "{:,}"), ("mean candidates/S1", "mean", "{:.2f}"),
            ("median candidates/S1", "median", "{:.1f}"), ("max candidates/S1", "max", "{:,}"),
            ("zero-candidate %", "zero", "{:.3f}%"), ("true pairs recovered", "tp", "{:,}"),
            ("pair-level recall", "recall", "{:.4f}%"), ("entities covered", "cov", "{:,}"),
            ("entity-level coverage", "cov_pct", "{:.4f}%")]
    out("\nCHANNELS AND FINAL UNION (new)")
    cols = ("exact", "address", "rare", "union")
    out(f"  {'metric':30s}" + "".join(f"{c:>16s}" for c in ("exact-name", "address top-200", "rare-name", "UNION")))
    for name, key, fmt in rows:
        out(f"  {name:30s}" + "".join(f"{fmt.format(m[c][key]):>16s}" for c in cols))
    out(f"  {'pairs only from this channel':30s}" + "".join(f"{only[c][0]:>16,}" for c in cols[:3]))
    out(f"  {'true pairs only from channel':30s}" + "".join(f"{only[c][1]:>16,}" for c in cols[:3]))
    out(f"  channel overlap (pairs counted in >1 channel): "
        f"{m['exact']['pairs'] + m['address']['pairs'] + m['rare']['pairs'] - m['union']['pairs']:,}")

    out("\nCOMPARISON WITH PREVIOUS 20k EXPERIMENT (repeated-token semantics, rare DF<=200)")
    out(f"  {'':12s}{'prev pairs':>14s}{'new pairs':>14s}{'change':>12s}{'prev true':>11s}{'new true':>10s}"
        f"{'change':>9s}{'prev recall':>13s}{'new recall':>12s}")
    for c in cols:
        p, n = PREVIOUS[c], m[c]
        out(f"  {c:12s}{p['pairs']:>14,}{n['pairs']:>14,}{n['pairs'] - p['pairs']:>+12,}{p['tp']:>11,}{n['tp']:>10,}"
            f"{n['tp'] - p['tp']:>+9,}{p['tp'] / total_true * 100:>12.4f}%{n['recall']:>11.4f}%")
    out(f"  union entity coverage: prev {PREVIOUS['union']['cov'] / ent_truth * 100:.4f}% -> new {m['union']['cov_pct']:.4f}%")

    out("\nWHY THE ADDRESS CHANNEL CHANGED (previous top-200 vs new top-200, per S1)")
    out(f"  S1 whose address list changed: {s1_affected:,} of {len(s1_ids):,}")
    out(f"  pairs REMOVED (in previous top-200, not in new): {sum(map(len, removed.values())):,} "
        f"(true pairs among them: {rem_true:,})")
    out("    distinct shared qualifying tokens of removed pairs: "
        + ", ".join(f"{k if k < 3 else '3+'}: {rem_shared[k]:,}" for k in range(4)))
    out("    distinct shared tokens of REMOVED TRUE pairs: "
        + ", ".join(f"{k if k < 3 else '3+'}: {rem_true_shared[k]:,}" for k in range(4)))
    out(f"  pairs ADDED (in new top-200, not in previous): {sum(map(len, added.values())):,} "
        f"(true pairs among them: {add_true:,}; of those, {add_true_prev_pool:,} were already retrieved "
        f"before top-200 in the previous run)")
    out("    distinct shared qualifying tokens of added pairs: "
        + ", ".join(f"{k if k < 3 else '3+'}: {add_shared[k]:,}" for k in range(4)))
    out("  Reading: removed pairs with <2 distinct shared tokens were admitted before only because a token")
    out("  repeated in an address was counted twice; they are now excluded by design. Their freed top-200 slots")
    out("  are refilled from the remaining (>=2 distinct) pool, and because each target is now posted at most")
    out("  once per token, the 5,000 posting cap reaches more distinct rows. Removed pairs with >=2 distinct")
    out("  shared tokens left through those rank/cap shifts or the new entity_id tie-break.")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

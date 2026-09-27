#!/usr/bin/env python3
"""
Final bounded comparison of address post-filters, alone and UNIONED with
Stage 1 exact-name candidates, on the same 20,000 stride-110 S1 sample.

Address retrieval: min_shared_tokens=2, cap=5,000, stopwords = current list
+ DF > 0.5% (161). Filters (from address_postfilter_eval_20k.py):
  A) token Jaccard >= 0.3
  B) token Jaccard >= 0.3 OR shared numeric >= 1
  C) top-200 per S1 by token Jaccard (tie: shared tokens, then retrieval order)

Reuses the per-pair feature cache written by address_postfilter_features_20k.py
(jaccard / shared / shared_num / is_true). That cache has no candidate ids,
so candidate ids are regenerated with the identical retrieval procedure (one
pruned-index pass over S2/S3) and aligned to the cache by position; the
regenerated per-pair retrieval counts and per-S1 offsets are checked for
exact equality with the cache before use. No features are recomputed and no
character similarity is computed.

Stage 1 reuses candidate_generation/output/train_candidate_pairs_exact_name.tsv.
Analysis only: no candidate TSV written, no blocker configuration changed.

Output: candidate_generation/analysis/address_filter_union_comparison_20k.txt

Usage:
    python candidate_generation/analysis/address_filter_union_comparison_20k.py --cache-dir <dir>
"""
import argparse
import json
import os
import statistics
import sys
import time
from array import array
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import address_blocking_keys  # noqa: E402
from candidate_generation.analysis.address_postfilter_eval_20k import load  # noqa: E402
from candidate_generation.analysis.exact_vs_address_union_20k import (  # noqa: E402
    DF_THRESHOLD_PCT, EXACT_NAME_OUTPUT, GROUND_TRUTH, MIN_SHARED_TOKENS, POSTING_CAP,
    SAMPLE_SIZE, SOURCE1, SOURCE2, SOURCE3, build_pruned_address_index,
    load_exact_name_candidates,
)
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    load_ground_truth_for_ids, sample_source1,
)
from candidate_generation.run_address_blocking_cap_stopword_experiment import (  # noqa: E402
    DF_TABLE_PATH, load_df_table, stopwords_for_threshold,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "address_filter_union_comparison_20k.txt")
TOP_K = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    t_all = time.time()

    with open(os.path.join(args.cache_dir, "meta.json")) as f:
        meta = json.load(f)
    c_offsets = load(args.cache_dir, "offsets", "q")
    c_retr = load(args.cache_dir, "retr", "B")
    shared = load(args.cache_dir, "shared", "B")
    shared_num = load(args.cache_dir, "shared_num", "B")
    jacc = load(args.cache_dir, "jacc", "f")

    # ---- regenerate candidate ids in the same order as the feature cache ----
    stopwords = stopwords_for_threshold(load_df_table(DF_TABLE_PATH), DF_THRESHOLD_PCT)
    sample_rows, total_s1, stride = sample_source1(SOURCE1, SAMPLE_SIZE)
    s1_ids = [eid for eid, _ in sample_rows]
    s1_keys = [[t for t in address_blocking_keys(normalize_business_address(addr)) if t not in stopwords]
               for _, addr in sample_rows]
    del sample_rows
    needed = {t for keys in s1_keys for t in keys}

    t0 = time.time()
    s2_index = build_pruned_address_index(SOURCE2, needed, POSTING_CAP)
    s3_index = build_pruned_address_index(SOURCE3, needed, POSTING_CAP)
    t_index = time.time() - t0

    t0 = time.time()
    offsets = array("q", [0])
    pair_cid = []
    retr = array("B")
    for keys in s1_keys:
        for index in (s2_index, s3_index):
            counts = Counter()
            for tok in keys:
                counts.update(index.get(tok, ()))
            for cid, ct in counts.items():
                if ct >= MIN_SHARED_TOKENS:
                    pair_cid.append(cid)
                    retr.append(min(ct, 255))
        offsets.append(len(pair_cid))
    del s2_index, s3_index, s1_keys
    t_gen = time.time() - t0
    if offsets != c_offsets or retr != c_retr:
        raise SystemExit("regenerated candidates do not align with the feature cache -- aborting")
    del c_retr, c_offsets
    print(f"regenerated {len(pair_cid):,} candidate ids (index {t_index:.0f}s, gen {t_gen:.0f}s); "
          f"offsets and retrieval counts match the cache exactly")

    wanted = set(s1_ids)
    truth = load_ground_truth_for_ids(GROUND_TRUTH, wanted)
    exact = load_exact_name_candidates(EXACT_NAME_OUTPUT, wanted)
    total_true = sum(len(t) for t in truth.values() if t)
    ent_truth = sum(1 for t in truth.values() if t)
    assert total_true == meta["total_true_pairs"] and ent_truth == meta["entities_with_truth"]

    # ---- address filters: per-S1 kept pair positions --------------------------
    def keep_threshold(pred):
        def f(a, b):
            return [p for p in range(a, b) if pred(p)]
        return f

    def keep_topk(a, b):
        if b - a <= TOP_K:
            return range(a, b)
        # identical ordering to address_postfilter_eval_20k.py (stable sort)
        return sorted(range(a, b), key=lambda p: (-jacc[p], -shared[p]))[:TOP_K]

    filters = [
        ("A: Jaccard>=0.3", keep_threshold(lambda p: jacc[p] >= 0.3)),
        ("B: Jaccard>=0.3 OR numeric>=1", keep_threshold(lambda p: jacc[p] >= 0.3 or shared_num[p] >= 1)),
        (f"C: top-{TOP_K}/S1 by Jaccard", keep_topk),
    ]

    def new_acc():
        return {"sizes": [], "tp": 0, "cov": 0}

    def add(acc, cands, t):
        acc["sizes"].append(len(cands))
        if t:
            hit = cands & t
            acc["tp"] += len(hit)
            acc["cov"] += bool(hit)

    def finish(acc):
        sz = acc["sizes"]
        pairs = sum(sz)
        return {"pairs": pairs, "mean": pairs / len(sz), "median": statistics.median(sz), "max": max(sz),
                "zero_pct": sum(1 for x in sz if x == 0) / len(sz) * 100,
                "tp": acc["tp"], "recall": acc["tp"] / total_true * 100,
                "cov": acc["cov"], "cov_pct": acc["cov"] / ent_truth * 100, "false": pairs - acc["tp"]}

    exact_acc = new_acc()
    for eid in s1_ids:
        add(exact_acc, exact[eid], truth.get(eid))
    exact_res = finish(exact_acc)

    results = []
    for label, keep in filters:
        t0 = time.time()
        addr_acc, union_acc = new_acc(), new_acc()
        both = only_exact = only_addr = neither = 0
        for i, eid in enumerate(s1_ids):
            a_set = frozenset(pair_cid[p] for p in keep(offsets[i], offsets[i + 1]))
            e_set = exact[eid]
            u_set = e_set | a_set
            t = truth.get(eid)
            add(addr_acc, a_set, t)
            add(union_acc, u_set, t)
            if t:
                te, ta = t & e_set, t & a_set
                both += len(te & ta)
                only_exact += len(te - ta)
                only_addr += len(ta - te)
                neither += len(t - te - ta)
        results.append({"label": label, "addr": finish(addr_acc), "union": finish(union_acc),
                        "both": both, "only_exact": only_exact, "only_addr": only_addr,
                        "neither": neither, "elapsed": time.time() - t0})
        print(f"  {label}: union {results[-1]['union']['pairs']:,} pairs, "
              f"recall {results[-1]['union']['recall']:.2f}%")

    # ---- report ---------------------------------------------------------------
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    labels = [r["label"] for r in results]
    W = 30

    def row(name, vals):
        out(f"{name:46s}" + "".join(f"{v:>{W}s}" for v in vals))

    out("=" * (46 + W * 3))
    out("EXACT-NAME UNION FILTERED-ADDRESS -- bounded comparison (analysis only)")
    out("=" * (46 + W * 3))
    out(f"Sample: {len(s1_ids):,} of {total_s1:,} S1 (stride {stride}); {ent_truth:,} S1 with >=1 true match; "
        f"{total_true:,} true pairs.")
    out(f"Stage 1: {os.path.basename(EXACT_NAME_OUTPUT)} (reused). Stage 2 retrieval: min_shared_tokens=2, "
        f"cap=5,000, stopwords={len(stopwords)} (current + DF > 0.5%); unfiltered {len(pair_cid):,} pairs.")
    out("Pair-level recall = true (S1, candidate) pairs recovered / all true pairs.")
    out("Entity-level coverage = S1 with >=1 true match recovered / S1 with >=1 true match.")
    out(f"Candidate ids regenerated in {t_index + t_gen:.0f}s and verified position-by-position against the "
        f"feature cache; no address features or char similarity recomputed.")
    out("")
    out("UNION CONFIGURATIONS (exact-name UNION filtered address)")
    row("Metric", [f"Exact U {l.split(':')[0]}" for l in labels])
    out("-" * (46 + W * 3))
    row("  Exact-name candidate pairs", [f"{exact_res['pairs']:,}"] * 3)
    row("  Address candidate pairs (filtered)", [f"{r['addr']['pairs']:,}" for r in results])
    row("  Union candidate pairs", [f"{r['union']['pairs']:,}" for r in results])
    row("  Overlap (in both candidate sets)",
        [f"{exact_res['pairs'] + r['addr']['pairs'] - r['union']['pairs']:,}" for r in results])
    row("  Mean candidates/S1", [f"{r['union']['mean']:.2f}" for r in results])
    row("  Median candidates/S1", [f"{r['union']['median']:.1f}" for r in results])
    row("  Max candidates/S1", [f"{r['union']['max']:,}" for r in results])
    row("  Zero-candidate %", [f"{r['union']['zero_pct']:.3f}%" for r in results])
    row("  True pairs recovered", [f"{r['union']['tp']:,}" for r in results])
    row("  Pair-level recall", [f"{r['union']['recall']:.4f}%" for r in results])
    row("  Entities covered", [f"{r['union']['cov']:,}" for r in results])
    row("  Entity-level coverage", [f"{r['union']['cov_pct']:.4f}%" for r in results])
    row("  False candidate pairs", [f"{r['union']['false']:,}" for r in results])
    row("  True pairs: recovered by BOTH", [f"{r['both']:,}" for r in results])
    row("  True pairs: ONLY exact-name (incremental)", [f"{r['only_exact']:,}" for r in results])
    row("  True pairs: ONLY address (incremental)", [f"{r['only_addr']:,}" for r in results])
    row("  True pairs: missed by BOTH", [f"{r['neither']:,}" for r in results])
    row("  Recall gain of union over exact-name",
        [f"+{r['union']['recall'] - exact_res['recall']:.4f} pts" for r in results])
    row("  Recall gain of union over address alone",
        [f"+{r['union']['recall'] - r['addr']['recall']:.4f} pts" for r in results])
    out("")
    out("ADDRESS-ONLY METRICS (filtered address candidates, no exact-name)")
    row("Metric", labels)
    out("-" * (46 + W * 3))
    for name, key, fmt in [("  Candidate pairs", "pairs", "{:,}"), ("  Mean candidates/S1", "mean", "{:.2f}"),
                           ("  Median candidates/S1", "median", "{:.1f}"), ("  Max candidates/S1", "max", "{:,}"),
                           ("  Zero-candidate %", "zero_pct", "{:.3f}%"), ("  True pairs recovered", "tp", "{:,}"),
                           ("  Pair-level recall", "recall", "{:.4f}%"), ("  Entities covered", "cov", "{:,}"),
                           ("  Entity-level coverage", "cov_pct", "{:.4f}%"), ("  False candidate pairs", "false", "{:,}")]:
        row(name, [fmt.format(r["addr"][key]) for r in results])
    out("")
    out(f"EXACT-NAME ONLY (reference): pairs {exact_res['pairs']:,}; mean {exact_res['mean']:.2f}; median "
        f"{exact_res['median']:.1f}; max {exact_res['max']:,}; zero {exact_res['zero_pct']:.3f}%; true "
        f"{exact_res['tp']:,}; pair recall {exact_res['recall']:.4f}%; entity coverage {exact_res['cov_pct']:.4f}%; "
        f"false {exact_res['false']:,}")
    out(f"Unfiltered reference (previous run): exact U address = 20,143,346 pairs, 57,979 true, 83.7387% pair "
        f"recall, 96.6372% entity coverage.")
    out("")
    out("Runtime: " + "; ".join(f"{r['label'].split(':')[0]} {r['elapsed']:.1f}s" for r in results)
        + f" (union evaluation); total script {time.time() - t_all:.0f}s")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

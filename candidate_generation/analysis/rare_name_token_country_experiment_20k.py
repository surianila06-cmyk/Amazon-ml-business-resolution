#!/usr/bin/env python3
"""
Bounded experiment: RARE NAME TOKEN as a third retrieval channel, with and
without a same-country restriction, against the baseline
    exact-name UNION top-200 address-by-Jaccard
on the same 20,000 stride-110 S1 sample. Experiment only -- no blocker code
is changed, no candidate TSV is written.

Rare-name-token channel (as specified):
  - name tokens = normalize_business_name(...) split on whitespace, len >= 2
    (same definition as missed_by_both_analysis_20k.py)
  - DF = combined S2+S3 document frequency (rows whose name contains the token)
  - per S1: the single rarest token with 1 <= DF <= 1,000 (tie: token string);
    candidates = every S2/S3 row whose name contains that token
  - A) no country restriction   B) candidate country == S1 country
    (country compared after strip + lowercase; the token is still selected
    by the global DF)

Why S2/S3 are streamed once: the saved true_pair_status.tsv covers true pairs
only, but union volume and overlap need the baseline's NON-true address
candidate ids, which were never saved. One pass therefore (1) rebuilds the
pruned address index -- candidates are regenerated and verified
position-by-position against the feature cache, then top-200 is recomputed
from the cached Jaccard/shared arrays and saved to
<cache-dir>/top200_address_candidates_20k.tsv for reuse -- and (2) counts DF
and collects postings (id, country) ONLY for tokens that occur in the
sampled S1 names; postings are kept up to DF 1,000 (longer lists are never
eligible and are dropped).

Output: candidate_generation/analysis/rare_name_token_country_experiment_20k.txt

Usage:
    python candidate_generation/analysis/rare_name_token_country_experiment_20k.py --cache-dir <dir>
"""
import argparse
import csv
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
    SAMPLE_SIZE, SOURCE1, SOURCE2, SOURCE3, load_exact_name_candidates,
)
from candidate_generation.analysis.missed_by_both_analysis_20k import name_tokens  # noqa: E402
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    count_rows, load_ground_truth_for_ids,
)
from candidate_generation.run_address_blocking_cap_stopword_experiment import (  # noqa: E402
    DF_TABLE_PATH, load_df_table, stopwords_for_threshold,
)
from utils.preprocessing import normalize_business_address, normalize_business_name  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rare_name_token_country_experiment_20k.txt")
TOP_K = 200
MAX_NAME_DF = 1000


def country_key(c):
    return (c or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    t_all = time.time()

    stopwords = stopwords_for_threshold(load_df_table(DF_TABLE_PATH), DF_THRESHOLD_PCT)

    # ---- S1 sample, all fields (same stride sampler as sample_source1) ------
    total_s1 = count_rows(SOURCE1)
    stride = max(1, total_s1 // SAMPLE_SIZE)
    s1 = []
    with open(SOURCE1, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(reader)
        ii, iname, iaddr, ictry = (h.index(x) for x in ("entity_id", "business_name", "business_address", "country"))
        for i, row in enumerate(reader):
            if i % stride == 0:
                s1.append((row[ii], row[iname], row[iaddr], row[ictry]))
                if len(s1) >= SAMPLE_SIZE:
                    break
    n_s1 = len(s1)
    s1_ids = [r[0] for r in s1]
    s1_country = [country_key(r[3]) for r in s1]
    s1_keys = [[t for t in address_blocking_keys(normalize_business_address(r[2])) if t not in stopwords]
               for r in s1]
    s1_ntoks = [name_tokens(normalize_business_name(r[1])) for r in s1]
    addr_needed = {t for keys in s1_keys for t in keys}
    name_needed = set().union(*s1_ntoks)

    wanted = set(s1_ids)
    truth = load_ground_truth_for_ids(GROUND_TRUTH, wanted)
    exact = load_exact_name_candidates(EXACT_NAME_OUTPUT, wanted)
    target_ids = set().union(*[t for t in truth.values() if t])
    total_true = sum(len(t) for t in truth.values() if t)
    ent_truth = sum(1 for t in truth.values() if t)

    # ---- one pass over S2/S3 ------------------------------------------------
    t0 = time.time()
    addr_index = {"S2": {}, "S3": {}}
    name_df = Counter()
    name_post = {}  # token -> list of (id, country); dropped once DF > MAX_NAME_DF
    target_country = {}
    for src, path in (("S2", SOURCE2), ("S3", SOURCE3)):
        index = addr_index[src]
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            h = next(reader)
            ii, iname, iaddr, ictry = (h.index(x) for x in ("entity_id", "business_name", "business_address", "country"))
            for row in reader:
                cid = row[ii]
                for tok in address_blocking_keys(normalize_business_address(row[iaddr])):
                    if tok in addr_needed:
                        lst = index.setdefault(tok, [])
                        if len(lst) < POSTING_CAP:
                            lst.append(cid)
                ctry = country_key(row[ictry])
                for tok in name_tokens(normalize_business_name(row[iname])) & name_needed:
                    name_df[tok] += 1
                    df = name_df[tok]
                    if df <= MAX_NAME_DF:
                        name_post.setdefault(tok, []).append((cid, ctry))
                    elif df == MAX_NAME_DF + 1:
                        name_post.pop(tok, None)
                if cid in target_ids:
                    target_country[cid] = (row[ictry], ctry)
        print(f"  streamed {src}")
    t_pass = time.time() - t0

    # ---- country agreement of true pairs (verified, not assumed) -------------
    ctry_same = ctry_diff = ctry_missing = raw_same = 0
    diff_examples = Counter()
    for i, eid in enumerate(s1_ids):
        for tid in truth.get(eid) or ():
            raw_t, kt = target_country[tid]
            if not s1_country[i] or not kt:
                ctry_missing += 1
            elif s1_country[i] == kt:
                ctry_same += 1
            else:
                ctry_diff += 1
                diff_examples[(s1[i][3], raw_t)] += 1
            raw_same += s1[i][3] == raw_t
    s1_country_dist = Counter(s1_country)

    # ---- baseline: regenerate address candidates, verify, recompute top-200 --
    c_offsets = load(args.cache_dir, "offsets", "q")
    c_retr = load(args.cache_dir, "retr", "B")
    jacc = load(args.cache_dir, "jacc", "f")
    shared = load(args.cache_dir, "shared", "B")
    ok = True
    top200 = []
    p = 0
    for i, keys in enumerate(s1_keys):
        ids = []
        for src in ("S2", "S3"):
            counts = Counter()
            for tok in keys:
                counts.update(addr_index[src].get(tok, ()))
            for cid, ct in counts.items():
                if ct >= MIN_SHARED_TOKENS:
                    if p >= len(c_retr) or c_retr[p] != min(ct, 255):
                        ok = False
                    ids.append(cid)
                    p += 1
        if c_offsets[i + 1] != p:
            ok = False
        a = c_offsets[i]
        if len(ids) > TOP_K:
            order = sorted(range(a, a + len(ids)), key=lambda q: (-jacc[q], -shared[q]))[:TOP_K]
            ids = [ids[q - a] for q in order]
        top200.append(frozenset(ids))
    del addr_index, jacc, shared
    if not ok or p != len(c_retr):
        raise SystemExit("regenerated address candidates do not align with the feature cache -- aborting")
    with open(os.path.join(args.cache_dir, "top200_address_candidates_20k.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, ids in zip(s1_ids, top200):
            f.write(f"{eid}\t{','.join(sorted(ids))}\n")
    print(f"baseline address candidates regenerated and verified; top-200 saved ({sum(map(len, top200)):,} pairs)")

    baseline = [exact[eid] | top200[i] for i, eid in enumerate(s1_ids)]

    # ---- rare-token selection -------------------------------------------------
    selected = []  # (token, df) or None
    for toks in s1_ntoks:
        elig = sorted((name_df[t], t) for t in toks if 1 <= name_df.get(t, 0) <= MAX_NAME_DF)
        selected.append((elig[0][1], elig[0][0]) if elig else None)

    # ---- metrics ------------------------------------------------------------
    def metrics(sets):
        sizes = [len(s) for s in sets]
        tp = cov = 0
        for eid, s in zip(s1_ids, sets):
            t = truth.get(eid)
            if t:
                h_ = len(s & t)
                tp += h_
                cov += h_ > 0
        pairs = sum(sizes)
        return {"pairs": pairs, "mean": pairs / n_s1, "median": statistics.median(sizes), "max": max(sizes),
                "zero": sum(1 for x in sizes if x == 0) / n_s1 * 100, "tp": tp,
                "recall": tp / total_true * 100, "cov": cov, "cov_pct": cov / ent_truth * 100,
                "false": pairs - tp}

    base_m = metrics(baseline)
    configs = []
    for label, same_country in (("A: 1 rare token, DF<=1000", False),
                                ("B: 1 rare token, DF<=1000, same country", True)):
        t0 = time.time()
        rare = []
        for i, sel in enumerate(selected):
            if sel is None:
                rare.append(frozenset())
                continue
            post = name_post[sel[0]]
            if same_country:
                rare.append(frozenset(c for c, k in post if k == s1_country[i]))
            else:
                rare.append(frozenset(c for c, _ in post))
        union = [b | r for b, r in zip(baseline, rare)]
        contributed = sum(len(r - b) for r, b in zip(rare, baseline))
        overlap = sum(len(r & b) for r, b in zip(rare, baseline))
        inc_true = 0
        for eid, r, b in zip(s1_ids, rare, baseline):
            t = truth.get(eid)
            if t:
                inc_true += len((r - b) & t)
        rare_m = metrics(rare)
        union_m = metrics(union)
        configs.append({"label": label, "rare": rare_m, "union": union_m, "contributed": contributed,
                        "overlap": overlap, "inc_true": inc_true, "elapsed": time.time() - t0,
                        "rare_sizes": [len(r) for r in rare]})
        print(f"  {label}: rare {rare_m['pairs']:,}, union recall {union_m['recall']:.2f}%")

    # ---- report ---------------------------------------------------------------
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    scale = total_s1 / n_s1
    W = "=" * 112
    out(W)
    out("RARE NAME TOKEN THIRD CHANNEL, with / without SAME-COUNTRY restriction (bounded 20k sample; experiment only)")
    out(W)
    out(f"Sample: {n_s1:,} of {total_s1:,} S1 (stride {stride}); {ent_truth:,} S1 with >=1 true match; "
        f"{total_true:,} true pairs.")
    out("Baseline: exact-name (reused train_candidate_pairs_exact_name.tsv) UNION top-200 address-by-Jaccard "
        "(retrieval min_shared=2, cap 5,000, 161 stopwords); address candidates regenerated and verified against "
        "the feature cache.")
    out(f"Rare-name channel: name tokens = normalize_business_name split on spaces, len >= 2; DF = combined S2+S3 "
        f"rows containing the token; per S1 the single rarest token with 1 <= DF <= {MAX_NAME_DF:,} (tie: token "
        f"string); B keeps only candidates whose country equals the S1 country (strip + lowercase).")
    out(f"Timing: one S2/S3 pass {t_pass:.0f}s; total {time.time() - t_all:.0f}s.")

    out("\n1. COUNTRY AGREEMENT OF TRUE PAIRS (verified on all sampled true pairs)")
    out(f"    same (normalized):   {ctry_same:,} ({ctry_same / total_true * 100:.4f}%)")
    out(f"    different:           {ctry_diff:,} ({ctry_diff / total_true * 100:.4f}%)")
    out(f"    missing either side: {ctry_missing:,} ({ctry_missing / total_true * 100:.4f}%)")
    out(f"    identical raw string: {raw_same:,} ({raw_same / total_true * 100:.4f}%)")
    if diff_examples:
        out("    differing (S1, target) country values: "
            + ", ".join(f"{a!r}/{b!r} x{n}" for (a, b), n in diff_examples.most_common(10)))
    out("    S1 country values in sample: " + ", ".join(f"{k or '<empty>'} {v:,}" for k, v in s1_country_dist.most_common(10)))

    out("\n2. RARE TOKEN ACTUALLY USED PER S1")
    n_sel = sum(1 for s in selected if s)
    out(f"    S1 with an eligible token (DF 1..{MAX_NAME_DF:,}): {n_sel:,} ({n_sel / n_s1 * 100:.2f}%); "
        f"without: {n_s1 - n_sel:,} ({(n_s1 - n_sel) / n_s1 * 100:.2f}%)")
    buckets = [(1, 1), (2, 5), (6, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
    dfs = [s[1] for s in selected if s]
    out(f"    {'DF of used token':>18s} {'S1 count':>9s} {'% of S1 w/ token':>17s}")
    for lo, hi in buckets:
        c = sum(1 for d in dfs if lo <= d <= hi)
        out(f"    {f'{lo}-{hi}' if lo != hi else str(lo):>18s} {c:>9,} {c / n_sel * 100:>16.2f}%")
    out(f"    median DF of used token: {statistics.median(dfs):.0f}; mean {statistics.mean(dfs):.1f}")
    out("    note: DF=1 tokens occur in exactly one S2/S3 row; names with no eligible token fall back to "
        "the baseline only.")

    out("\n3. RARE-NAME CHANNEL ALONE")
    hdr = ["A (no country)", "B (same country)"]
    out(f"    {'metric':40s}" + "".join(f"{h:>22s}" for h in hdr))
    for name, key, fmt in [("candidate pairs", "pairs", "{:,}"), ("mean candidates/S1", "mean", "{:.2f}"),
                           ("median candidates/S1", "median", "{:.1f}"), ("max candidates/S1", "max", "{:,}"),
                           ("zero-candidate %", "zero", "{:.2f}%"), ("true pairs recovered", "tp", "{:,}"),
                           ("pair-level recall", "recall", "{:.4f}%"), ("entities covered", "cov", "{:,}"),
                           ("entity-level coverage", "cov_pct", "{:.4f}%"), ("false candidates", "false", "{:,}")]:
        out(f"    {name:40s}" + "".join(f"{fmt.format(c['rare'][key]):>22s}" for c in configs))

    out("\n4. UNION WITH BASELINE (exact-name U top-200 address U rare-name)")
    hdr = ["Baseline", "Baseline U A", "Baseline U B"]
    out(f"    {'metric':40s}" + "".join(f"{h:>22s}" for h in hdr))
    cols = [base_m] + [c["union"] for c in configs]
    for name, key, fmt in [("total candidate pairs", "pairs", "{:,}"), ("mean candidates/S1", "mean", "{:.2f}"),
                           ("median candidates/S1", "median", "{:.1f}"), ("max candidates/S1", "max", "{:,}"),
                           ("zero-candidate %", "zero", "{:.3f}%"), ("true pairs recovered", "tp", "{:,}"),
                           ("pair-level recall", "recall", "{:.4f}%"), ("entities covered", "cov", "{:,}"),
                           ("entity-level coverage", "cov_pct", "{:.4f}%"), ("false candidates", "false", "{:,}")]:
        out(f"    {name:40s}" + "".join(f"{fmt.format(m[key]):>22s}" for m in cols))
    out(f"    {'incremental true pairs from rare-name':40s}{'-':>22s}" + "".join(f"{c['inc_true']:>22,}" for c in configs))
    out(f"    {'recall gain (pts)':40s}{'-':>22s}"
        + "".join(f"{'+%.4f' % (c['union']['recall'] - base_m['recall']):>22s}" for c in configs))
    out(f"    {'entity coverage gain (pts)':40s}{'-':>22s}"
        + "".join(f"{'+%.4f' % (c['union']['cov_pct'] - base_m['cov_pct']):>22s}" for c in configs))
    out(f"    {'candidates contributed (new pairs)':40s}{'-':>22s}" + "".join(f"{c['contributed']:>22,}" for c in configs))
    out(f"    {'overlap with baseline candidates':40s}{'-':>22s}" + "".join(f"{c['overlap']:>22,}" for c in configs))
    out(f"    {'new candidates per new true pair':40s}{'-':>22s}"
        + "".join(f"{c['contributed'] / max(1, c['inc_true']):>22,.1f}" for c in configs))

    out("\n5. NO COUNTRY vs SAME COUNTRY")
    a, b = configs
    out(f"    rare-channel candidates: {a['rare']['pairs']:,} -> {b['rare']['pairs']:,} "
        f"({(1 - b['rare']['pairs'] / a['rare']['pairs']) * 100:.2f}% fewer)")
    out(f"    rare-channel true pairs: {a['rare']['tp']:,} -> {b['rare']['tp']:,} (lost {a['rare']['tp'] - b['rare']['tp']:,})")
    out(f"    incremental true pairs over baseline: {a['inc_true']:,} -> {b['inc_true']:,}")
    out(f"    candidates contributed over baseline: {a['contributed']:,} -> {b['contributed']:,}")

    out("\n6. FULL-SCALE ESTIMATE (measured 20k volume x {:.2f} = {:,} / {:,})".format(scale, total_s1, n_s1))
    for c in configs:
        out(f"    {c['label']}: rare channel ~{c['rare']['pairs'] * scale / 1e6:,.1f}M pairs; adds "
            f"~{c['contributed'] * scale / 1e6:,.1f}M new pairs over baseline; union "
            f"~{c['union']['pairs'] * scale / 1e6:,.1f}M pairs (baseline ~{base_m['pairs'] * scale / 1e6:,.1f}M)")
    out("    (per-S1 rare volume depends on full-corpus DF, which this experiment measured on the full S2/S3 "
        "files, so the linear scaling is not extrapolating DF from a sample)")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

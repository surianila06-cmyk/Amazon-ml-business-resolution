#!/usr/bin/env python3
"""
Bounded comparison of the SAME-COUNTRY rare-name-token channel at three DF
thresholds (200 / 500 / 1,000) against the baseline
    exact-name UNION top-200 address-by-Jaccard
on the same 20,000 stride-110 S1 sample. Experiment only -- no blocker code
changed, no candidate TSV written.

Channel definition (identical to rare_name_token_country_experiment_20k.py):
  name tokens = normalize_business_name split on whitespace, len >= 2;
  DF = combined S2+S3 rows containing the token; per S1 the single rarest
  token with 1 <= DF <= threshold (tie: token string); candidates = S2/S3
  rows containing it whose country equals the S1 country (strip + lower).
  Because the rarest token is always chosen, a lower threshold uses the SAME
  token as a higher one whenever it is eligible; it only drops S1 whose
  rarest token exceeds the threshold.

Inputs reused (no address regeneration):
  <cache-dir>/top200_address_candidates_20k.tsv   baseline address side
  <cache-dir>/true_pair_status.tsv                cross-check of baseline hits
  candidate_generation/output/train_candidate_pairs_exact_name.tsv
The name DF/postings were not persisted by the previous experiment, so one
name-only pass over S2/S3 rebuilds them for the tokens of the sampled S1
names; they are saved to <cache-dir>/rare_name_postings_20k.tsv (DF, and
id:country postings up to DF 1,000) and reused on later runs if present.

Output: candidate_generation/analysis/rare_name_df_threshold_comparison_20k.txt

Usage:
    python candidate_generation/analysis/rare_name_df_threshold_comparison_20k.py --cache-dir <dir>
"""
import argparse
import csv
import os
import statistics
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.analysis.exact_vs_address_union_20k import (  # noqa: E402
    EXACT_NAME_OUTPUT, GROUND_TRUTH, SAMPLE_SIZE, SOURCE1, SOURCE2, SOURCE3, load_exact_name_candidates,
)
from candidate_generation.analysis.missed_by_both_analysis_20k import name_tokens  # noqa: E402
from candidate_generation.analysis.rare_name_token_country_experiment_20k import country_key  # noqa: E402
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    count_rows, load_ground_truth_for_ids,
)
from utils.preprocessing import normalize_business_name  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rare_name_df_threshold_comparison_20k.txt")
THRESHOLDS = (("A", 200), ("B", 500), ("C", 1000))
MAX_DF = max(t for _, t in THRESHOLDS)


def load_candidate_tsv(path):
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            out[row[0]] = frozenset(row[1].split(",")) if len(row) > 1 and row[1] else frozenset()
    return out


def build_name_postings(needed):
    df = Counter()
    post = {}
    for path in (SOURCE2, SOURCE3):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            h = next(reader)
            ii, iname, ictry = (h.index(x) for x in ("entity_id", "business_name", "country"))
            for row in reader:
                toks = name_tokens(normalize_business_name(row[iname])) & needed
                if not toks:
                    continue
                ctry = country_key(row[ictry])
                for tok in toks:
                    df[tok] += 1
                    d = df[tok]
                    if d <= MAX_DF:
                        post.setdefault(tok, []).append((row[ii], ctry))
                    elif d == MAX_DF + 1:
                        post.pop(tok, None)
        print(f"  streamed {os.path.basename(path)}")
    return df, post


def save_postings(path, df, post):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# name-token DF over S2+S3 for tokens of the 20k sampled S1 names; postings kept for DF <= {MAX_DF}\n")
        f.write("token\tdf\tpostings(id:country)\n")
        for tok, d in df.items():
            p = ",".join(f"{c}:{k}" for c, k in post[tok]) if d <= MAX_DF else ""
            f.write(f"{tok}\t{d}\t{p}\n")


def load_postings(path):
    df, post = {}, {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith("#") or line.startswith("token\t"):
                continue
            tok, d, p = line.rstrip("\n").split("\t")
            df[tok] = int(d)
            if int(d) <= MAX_DF:
                post[tok] = [tuple(x.rsplit(":", 1)) for x in p.split(",")]
    return df, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    t_all = time.time()

    # ---- S1 sample (same stride sampler) -------------------------------------
    total_s1 = count_rows(SOURCE1)
    stride = max(1, total_s1 // SAMPLE_SIZE)
    s1 = []
    with open(SOURCE1, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(reader)
        ii, iname, ictry = (h.index(x) for x in ("entity_id", "business_name", "country"))
        for i, row in enumerate(reader):
            if i % stride == 0:
                s1.append((row[ii], row[iname], row[ictry]))
                if len(s1) >= SAMPLE_SIZE:
                    break
    n_s1 = len(s1)
    s1_ids = [r[0] for r in s1]
    s1_country = [country_key(r[2]) for r in s1]
    s1_ntoks = [name_tokens(normalize_business_name(r[1])) for r in s1]
    wanted = set(s1_ids)
    truth = load_ground_truth_for_ids(GROUND_TRUTH, wanted)
    total_true = sum(len(t) for t in truth.values() if t)
    ent_truth = sum(1 for t in truth.values() if t)

    # ---- baseline from saved artifacts ---------------------------------------
    exact = load_exact_name_candidates(EXACT_NAME_OUTPUT, wanted)
    top200 = load_candidate_tsv(os.path.join(args.cache_dir, "top200_address_candidates_20k.tsv"))
    if set(top200) != wanted:
        raise SystemExit("top200_address_candidates_20k.tsv does not cover the sample")
    baseline = [exact[e] | top200[e] for e in s1_ids]
    # cross-check against true_pair_status.tsv
    status_hits = 0
    with open(os.path.join(args.cache_dir, "true_pair_status.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            _, _, ie, _, it, _ = line.rstrip("\n").split("\t")
            status_hits += ie == "1" or it == "1"

    # ---- name DF / postings ----------------------------------------------------
    post_path = os.path.join(args.cache_dir, "rare_name_postings_20k.tsv")
    t0 = time.time()
    if os.path.exists(post_path):
        df, post = load_postings(post_path)
        how = f"loaded from {os.path.basename(post_path)}"
    else:
        df, post = build_name_postings(set().union(*s1_ntoks))
        save_postings(post_path, df, post)
        how = f"built by one name-only S2/S3 pass and saved to {os.path.basename(post_path)}"
    t_post = time.time() - t0

    rarest = []
    for toks in s1_ntoks:
        elig = sorted((df[t], t) for t in toks if 1 <= df.get(t, 0) <= MAX_DF)
        rarest.append((elig[0][1], elig[0][0]) if elig else None)

    def metrics(sets):
        sizes = [len(s) for s in sets]
        tp = cov = 0
        for e, s in zip(s1_ids, sets):
            t = truth.get(e)
            if t:
                h_ = len(s & t)
                tp += h_
                cov += h_ > 0
        pairs = sum(sizes)
        return {"pairs": pairs, "mean": pairs / n_s1, "median": statistics.median(sizes), "max": max(sizes),
                "zero": sum(1 for x in sizes if x == 0) / n_s1 * 100, "tp": tp,
                "recall": tp / total_true * 100, "cov": cov, "cov_pct": cov / ent_truth * 100}

    base_m = metrics(baseline)
    if base_m["tp"] != status_hits:
        raise SystemExit(f"baseline true hits {base_m['tp']} != true_pair_status {status_hits}")

    results = []
    for label, th in THRESHOLDS:
        t0 = time.time()
        rare = []
        used = []
        for i, sel in enumerate(rarest):
            if sel is None or sel[1] > th:
                rare.append(frozenset())
                continue
            used.append(sel[1])
            rare.append(frozenset(c for c, k in post[sel[0]] if k == s1_country[i]))
        union = [b | r for b, r in zip(baseline, rare)]
        inc = 0
        for e, r, b in zip(s1_ids, rare, baseline):
            t = truth.get(e)
            if t:
                inc += len((r - b) & t)
        results.append({
            "label": f"{label}: DF<={th:,}", "th": th, "used": used,
            "rare": metrics(rare), "union": metrics(union), "inc": inc,
            "new": sum(len(r - b) for r, b in zip(rare, baseline)),
            "overlap": sum(len(r & b) for r, b in zip(rare, baseline)),
            "elapsed": time.time() - t0,
        })
        print(f"  {label}: union recall {results[-1]['union']['recall']:.3f}%")

    # ---- report -----------------------------------------------------------------
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    scale = total_s1 / n_s1
    W = "=" * 110
    out(W)
    out("SAME-COUNTRY RARE-NAME-TOKEN: DF THRESHOLD COMPARISON (bounded 20k sample; experiment only)")
    out(W)
    out(f"Sample: {n_s1:,} of {total_s1:,} S1 (stride {stride}); {ent_truth:,} S1 with >=1 true match; {total_true:,} true pairs.")
    out("Baseline = exact-name U top-200 address, from saved top200_address_candidates_20k.tsv + "
        "train_candidate_pairs_exact_name.tsv (no address regeneration); baseline true hits cross-checked "
        f"against true_pair_status.tsv ({status_hits:,}, match).")
    out(f"Name DF/postings: {how} ({t_post:.0f}s). Channel: rarest name token (len>=2) with 1<=DF<=threshold, "
        "same-country candidates only.")

    out("\n1. ELIGIBLE TOKENS AND DF OF THE TOKEN USED")
    buckets = [(1, 1), (2, 5), (6, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
    out(f"    {'':28s}" + "".join(f"{r['label']:>18s}" for r in results))
    out(f"    {'S1 with eligible token':28s}" + "".join(f"{len(r['used']):>11,} ({len(r['used']) / n_s1 * 100:4.1f}%)" for r in results))
    for lo, hi in buckets:
        name = str(lo) if lo == hi else f"DF {lo}-{hi}"
        out(f"    {name:28s}" + "".join(
            f"{sum(1 for d in r['used'] if lo <= d <= hi):>18,}" for r in results))
    out(f"    {'median DF of used token':28s}" + "".join(f"{statistics.median(r['used']):>18.0f}" for r in results))
    out(f"    {'mean DF of used token':28s}" + "".join(f"{statistics.mean(r['used']):>18.1f}" for r in results))

    rows = [("candidate pairs", "pairs", "{:,}"), ("mean candidates/S1", "mean", "{:.2f}"),
            ("median candidates/S1", "median", "{:.1f}"), ("max candidates/S1", "max", "{:,}"),
            ("zero-candidate %", "zero", "{:.3f}%"), ("true pairs recovered", "tp", "{:,}"),
            ("pair-level recall", "recall", "{:.4f}%"), ("entities covered", "cov", "{:,}"),
            ("entity-level coverage", "cov_pct", "{:.4f}%")]

    out("\n2. RARE-NAME CHANNEL ONLY (same country)")
    out(f"    {'metric':36s}" + "".join(f"{r['label']:>18s}" for r in results))
    for name, key, fmt in rows:
        out(f"    {name:36s}" + "".join(f"{fmt.format(r['rare'][key]):>18s}" for r in results))

    out("\n3. UNION WITH BASELINE (exact-name U top-200 address U rare-name)")
    out(f"    {'metric':36s}{'Baseline':>18s}" + "".join(f"{'Base U ' + r['label'][:1]:>18s}" for r in results))
    for name, key, fmt in rows:
        label = "total union candidates" if key == "pairs" else name
        out(f"    {label:36s}{fmt.format(base_m[key]):>18s}" + "".join(f"{fmt.format(r['union'][key]):>18s}" for r in results))
    out(f"    {'incremental true pairs over base':36s}{'-':>18s}" + "".join(f"{r['inc']:>18,}" for r in results))
    out(f"    {'pair recall gain (pts)':36s}{'-':>18s}" + "".join(f"{'+%.4f' % (r['union']['recall'] - base_m['recall']):>18s}" for r in results))
    out(f"    {'entity coverage gain (pts)':36s}{'-':>18s}" + "".join(f"{'+%.4f' % (r['union']['cov_pct'] - base_m['cov_pct']):>18s}" for r in results))
    out(f"    {'new candidates added':36s}{'-':>18s}" + "".join(f"{r['new']:>18,}" for r in results))
    out(f"    {'candidate volume increase':36s}{'-':>18s}" + "".join(f"{'+%.1f%%' % (r['new'] / base_m['pairs'] * 100):>18s}" for r in results))
    out(f"    {'overlap with baseline':36s}{'-':>18s}" + "".join(f"{r['overlap']:>18,}" for r in results))
    out(f"    {'new candidates per new true pair':36s}{'-':>18s}" + "".join(f"{r['new'] / max(1, r['inc']):>18,.1f}" for r in results))

    out("\n4. RECALL vs CANDIDATE-VOLUME TRADEOFF")
    out(f"    baseline: {base_m['pairs']:,} candidates, {base_m['recall']:.4f}% pair recall, "
        f"{base_m['pairs'] / max(1, base_m['tp']):.1f} candidates per recovered true pair")
    prev = None
    for r in results:
        line = (f"    {r['label']:12s} +{r['new']:>9,} candidates (+{r['new'] / base_m['pairs'] * 100:5.1f}%) for "
                f"+{r['inc']:>5,} true pairs (+{r['union']['recall'] - base_m['recall']:.3f} pts): "
                f"{r['new'] / max(1, r['inc']):6.1f} new candidates per new true pair")
        if prev:
            dn, di = r["new"] - prev["new"], r["inc"] - prev["inc"]
            line += f" | marginal step from {prev['label'][:1]}: +{dn:,} cand for +{di:,} true ({dn / max(1, di):.1f}/pair)"
        out(line)
        prev = r

    out(f"\n5. FULL-SCALE VOLUME ESTIMATE (measured 20k volume x {scale:.2f})")
    out(f"    baseline ~{base_m['pairs'] * scale / 1e6:,.1f}M")
    for r in results:
        out(f"    {r['label']:12s} rare ~{r['rare']['pairs'] * scale / 1e6:,.1f}M; new over baseline "
            f"~{r['new'] * scale / 1e6:,.1f}M; union ~{r['union']['pairs'] * scale / 1e6:,.1f}M")
    out(f"\nTotal runtime: {time.time() - t_all:.0f}s")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

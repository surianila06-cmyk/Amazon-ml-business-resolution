#!/usr/bin/env python3
"""
Address post-filter analysis, step 2 of 2: read the per-pair feature cache
written by address_postfilter_features_20k.py and evaluate candidate
post-retrieval filters on the bounded 20,000-S1 address candidate set.
Writes candidate_generation/analysis/address_postfilter_20k.txt.

Analysis only: no blocker configuration is changed, no candidate TSV is
written.

Usage:
    python candidate_generation/analysis/address_postfilter_eval_20k.py --cache-dir <dir>
"""
import argparse
import json
import os
import statistics
import time
from array import array
from collections import Counter

OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "address_postfilter_20k.txt")


def load(cache, name, typecode):
    arr = array(typecode)
    path = os.path.join(cache, f"{name}.bin")
    with open(path, "rb") as f:
        arr.frombytes(f.read())
    return arr


def frac_bucket(v):
    """0..9 for [k/10, (k+1)/10), 10 for 1.0; epsilon absorbs float32 rounding."""
    return min(int(v * 10 + 1e-6), 10)


# Each filter: (label, predicate over (retr, shared, shared_num, jacc, overlap)).
THRESHOLD_FILTERS = [
    ("baseline: retrieval count >= 2 (current)", lambda r, s, n, j, o: True),
    ("shared DISTINCT tokens >= 2 (dedup keys)", lambda r, s, n, j, o: s >= 2),
    ("retrieval count >= 3", lambda r, s, n, j, o: r >= 3),
    ("shared tokens >= 3", lambda r, s, n, j, o: s >= 3),
    ("shared tokens >= 4", lambda r, s, n, j, o: s >= 4),
    ("Jaccard >= 0.1", lambda r, s, n, j, o: j >= 0.1),
    ("Jaccard >= 0.2", lambda r, s, n, j, o: j >= 0.2),
    ("Jaccard >= 0.3", lambda r, s, n, j, o: j >= 0.3),
    ("Jaccard >= 0.4", lambda r, s, n, j, o: j >= 0.4),
    ("overlap/S1 >= 0.5", lambda r, s, n, j, o: o >= 0.5),
    ("shared numeric >= 1", lambda r, s, n, j, o: n >= 1),
    ("Jaccard >= 0.2 AND shared numeric >= 1", lambda r, s, n, j, o: j >= 0.2 and n >= 1),
    ("shared >= 3 OR shared numeric >= 1", lambda r, s, n, j, o: s >= 3 or n >= 1),
    ("Jaccard >= 0.2 OR shared numeric >= 1", lambda r, s, n, j, o: j >= 0.2 or n >= 1),
    ("Jaccard >= 0.3 OR shared numeric >= 1", lambda r, s, n, j, o: j >= 0.3 or n >= 1),
    ("Jaccard >= 0.2 OR (numeric >= 1 AND J >= 0.1)", lambda r, s, n, j, o: j >= 0.2 or (n >= 1 and j >= 0.1)),
    ("shared>=2 AND Jaccard >= 0.3", lambda r, s, n, j, o: s >= 2 and j >= 0.3),
    ("shared>=2 AND (Jaccard >= 0.3 OR numeric >= 1)", lambda r, s, n, j, o: s >= 2 and (j >= 0.3 or n >= 1)),
    ("shared>=2 AND Jaccard >= 0.4", lambda r, s, n, j, o: s >= 2 and j >= 0.4),
]
TOP_K = (50, 100, 200, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    c = args.cache_dir
    with open(os.path.join(c, "meta.json")) as f:
        meta = json.load(f)
    offsets = load(c, "offsets", "q")
    retr = load(c, "retr", "B")
    shared = load(c, "shared", "B")
    shared_num = load(c, "shared_num", "B")
    jacc = load(c, "jacc", "f")
    overlap = load(c, "overlap", "f")
    is_true = load(c, "is_true", "B")
    char_idx = load(c, "char_idx", "q")
    char_sim = load(c, "char_sim", "f")

    n_s1 = meta["n_s1"]
    n_pairs = meta["n_pairs"]
    total_true = meta["total_true_pairs"]
    ent_truth = meta["entities_with_truth"]
    has_truth = meta["s1_has_truth"]
    base_pairs = n_pairs

    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    def summarize(label, per_s1_counts, per_s1_true, elapsed):
        pairs = sum(per_s1_counts)
        tp = sum(per_s1_true)
        cov = sum(1 for i in range(n_s1) if has_truth[i] and per_s1_true[i] > 0)
        return {
            "label": label, "pairs": pairs,
            "reduction_pct": (1 - pairs / base_pairs) * 100,
            "tp": tp, "recall": tp / total_true * 100,
            "cov": cov, "cov_pct": cov / ent_truth * 100,
            "false": pairs - tp,
            "mean": pairs / n_s1, "median": statistics.median(per_s1_counts),
            "max": max(per_s1_counts),
            "zero_pct": sum(1 for x in per_s1_counts if x == 0) / n_s1 * 100,
            "elapsed": elapsed,
        }

    results = []
    for label, pred in THRESHOLD_FILTERS:
        t0 = time.time()
        counts, trues = [], []
        for i in range(n_s1):
            a, b = offsets[i], offsets[i + 1]
            k = t = 0
            for r, s, n, j, o, tr in zip(retr[a:b], shared[a:b], shared_num[a:b],
                                         jacc[a:b], overlap[a:b], is_true[a:b]):
                if pred(r, s, n, j, o):
                    k += 1
                    t += tr
            counts.append(k)
            trues.append(t)
        results.append(summarize(label, counts, trues, time.time() - t0))
        print(f"  {label}: {results[-1]['pairs']:,} pairs, recall {results[-1]['recall']:.2f}%")

    for K in TOP_K:
        t0 = time.time()
        counts, trues = [], []
        for i in range(n_s1):
            a, b = offsets[i], offsets[i + 1]
            if b - a <= K:
                counts.append(b - a)
                trues.append(sum(is_true[a:b]))
                continue
            order = sorted(range(a, b), key=lambda p: (-jacc[p], -shared[p]))[:K]
            counts.append(K)
            trues.append(sum(is_true[p] for p in order))
        results.append(summarize(f"top-{K} per S1 by Jaccard (tie: shared)", counts, trues, time.time() - t0))
        print(f"  top-{K}: {results[-1]['pairs']:,} pairs, recall {results[-1]['recall']:.2f}%")

    # ---------------------------------------------------------------- report
    n_true_cand = sum(is_true)
    n_false_cand = n_pairs - n_true_cand
    tm = meta["timings_s"]
    out("=" * 100)
    out("ADDRESS POST-RETRIEVAL FILTER ANALYSIS -- bounded 20,000-S1 sample (analysis only)")
    out("=" * 100)
    out(f"Sample: {n_s1:,} of {meta['total_s1']:,} S1 (stride {meta['stride']}), same sampler as previous experiments.")
    out(f"Address config: min_shared_tokens=2, posting cap=5,000, stopwords={meta['n_stopwords']} "
        f"(current list + DF > 0.5%), same preprocessing.")
    out(f"Candidate pairs: {n_pairs:,}  (true {n_true_cand:,} / false {n_false_cand:,}); "
        f"all true pairs in sample: {total_true:,}; S1 with >=1 true match: {ent_truth:,}")
    out("")
    out("Feature definitions (Q = S1 qualifying address tokens, C = candidate qualifying tokens; qualifying =")
    out("address_blocking_keys minus the 161 stopwords; numeric = token with >= 3 digits):")
    out("  retrieval count = what the blocker thresholds (>=2); sums postings hits over the query key LIST,")
    out("                    so a token repeated in either address can count more than once")
    out("  shared          = |Q & C| on de-duplicated sets;  shared numeric = numeric tokens in Q & C")
    out("  Jaccard         = |Q & C| / |Q | C|;               overlap/S1 = |Q & C| / |Q|")
    out(f"  char 3-gram Jaccard on full normalized address strings: computed for ALL true pairs and a seeded "
        f"{meta['false_char_sample_rate'] * 100:.0f}% sample of false pairs only")
    out("")
    out("Cost of computing features (this run):")
    out(f"  index build (pass 1)            {tm['index_build']:8.1f}s")
    out(f"  candidate generation            {tm['candidate_generation']:8.1f}s")
    out(f"  forward token sets (pass 2)     {tm['forward_token_sets']:8.1f}s  (needs a candidate->tokens map; "
        f"not needed for retrieval-count filters)")
    out(f"  token features, all pairs       {tm['token_features']:8.1f}s  "
        f"({tm['token_features'] / n_pairs * 1e6:.2f} us/pair)")
    out(f"  char 3-gram Jaccard, sampled    {tm['char_features']:8.1f}s  "
        f"({tm['char_features'] / max(1, len(char_idx)) * 1e6:.2f} us/pair -> ~"
        f"{tm['char_features'] / max(1, len(char_idx)) * n_pairs / 60:.0f} min if run on all {n_pairs:,} pairs)")

    # distributions: one pass per feature, bucketed by key function
    def dist(arr, key, names, label):
        tcount, fcount = Counter(), Counter()
        for v, tr in zip(arr, is_true):
            (tcount if tr else fcount)[key(v)] += 1
        out("")
        out(f"DISTRIBUTION: {label}")
        out(f"  {'bucket':>10s} {'true':>10s} {'true %':>9s} {'false':>13s} {'false %':>9s} {'false:true':>11s}")
        for k, name in names:
            tc, fc = tcount[k], fcount[k]
            ratio = f"{fc / tc:,.0f}:1" if tc else "-"
            out(f"  {name:>10s} {tc:>10,} {tc / n_true_cand * 100:>8.2f}% {fc:>13,} "
                f"{fc / n_false_cand * 100:>8.2f}% {ratio:>11s}")

    def int_names(lo, cap):
        return [(k, str(k)) for k in range(lo, cap)] + [(cap, f"{cap}+")]

    dist(retr, lambda v: min(v, 6), int_names(2, 6), "retrieval count")
    dist(shared, lambda v: min(v, 6), int_names(0, 6), "shared qualifying tokens |Q & C|")
    dist(shared_num, lambda v: min(v, 4), int_names(0, 4), "shared numeric tokens")
    frac_names = [(k, f"[{k / 10:.1f},{(k + 1) / 10:.1f})") for k in range(10)] + [(10, "1.0")]
    dist(jacc, frac_bucket, frac_names, "token Jaccard")
    dist(overlap, frac_bucket, frac_names, "token overlap relative to S1 (|Q & C| / |Q|)")

    # char similarity (sampled false)
    rate = meta["false_char_sample_rate"]
    t_sims = [s for p, s in zip(char_idx, char_sim) if is_true[p]]
    f_sims = [s for p, s in zip(char_idx, char_sim) if not is_true[p]]
    out("")
    out(f"DISTRIBUTION: char 3-gram Jaccard (true: all {len(t_sims):,}; false: {len(f_sims):,} sampled, "
        f"counts scaled x{1 / rate:.0f})")
    out(f"  {'bucket':>10s} {'true':>10s} {'true %':>9s} {'false (est)':>13s} {'false %':>9s}")
    tb = Counter(frac_bucket(v) for v in t_sims)
    fb = Counter(frac_bucket(v) for v in f_sims)
    for k, name in frac_names:
        tc, fc = tb[k], fb[k]
        out(f"  {name:>10s} {tc:>10,} {tc / len(t_sims) * 100:>8.2f}% {fc / rate:>13,.0f} "
            f"{fc / len(f_sims) * 100:>8.2f}%")
    for q in (0.1, 0.25, 0.5):
        out(f"  true-pair char-sim quantile p{int(q * 100)}: {statistics.quantiles(t_sims, n=100)[int(q * 100) - 1]:.3f}"
            f" | false p{int(q * 100)}: {statistics.quantiles(f_sims, n=100)[int(q * 100) - 1]:.3f}")
    out("  char-sim threshold estimates (pair recall exact over true pairs; false/total counts estimated from sample;")
    out("  per-S1 stats not available because false pairs are sampled):")
    for th in (0.3, 0.4, 0.5):
        tp = sum(1 for v in t_sims if v >= th)
        fe = sum(1 for v in f_sims if v >= th) / rate
        out(f"    char-sim >= {th}: pairs ~{tp + fe:,.0f} (reduction ~{(1 - (tp + fe) / n_pairs) * 100:.1f}%), "
            f"true {tp:,}, pair recall {tp / total_true * 100:.2f}%, false ~{fe:,.0f}")

    # filter table
    out("")
    out("=" * 100)
    out(f"FILTER COMPARISON (applied after retrieval; baseline = {base_pairs:,} pairs; recall denominator = "
        f"{total_true:,} true pairs;")
    out(f"entity coverage denominator = {ent_truth:,} S1 with >=1 true match; runtime = filter evaluation over "
        f"precomputed features)")
    out("=" * 100)
    hdr = (f"{'filter':48s} {'pairs':>12s} {'reduct.':>8s} {'true rec':>9s} {'pair rec':>9s} {'ent cov':>9s} "
           f"{'false':>12s} {'mean':>8s} {'median':>7s} {'max':>7s} {'zero%':>7s} {'time s':>7s}")
    out(hdr)
    out("-" * len(hdr))
    for r in results:
        out(f"{r['label'][:48]:48s} {r['pairs']:>12,} {r['reduction_pct']:>7.2f}% {r['tp']:>9,} "
            f"{r['recall']:>8.2f}% {r['cov_pct']:>8.2f}% {r['false']:>12,} {r['mean']:>8.1f} "
            f"{r['median']:>7.1f} {r['max']:>7,} {r['zero_pct']:>6.2f}% {r['elapsed']:>7.1f}")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analysis of true pairs MISSED BY BOTH Stage 1 (exact normalized name) and the
selected Stage 2 address channel (top-200 per S1 by token Jaccard; retrieval
min_shared_tokens=2, cap=5,000, 161 stopwords) on the same 20,000 stride-110
S1 sample. Goal: characterize the misses and size a possible THIRD retrieval
channel from data (nothing is implemented).

One streaming pass over S2/S3 does three things at once:
  - rebuilds the pruned address index (same semantics as
    exact_vs_address_union_20k.build_pruned_address_index); candidates are
    regenerated and checked position-by-position against the feature cache
    (offsets + retrieval counts) before the cached Jaccard/shared arrays are
    used to recompute the top-200 selection;
  - counts per-source normalized-NAME token document frequency (used only to
    estimate the candidate volume of a rare-name-token channel);
  - keeps the raw rows of every ground-truth target of the sampled S1.

Also writes <cache-dir>/true_pair_status.tsv (every sampled true pair with
exact / address-retrieved / top-200 flags) so later analyses need not
regenerate candidates. Ground truth is used for analysis only.

No candidate TSV, no blocker change, no full-scale run.

Output: candidate_generation/analysis/missed_by_both_analysis_20k.txt

Usage:
    python candidate_generation/analysis/missed_by_both_analysis_20k.py --cache-dir <dir>
"""
import argparse
import csv
import os
import statistics
import sys
import time
import unicodedata
from array import array
from collections import Counter, defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import address_blocking_keys, address_tokens  # noqa: E402
from candidate_generation.analysis.address_postfilter_eval_20k import load  # noqa: E402
from candidate_generation.analysis.address_postfilter_features_20k import char3grams  # noqa: E402
from candidate_generation.analysis.exact_vs_address_union_20k import (  # noqa: E402
    DF_THRESHOLD_PCT, EXACT_NAME_OUTPUT, GROUND_TRUTH, MIN_SHARED_TOKENS, POSTING_CAP,
    SAMPLE_SIZE, SOURCE1, SOURCE2, SOURCE3, load_exact_name_candidates,
)
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

OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missed_by_both_analysis_20k.txt")
TOP_K = 200
SIM_HIGH = 0.5
NAME_DF_GRID = (50, 200, 1000, 5000)
NAME_K_GRID = (1, 2, 3)


# ---------------------------------------------------------------- helpers
def name_tokens(norm_name):
    return {t for t in norm_name.split() if len(t) >= 2}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_sim(a, b):
    if not a or not b:
        return 0.0
    return jaccard(char3grams(a), char3grams(b))


def dominant_script(text):
    c = Counter()
    for ch in text:
        if ch.isalpha():
            c[unicodedata.name(ch, "UNKNOWN").split()[0]] += 1
    return c.most_common(1)[0][0] if c else "NONE"


def norm_country(c):
    c = (c or "").strip().lower()
    return "" if c in ("", "null", "none", "nan", "n/a", "na", "-", "unknown") else c


def abbreviation_like(ta, tb):
    """Heuristic: initials of one name's tokens form a token of the other, or
    a token of one is a strict prefix (>= 2 chars) of a token of the other."""
    for x, y in ((ta, tb), (tb, ta)):
        if len(x) >= 2 and "".join(t[0] for t in sorted(x)) in y:
            return True
    for x in ta:
        for y in tb:
            if x != y and len(x) >= 2 and len(y) >= 2 and (y.startswith(x) or x.startswith(y)):
                return True
    return False


def pct(n, d):
    return n / d * 100 if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    t_all = time.time()

    stopwords = stopwords_for_threshold(load_df_table(DF_TABLE_PATH), DF_THRESHOLD_PCT)

    # ---- S1 sample with all fields (same stride sampler as sample_source1) ---
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
    s1_ids = [r[0] for r in s1]
    s1_norm_addr = [normalize_business_address(r[2]) for r in s1]
    s1_keys = [[t for t in address_blocking_keys(n) if t not in stopwords] for n in s1_norm_addr]
    needed = {t for keys in s1_keys for t in keys}

    wanted = set(s1_ids)
    truth = load_ground_truth_for_ids(GROUND_TRUTH, wanted)
    exact = load_exact_name_candidates(EXACT_NAME_OUTPUT, wanted)
    target_ids = set().union(*[t for t in truth.values() if t])

    # ---- one pass over S2/S3: address index + name DF + target rows ---------
    t0 = time.time()
    indexes = {}
    name_df = {}
    targets = {}
    for src, path in (("S2", SOURCE2), ("S3", SOURCE3)):
        index = {}
        df = Counter()
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            h = next(reader)
            ii, iname, iaddr, ictry = (h.index(x) for x in ("entity_id", "business_name", "business_address", "country"))
            for row in reader:
                norm_addr = normalize_business_address(row[iaddr])
                for tok in address_blocking_keys(norm_addr):
                    if tok in needed:
                        lst = index.setdefault(tok, [])
                        if len(lst) < POSTING_CAP:
                            lst.append(row[ii])
                norm_name = normalize_business_name(row[iname])
                df.update(name_tokens(norm_name))
                if row[ii] in target_ids:
                    targets[row[ii]] = (row[iname], row[iaddr], row[ictry], norm_name, norm_addr)
        indexes[src] = index
        name_df[src] = df
        print(f"  streamed {src}: {len(df):,} distinct name tokens")
    t_pass = time.time() - t0

    # ---- regenerate address candidates and align with the cache -------------
    c_offsets = load(args.cache_dir, "offsets", "q")
    c_retr = load(args.cache_dir, "retr", "B")
    offsets = array("q", [0])
    pair_cid = []
    retr = array("B")
    for keys in s1_keys:
        for src in ("S2", "S3"):
            counts = Counter()
            for tok in keys:
                counts.update(indexes[src].get(tok, ()))
            for cid, ct in counts.items():
                if ct >= MIN_SHARED_TOKENS:
                    pair_cid.append(cid)
                    retr.append(min(ct, 255))
        offsets.append(len(pair_cid))
    del indexes
    if offsets != c_offsets or retr != c_retr:
        raise SystemExit("regenerated candidates do not align with the feature cache -- aborting")
    jacc = load(args.cache_dir, "jacc", "f")
    shared = load(args.cache_dir, "shared", "B")
    print(f"regenerated {len(pair_cid):,} address candidates; aligned with cache")

    # ---- status of every sampled true pair ----------------------------------
    status = []  # (s1_idx, tid, in_exact, in_addr_raw, in_top200, addr_rank)
    for i, eid in enumerate(s1_ids):
        t = truth.get(eid)
        if not t:
            continue
        a, b = offsets[i], offsets[i + 1]
        order = sorted(range(a, b), key=lambda p: (-jacc[p], -shared[p]))
        rank = {pair_cid[p]: r for r, p in enumerate(order) if pair_cid[p] in t}
        for tid in sorted(t):
            r = rank.get(tid)
            status.append((i, tid, tid in exact[eid], r is not None, r is not None and r < TOP_K,
                           -1 if r is None else r + 1))
    del pair_cid, jacc, shared
    with open(os.path.join(args.cache_dir, "true_pair_status.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\ttrue_entity_id\tin_exact\tin_address_retrieval\tin_top200\taddress_rank\n")
        for i, tid, ie, ia, it, r in status:
            f.write(f"{s1_ids[i]}\t{tid}\t{int(ie)}\t{int(ia)}\t{int(it)}\t{r}\n")

    total_true = len(status)
    recovered = [s for s in status if s[2] or s[4]]
    missed = [s for s in status if not s[2] and not s[4]]

    # ---- per-pair descriptors ----------------------------------------------
    s1_norm_name = [normalize_business_name(r[1]) for r in s1]

    def describe(s):
        i, tid, _, in_addr_raw, _, _ = s
        tname, taddr, tctry, tnn, tna = targets[tid]
        nn1, na1 = s1_norm_name[i], s1_norm_addr[i]
        nt1, nt2 = name_tokens(nn1), name_tokens(tnn)
        q1 = {t for t in s1_keys[i]}
        q2 = {t for t in address_blocking_keys(tna) if t not in stopwords}
        raw1, raw2 = set(address_tokens(na1)), set(address_tokens(tna))
        shared_q = q1 & q2
        c1, c2 = norm_country(s1[i][3]), norm_country(tctry)
        d = {
            "src": tid[:2],
            "name_exact": nn1 == tnn and nn1 != "",
            "name_empty": not nn1 or not tnn,
            "name_tok_j": jaccard(nt1, nt2),
            "name_char": char_sim(nn1, tnn),
            "name_same_tokens_diff_order": bool(nt1) and nt1 == nt2 and nn1 != tnn,
            "name_subset": bool(nt1) and bool(nt2) and nt1 != nt2 and (nt1 <= nt2 or nt2 <= nt1),
            "name_abbrev": abbreviation_like(nt1, nt2),
            "addr_empty": not na1 or not tna,
            "addr_q_j": jaccard(q1, q2),
            "addr_shared": len(shared_q),
            "addr_shared_num": sum(1 for t in shared_q if any(ch.isdigit() for ch in t)),
            "addr_raw_j": jaccard(raw1, raw2),
            "addr_char": char_sim(na1, tna),
            "s1_q_len": len(q1), "t_q_len": len(q2),
            "s1_raw_len": len(raw1), "t_raw_len": len(raw2),
            "country": "missing" if not c1 or not c2 else ("same" if c1 == c2 else "different"),
            "name_script_diff": dominant_script(nn1) != dominant_script(tnn),
            "addr_script_diff": bool(na1) and bool(tna) and dominant_script(na1) != dominant_script(tna),
            "name_scripts": f"{dominant_script(nn1)}->{dominant_script(tnn)}",
            "in_addr_raw": in_addr_raw,
            "nt1": nt1, "nt2": nt2,
        }
        if d["addr_empty"]:
            cat = "missing address (S1 or target)"
        elif d["name_script_diff"] or d["addr_script_diff"]:
            cat = "script/transliteration variation"
        elif in_addr_raw:
            cat = "address retrieved but ranked > 200"
        else:
            name_ok = d["name_tok_j"] >= SIM_HIGH
            addr_ok = d["addr_char"] >= SIM_HIGH
            if name_ok and addr_ok:
                cat = "name & address both similar (blocking-rule gap)"
            elif name_ok:
                cat = "name similar, address heavily changed"
            elif addr_ok:
                cat = "address similar, name heavily changed"
            else:
                cat = "both name and address heavily changed"
        d["category"] = cat
        return d

    t0 = time.time()
    M = [describe(s) for s in missed]
    R = [describe(s) for s in recovered]
    t_desc = time.time() - t0

    # ---- report -------------------------------------------------------------
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    def frac_hist(key, label):
        out(f"\n  {label}")
        out(f"    {'bucket':>12s} {'missed':>9s} {'missed %':>9s} {'recovered':>10s} {'recov %':>8s}")
        bk = lambda v: min(int(v * 10 + 1e-9), 10)  # noqa: E731
        cm = Counter(bk(d[key]) for d in M)
        cr = Counter(bk(d[key]) for d in R)
        for k in range(11):
            name = "1.0" if k == 10 else f"[{k / 10:.1f},{(k + 1) / 10:.1f})"
            out(f"    {name:>12s} {cm[k]:>9,} {pct(cm[k], len(M)):>8.2f}% {cr[k]:>10,} {pct(cr[k], len(R)):>7.2f}%")
        out(f"    median: missed {statistics.median(d[key] for d in M):.3f} | "
            f"recovered {statistics.median(d[key] for d in R):.3f}")

    def int_hist(key, label, cap):
        out(f"\n  {label}")
        out(f"    {'bucket':>12s} {'missed':>9s} {'missed %':>9s} {'recovered':>10s} {'recov %':>8s}")
        cm = Counter(min(d[key], cap) for d in M)
        cr = Counter(min(d[key], cap) for d in R)
        for k in range(cap + 1):
            name = f"{k}+" if k == cap else str(k)
            out(f"    {name:>12s} {cm[k]:>9,} {pct(cm[k], len(M)):>8.2f}% {cr[k]:>10,} {pct(cr[k], len(R)):>7.2f}%")

    def flag(key, label):
        m = sum(1 for d in M if d[key])
        r = sum(1 for d in R if d[key])
        out(f"    {label:52s} missed {m:>7,} ({pct(m, len(M)):6.2f}%)   recovered {r:>7,} ({pct(r, len(R)):6.2f}%)")

    W = "=" * 100
    out(W)
    out("TRUE PAIRS MISSED BY BOTH exact-name AND top-200 address (bounded 20k sample; analysis only)")
    out(W)
    out(f"Sample: {len(s1_ids):,} of {total_s1:,} S1 (stride {stride}); true pairs {total_true:,}; "
        f"recovered by exact U top-200 {len(recovered):,} ({pct(len(recovered), total_true):.2f}%); "
        f"MISSED BY BOTH {len(missed):,} ({pct(len(missed), total_true):.2f}%).")
    out("Recovered pairs are shown alongside as a reference distribution.")
    out(f"Similarity definitions: name tokens = normalize_business_name split on spaces, len >= 2; address "
        f"qualifying tokens = address_blocking_keys minus the {len(stopwords)} stopwords; char sim = 3-gram "
        f"Jaccard of the normalized strings; 'similar' threshold = {SIM_HIGH}.")
    out(f"One S2/S3 pass (address index + name DF + target rows): {t_pass:.0f}s; descriptors: {t_desc:.1f}s.")

    out("\n1. SOURCE BREAKDOWN")
    for src in ("S2", "S3"):
        tot = sum(1 for s in status if s[1].startswith(src))
        m = sum(1 for d in M if d["src"] == src)
        out(f"    {src}: missed {m:,} of {tot:,} true pairs ({pct(m, tot):.2f}% of {src} true pairs; "
            f"{pct(m, len(M)):.2f}% of all misses)")

    out("\n2. NAME SIMILARITY (S1 vs true target)")
    flag("name_exact", "normalized name exactly equal")
    flag("name_empty", "name empty on either side")
    flag("name_same_tokens_diff_order", "same name token set, different order")
    flag("name_subset", "one name's tokens a strict subset of the other")
    flag("name_abbrev", "abbreviation-like (initials / token prefix)")
    frac_hist("name_tok_j", "name token Jaccard")
    frac_hist("name_char", "name char 3-gram Jaccard")

    out("\n3. ADDRESS SIMILARITY")
    flag("addr_empty", "normalized address empty on either side")
    flag("in_addr_raw", "retrieved by address channel (before top-200)")
    frac_hist("addr_q_j", "qualifying-token Jaccard (what top-200 ranks on)")
    frac_hist("addr_raw_j", "raw normalized-token Jaccard (incl. stopwords)")
    frac_hist("addr_char", "address char 3-gram Jaccard")
    int_hist("addr_shared", "shared distinct qualifying tokens", 4)
    int_hist("addr_shared_num", "shared numeric tokens", 3)
    int_hist("s1_q_len", "S1 qualifying-token count", 6)
    int_hist("t_q_len", "target qualifying-token count", 6)
    int_hist("s1_raw_len", "S1 raw normalized-token count", 10)

    out("\n4. COUNTRY AGREEMENT")
    for grp, D in (("missed", M), ("recovered", R)):
        c = Counter(d["country"] for d in D)
        out(f"    {grp:10s} " + "  ".join(f"{k}: {c[k]:,} ({pct(c[k], len(D)):.2f}%)" for k in ("same", "different", "missing")))

    out("\n5. SCRIPT VARIATION")
    flag("name_script_diff", "dominant script of names differs")
    flag("addr_script_diff", "dominant script of addresses differs")
    sc = Counter(d["name_scripts"] for d in M if d["name_script_diff"])
    out("    top name script transitions among misses: "
        + ", ".join(f"{k} {v:,}" for k, v in sc.most_common(6)))

    out("\n6. PRIMARY FAILURE CATEGORY (exclusive, assigned in this priority order)")
    cats = Counter(d["category"] for d in M)
    order = ["missing address (S1 or target)", "script/transliteration variation",
             "address retrieved but ranked > 200", "name & address both similar (blocking-rule gap)",
             "name similar, address heavily changed", "address similar, name heavily changed",
             "both name and address heavily changed"]
    for c in order:
        sub = [d for d in M if d["category"] == c]
        s2 = sum(1 for d in sub if d["src"] == "S2")
        out(f"    {c:50s} {cats[c]:>7,} ({pct(cats[c], len(M)):6.2f}%)   S2 {s2:,} / S3 {len(sub) - s2:,}")
    out("    Secondary (non-exclusive) name patterns within misses: word order "
        f"{sum(d['name_same_tokens_diff_order'] for d in M):,}; token subset {sum(d['name_subset'] for d in M):,}; "
        f"abbreviation-like {sum(d['name_abbrev'] for d in M):,}")

    out("\n    Examples per category (S1 name | S1 address  ==>  target name | target address):")
    ex = defaultdict(list)
    for s, d in zip(missed, M):
        if len(ex[d["category"]]) < 4:
            ex[d["category"]].append((s, d))
    for c in order:
        out(f"    [{c}]")
        for s, d in ex[c]:
            i, tid = s[0], s[1]
            tn, ta = targets[tid][0], targets[tid][1]
            out(f"      {s1[i][1][:40]!r} | {s1[i][2][:50]!r}  ==>  {tn[:40]!r} | {ta[:50]!r}")

    # ---- third channel sizing: rare name-token retrieval ---------------------
    out("\n" + W)
    out("7. THIRD-CHANNEL SIZING (estimates only; nothing implemented)")
    out(W)
    out("  Rare-name-token retrieval: for each S1, take its k rarest normalized-name tokens whose combined")
    out("  S2+S3 document frequency <= maxDF; a target is retrieved if its name contains any selected token.")
    out("  Volume = sum of selected tokens' DF (exact for k=1, UPPER BOUND for k>1 because a target sharing")
    out("  several selected tokens is counted once per token); duplicates with existing channels not removed.")
    out("  'new' = missed-by-both pairs recovered; 'overlap' = already-recovered pairs also hit.")
    combined_df = name_df["S2"].copy()
    combined_df.update(name_df["S3"])
    del name_df
    s1_nt = [name_tokens(n) for n in s1_norm_name]
    hdr = (f"    {'k':>2s} {'maxDF':>6s} {'new pairs':>10s} {'+recall':>8s} {'overlap':>9s} {'vol (20k)':>12s} "
           f"{'mean/S1':>8s} {'median':>7s} {'max':>7s} {'S1 w/ token':>11s} {'vol x110 (full est)':>20s}")
    out(hdr)
    missed_by_s1 = defaultdict(list)
    for s, d in zip(missed, M):
        missed_by_s1[s[0]].append(d["nt2"])
    rec_by_s1 = defaultdict(list)
    for s, d in zip(recovered, R):
        rec_by_s1[s[0]].append(d["nt2"])
    best = None
    for k in NAME_K_GRID:
        for max_df in NAME_DF_GRID:
            vols = []
            new = ov = has = 0
            for i, toks in enumerate(s1_nt):
                elig = sorted((combined_df.get(t, 0), t) for t in toks if 0 < combined_df.get(t, 0) <= max_df)[:k]
                sel = {t for _, t in elig}
                vols.append(sum(c for c, _ in elig))
                has += bool(sel)
                new += sum(1 for nt2 in missed_by_s1.get(i, ()) if nt2 & sel)
                ov += sum(1 for nt2 in rec_by_s1.get(i, ()) if nt2 & sel)
            vol = sum(vols)
            out(f"    {k:>2d} {max_df:>6,} {new:>10,} {pct(new, total_true):>7.2f}% {ov:>9,} {vol:>12,} "
                f"{vol / len(vols):>8.1f} {statistics.median(vols):>7.1f} {max(vols):>7,} "
                f"{pct(has, len(vols)):>10.1f}% {vol * total_s1 / len(vols):>20,.0f}")
            if (k, max_df) == (2, 1000):
                best = (new, ov, vol)

    out("\n  Character n-gram name retrieval potential (recall ceiling only; volume not estimated here --")
    out("  3-gram postings over 10.3M names are far denser than token postings):")
    for th in (0.3, 0.5, 0.7):
        n = sum(1 for d in M if d["name_char"] >= th)
        out(f"    missed pairs with name char 3-gram Jaccard >= {th}: {n:,} ({pct(n, len(M)):.2f}% of misses, "
            f"+{pct(n, total_true):.2f} pts recall ceiling)")
    n_tok = sum(1 for d in M if d["nt1"] & d["nt2"])
    out(f"    missed pairs sharing >= 1 name token (any DF): {n_tok:,} ({pct(n_tok, len(M)):.2f}% of misses) "
        f"-- ceiling for any token-based name channel")
    n_num1 = sum(1 for d in M if d["addr_shared_num"] >= 1)
    out(f"  Another address signal: missed pairs sharing >= 1 numeric qualifying address token: {n_num1:,} "
        f"({pct(n_num1, len(M)):.2f}% of misses); sharing exactly 1 qualifying token: "
        f"{sum(1 for d in M if d['addr_shared'] == 1):,}")
    if best:
        out(f"\n  Reference point k=2, maxDF=1,000: +{best[0]:,} new true pairs "
            f"(+{pct(best[0], total_true):.2f} pts), overlap {best[1]:,}, <= {best[2]:,} candidates on 20k "
            f"(<= ~{best[2] * total_s1 / len(s1_ids) / 1e6:.0f}M at full scale, before dedup with other channels)")
    out(f"\nTotal runtime: {time.time() - t_all:.0f}s")

    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Address post-filter analysis, step 1 of 2: build per-pair features for the
bounded Stage 2 address candidate set (same 20,000 stride-sampled S1,
min_shared_tokens=2, cap=5,000, stopwords = current list + DF > 0.5%).

Pass 1 streams S2/S3 into a pruned address index (only tokens the sample
queries; file-order first-seen cap, so identical to a full-index run) and
generates the candidate pairs. Pass 2 streams S2/S3 again and, for candidate
ids only, stores the candidate's qualifying token set (same keys + stopwords
as the query side) -- rows that are not candidates are skipped before
normalization.

Per (S1, candidate) pair features, with Q = S1 qualifying tokens and C =
candidate qualifying tokens:
  retr      retrieval count: what the blocker itself thresholds at >= 2
            (sum over the query key LIST of postings hits, so a token
            repeated in either address counts more than once)
  shared    |Q & C| over de-duplicated qualifying token sets
  shared_num  |Q & C| restricted to numeric tokens (>= 3 digits, per
            is_keepable_address_token)
  jaccard   |Q & C| / |Q | C|
  overlap   |Q & C| / |Q|
  is_true   ground-truth match (scoring only)

Character-level similarity (3-gram Jaccard on the full normalized address
strings) is NOT computed over all ~20M pairs; it is computed for ALL true
pairs plus a seeded 1% random sample of false pairs, so its distribution
can be compared without an expensive all-pairs pass.

Features are written as raw `array` files to --cache-dir (outside the repo,
~250MB) and consumed by address_postfilter_eval_20k.py. No candidate TSV is
written; address_blocking.py / preprocessing.py are not modified.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from array import array
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import address_blocking_keys  # noqa: E402
from candidate_generation.analysis.exact_vs_address_union_20k import (  # noqa: E402
    DF_THRESHOLD_PCT, GROUND_TRUTH, MIN_SHARED_TOKENS, POSTING_CAP, SAMPLE_SIZE,
    SOURCE1, SOURCE2, SOURCE3, build_pruned_address_index,
)
from candidate_generation.run_address_blocking_bounded_experiment import (  # noqa: E402
    load_ground_truth_for_ids, sample_source1,
)
from candidate_generation.run_address_blocking_cap_stopword_experiment import (  # noqa: E402
    DF_TABLE_PATH, load_df_table, stopwords_for_threshold,
)
from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

FALSE_CHAR_SAMPLE_RATE = 0.01
SEED = 20260926


def qualifying_tokens(norm, stopwords):
    return {sys.intern(t) for t in address_blocking_keys(norm) if t not in stopwords}


def is_numeric(tok):
    return any(ch.isdigit() for ch in tok)


def char3grams(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    t_all = time.time()

    stopwords = stopwords_for_threshold(load_df_table(DF_TABLE_PATH), DF_THRESHOLD_PCT)
    sample_rows, total_s1, stride = sample_source1(SOURCE1, SAMPLE_SIZE)
    s1_ids = [eid for eid, _ in sample_rows]
    s1_norm = [normalize_business_address(addr) for _, addr in sample_rows]
    # Retrieval iterates the key LIST (duplicates included), exactly as the
    # blocker does; similarity features use the de-duplicated SET.
    s1_keys = [[t for t in address_blocking_keys(n) if t not in stopwords] for n in s1_norm]
    s1_q = [set(map(sys.intern, k)) for k in s1_keys]
    del sample_rows
    truth = load_ground_truth_for_ids(GROUND_TRUTH, set(s1_ids))
    needed = set().union(*s1_q)
    print(f"sample {len(s1_ids):,} (stride {stride}); stopwords {len(stopwords)}; query tokens {len(needed):,}")

    # ---- pass 1: pruned index + candidate generation -------------------------
    t0 = time.time()
    s2_index = build_pruned_address_index(SOURCE2, needed, POSTING_CAP)
    s3_index = build_pruned_address_index(SOURCE3, needed, POSTING_CAP)
    t_index = time.time() - t0
    print(f"pass 1 index build: {t_index:.1f}s")

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
    del s2_index, s3_index
    t_gen = time.time() - t0
    print(f"candidate generation: {len(pair_cid):,} pairs in {t_gen:.1f}s")

    # Pairs for character-level similarity: all true + seeded 1% of false.
    rng = random.Random(SEED)
    is_true = array("B", bytes(len(pair_cid)))
    char_pairs = []  # (pair_index, s1_index)
    for i, eid in enumerate(s1_ids):
        t = truth.get(eid, frozenset())
        for p in range(offsets[i], offsets[i + 1]):
            if pair_cid[p] in t:
                is_true[p] = 1
                char_pairs.append((p, i))
            elif rng.random() < FALSE_CHAR_SAMPLE_RATE:
                char_pairs.append((p, i))
    need_str = {pair_cid[p] for p, _ in char_pairs}

    # ---- pass 2: forward token sets for candidate ids only -------------------
    t0 = time.time()
    cand_ids = set(pair_cid)
    fwd = {}
    cand_norm = {}
    for path in (SOURCE2, SOURCE3):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            idx_id, idx_addr = header.index("entity_id"), header.index("business_address")
            for row in reader:
                cid = row[idx_id]
                if cid not in cand_ids:
                    continue
                norm = normalize_business_address(row[idx_addr])
                fwd[cid] = tuple(qualifying_tokens(norm, stopwords))
                if cid in need_str:
                    cand_norm[cid] = norm
    del cand_ids
    t_fwd = time.time() - t0
    print(f"pass 2 forward token sets: {len(fwd):,} distinct candidates in {t_fwd:.1f}s")

    # ---- per-pair features ---------------------------------------------------
    t0 = time.time()
    shared = array("B")
    shared_num = array("B")
    jacc = array("f")
    overlap = array("f")
    for i, q in enumerate(s1_q):
        qn = len(q)
        qnum = {t for t in q if is_numeric(t)}
        for p in range(offsets[i], offsets[i + 1]):
            c = fwd[pair_cid[p]]
            s = sn = 0
            for t in c:
                if t in q:
                    s += 1
                    if t in qnum:
                        sn += 1
            shared.append(min(s, 255))
            shared_num.append(min(sn, 255))
            jacc.append(s / (qn + len(c) - s))
            overlap.append(s / qn)
    t_feat = time.time() - t0
    print(f"per-pair token features: {t_feat:.1f}s ({t_feat / len(pair_cid) * 1e6:.2f} us/pair)")

    t0 = time.time()
    s1_grams = {}
    char_idx = array("q")
    char_sim = array("f")
    for p, i in char_pairs:
        g1 = s1_grams.get(i)
        if g1 is None:
            g1 = s1_grams[i] = char3grams(s1_norm[i])
        g2 = char3grams(cand_norm[pair_cid[p]])
        union = len(g1 | g2)
        char_idx.append(p)
        char_sim.append(len(g1 & g2) / union if union else 0.0)
    t_char = time.time() - t0
    print(f"char 3-gram Jaccard on {len(char_pairs):,} pairs: {t_char:.1f}s "
          f"({t_char / max(1, len(char_pairs)) * 1e6:.2f} us/pair)")

    for name, arr in (("offsets", offsets), ("retr", retr), ("shared", shared),
                      ("shared_num", shared_num), ("jacc", jacc), ("overlap", overlap),
                      ("is_true", is_true), ("char_idx", char_idx), ("char_sim", char_sim)):
        with open(os.path.join(args.cache_dir, f"{name}.bin"), "wb") as f:
            arr.tofile(f)

    total_true = sum(len(truth[e]) for e in s1_ids if truth.get(e))
    entities_with_truth = sum(1 for e in s1_ids if truth.get(e))
    meta = {
        "n_s1": len(s1_ids), "total_s1": total_s1, "stride": stride, "n_pairs": len(pair_cid),
        "n_stopwords": len(stopwords), "total_true_pairs": total_true,
        "entities_with_truth": entities_with_truth,
        "s1_has_truth": [1 if truth.get(e) else 0 for e in s1_ids],
        "false_char_sample_rate": FALSE_CHAR_SAMPLE_RATE, "seed": SEED,
        "timings_s": {"index_build": t_index, "candidate_generation": t_gen,
                      "forward_token_sets": t_fwd, "token_features": t_feat,
                      "char_features": t_char, "total": time.time() - t_all},
    }
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote features to {args.cache_dir}; total {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()

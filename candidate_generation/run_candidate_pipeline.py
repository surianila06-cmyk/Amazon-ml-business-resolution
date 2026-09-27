#!/usr/bin/env python3
"""
ML Challenge 2026 -- Role 2: Candidate Generation / Blocking (CLI)
PRODUCTION pipeline: exact-name UNION top-200 address UNION same-country
rare-name-token (see candidate_pipeline.py for the locked configuration).

The candidate file written here is the FINAL union produced by
candidate_pipeline.S1Candidates.union -- the exact set intended for Role 3.
No intermediate stage is ever written as a candidate file.

Outputs (for --out X.tsv):
  X.tsv                    header 'source1_entity_id\\tcandidate_entity_ids',
                           one row per selected S1 (zero-candidate rows kept),
                           candidate ids sorted, comma-separated
  X.channel_counts.tsv     per-S1 channel sizes and unique contributions (no
                           ground truth); used for the run summary
  X.summary.json           run summary (volumes, per-channel contribution,
                           runtime, peak memory)
  X.channels.tsv           only with --channel-diagnostics: per-S1 id lists
                           per channel, for validation runs
While running, X.tsv.partial + X.checkpoint.json allow --resume after an
interruption (indexes are rebuilt deterministically; already-written S1 rows
are skipped).

Ground truth is never read. Writing a file named candidate_pairs.tsv needs
--allow-final.

Usage (20k validation, same stride sample as the bounded experiments):
    python candidate_generation/run_candidate_pipeline.py \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --sample-size 20000 --channel-diagnostics \\
        --out candidate_generation/output/pilot_candidate_pairs.tsv
"""

import argparse
import json
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.candidate_pipeline import (  # noqa: E402
    DEFAULT_STOPWORDS_PATH,
    LOCKED_CONFIG,
    build_target_index,
    collect_s1_vocabulary,
    generate_candidates,
    iter_source1_selection,
    load_address_stopwords,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except AttributeError:
    pass

# *_only = pairs contributed by that channel alone; rare_only is also the
# rare channel's new pairs over (exact U address).
COUNT_COLUMNS = ("exact", "address", "rare", "union", "exact_only", "address_only", "rare_only")


def peak_memory_mb():
    """Peak resident memory of this process in MB (Windows or POSIX)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def channel_counts(c):
    union = c.exact | c.address | c.rare
    return (len(c.exact), len(c.address), len(c.rare), len(union),
            len(c.exact - c.address - c.rare), len(c.address - c.exact - c.rare),
            len(c.rare - c.exact - c.address))


def summarize_counts(path):
    cols = {k: [] for k in COUNT_COLUMNS}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            vals = line.rstrip("\n").split("\t")[1:]
            for k, v in zip(COUNT_COLUMNS, vals):
                cols[k].append(int(v))
    n = len(cols["union"])
    out = {"s1_rows": n}
    for k in ("exact", "address", "rare", "union"):
        v = cols[k]
        out[k] = {"pairs": sum(v), "mean": sum(v) / n if n else 0, "median": statistics.median(v) if n else 0,
                  "max": max(v) if n else 0, "zero_pct": (sum(1 for x in v if x == 0) / n * 100) if n else 0}
    for k in ("exact_only", "address_only", "rare_only"):
        out[k] = sum(cols[k])
    out["channel_overlap_pairs"] = out["exact"]["pairs"] + out["address"]["pairs"] + out["rare"]["pairs"] - out["union"]["pairs"]
    return out


def run(args, log=print, _stop_after_chunks=None):
    t_start = time.time()
    if os.path.basename(args.out) == "candidate_pairs.tsv" and not args.allow_final:
        raise SystemExit("refusing to write candidate_pairs.tsv without --allow-final")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    partial = args.out + ".partial"
    ckpt_path = args.out + ".checkpoint.json"
    counts_path = args.out.rsplit(".", 1)[0] + ".channel_counts.tsv"
    diag_path = args.out.rsplit(".", 1)[0] + ".channels.tsv" if args.channel_diagnostics else None

    stopwords = load_address_stopwords(args.stopwords)
    cfg = LOCKED_CONFIG
    log(f"config: {cfg}; stopwords: {len(stopwords)} from {args.stopwords}")

    def s1_rows():
        return iter_source1_selection(args.source1, sample_size=args.sample_size, limit=args.limit)

    t0 = time.time()
    vocab = collect_s1_vocabulary(s1_rows(), stopwords)
    t_vocab = time.time() - t0
    log(f"pass 1 (S1 vocabulary): {vocab.n_rows:,} S1 rows, {len(vocab.names):,} names, "
        f"{len(vocab.address_tokens):,} address tokens, {len(vocab.name_tokens):,} name tokens ({t_vocab:.0f}s)")

    t0 = time.time()
    idx = build_target_index([args.source2, args.source3], vocab, stopwords, cfg, log=log)
    del vocab
    t_index = time.time() - t0
    log(f"pass 2 (S2/S3 index): {len(idx.ids):,} target rows ({t_index:.0f}s); peak memory so far "
        f"{peak_memory_mb():,.0f} MB")

    rows_done = 0
    if args.resume and os.path.exists(ckpt_path) and os.path.exists(partial):
        with open(ckpt_path) as f:
            ck = json.load(f)
        rows_done = ck["rows_done"]
        for path, size in ((partial, ck["out_bytes"]), (counts_path, ck["counts_bytes"]),
                           (diag_path, ck.get("diag_bytes"))):
            if path:
                with open(path, "r+b") as fh:
                    fh.truncate(size)
        log(f"resuming after {rows_done:,} S1 rows")
        mode = "a"
    else:
        mode = "w"

    t0 = time.time()
    f_out = open(partial, mode, encoding="utf-8", newline="\n")
    f_cnt = open(counts_path, mode, encoding="utf-8", newline="\n")
    f_diag = open(diag_path, mode, encoding="utf-8", newline="\n") if diag_path else None
    if mode == "w":
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        f_cnt.write("source1_entity_id\t" + "\t".join(COUNT_COLUMNS) + "\n")
        if f_diag:
            f_diag.write("source1_entity_id\texact\taddress\trare\n")

    def checkpoint():
        for fh in (f_out, f_cnt, f_diag):
            if fh:
                fh.flush()
                os.fsync(fh.fileno())
        with open(ckpt_path + ".tmp", "w") as f:
            json.dump({"rows_done": n, "out_bytes": os.path.getsize(partial),
                       "counts_bytes": os.path.getsize(counts_path),
                       "diag_bytes": os.path.getsize(diag_path) if diag_path else None}, f)
        os.replace(ckpt_path + ".tmp", ckpt_path)

    n = 0
    chunks = 0
    rows = s1_rows()
    for _ in range(rows_done):
        next(rows)
    n = rows_done
    for c in generate_candidates(rows, idx, stopwords, cfg):
        f_out.write(f"{c.entity_id}\t{','.join(c.union)}\n")
        f_cnt.write(c.entity_id + "\t" + "\t".join(map(str, channel_counts(c))) + "\n")
        if f_diag:
            f_diag.write(f"{c.entity_id}\t{','.join(sorted(c.exact))}\t{','.join(sorted(c.address))}\t"
                         f"{','.join(sorted(c.rare))}\n")
        n += 1
        if n % args.chunk_size == 0:
            checkpoint()
            chunks += 1
            log(f"  {n:,} S1 rows written ({time.time() - t0:.0f}s)")
            if _stop_after_chunks is not None and chunks >= _stop_after_chunks:
                for fh in (f_out, f_cnt, f_diag):
                    if fh:
                        fh.close()
                return None
    checkpoint()
    for fh in (f_out, f_cnt, f_diag):
        if fh:
            fh.close()
    t_query = time.time() - t0

    os.replace(partial, args.out)
    os.remove(ckpt_path)

    summary = summarize_counts(counts_path)
    summary.update({
        "config": cfg.__dict__, "stopwords": len(stopwords), "out": args.out,
        "out_bytes": os.path.getsize(args.out),
        "runtime_s": {"s1_vocabulary": t_vocab, "target_index": t_index, "candidate_generation": t_query,
                      "total": time.time() - t_start},
        "peak_memory_mb": peak_memory_mb(),
        "resumed_from_row": rows_done,
    })
    with open(args.out.rsplit(".", 1)[0] + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done: {summary['s1_rows']:,} S1 rows, {summary['union']['pairs']:,} union pairs, "
        f"{summary['runtime_s']['total']:.0f}s, peak {summary['peak_memory_mb']:,.0f} MB -> {args.out}")
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source1", required=True)
    p.add_argument("--source2", required=True)
    p.add_argument("--source3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stopwords", default=DEFAULT_STOPWORDS_PATH)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sample-size", type=int, default=None,
                   help="stride-sample this many S1 rows (same sampler as the bounded experiments)")
    g.add_argument("--limit", type=int, default=None, help="process only the first N S1 rows")
    p.add_argument("--chunk-size", type=int, default=50_000, help="S1 rows between checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--channel-diagnostics", action="store_true",
                   help="also write per-channel id lists (validation runs only; large at full scale)")
    p.add_argument("--allow-final", action="store_true", help="permit writing candidate_pairs.tsv")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

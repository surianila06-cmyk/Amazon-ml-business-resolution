"""
Initial read-only data inspection for the Amazon ML Challenge 2026
Business Entity Resolution problem (Role 2: Candidate Generation / Blocking).

This script ONLY inspects the train_source1/2/3.tsv files. It does not
modify, rewrite, or convert any original dataset files, and it does not
build candidate pairs, similarity scores, or a blocking index.

Memory approach:
    - Files are processed ONE AT A TIME (never all three loaded together).
    - Each file is read in a single streaming pass using csv.reader over
      a text-mode file handle (line-by-line, not read() into memory).
    - Per-row aggregates (counts, Counters) are O(small) in memory.
    - entity_id uniqueness is tracked with a set of IDs for the file
      currently being processed only (cleared before moving to next file).
    - business_name / business_address length distributions use bounded
      reservoir sampling (fixed-size sample) instead of storing all values.

Usage:
    python inspect_data.py
"""

import csv
import os
import sys
import random
import codecs
from collections import Counter
from pathlib import Path

# The Windows terminal's default stdout codec is often cp1252, which cannot
# represent many Unicode characters found in the (correctly UTF-8 encoded)
# data (e.g. Devanagari script). Reconfigure stdout to UTF-8 so this
# inspection script itself doesn't crash or silently mis-render output.
# This does NOT touch/convert any dataset file - it only affects how this
# script's own print() output is encoded to the console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

# <repo root>/dataset/train, derived from this script's location (candidate_generation/analysis/)
DATA_DIR = str(Path(__file__).resolve().parents[2] / "dataset" / "train")

FILES = [
    ("source1", "train_source1.tsv", "S1"),
    ("source2", "train_source2.tsv", "S2"),
    ("source3", "train_source3.tsv", "S3"),
]

EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
DELIMITER = "\t"
RESERVOIR_SIZE = 100_000
NULL_LIKE_STRINGS = {"null", "none", "nan", "n/a", "na", "-", "unknown", "?"}
RANDOM_SEED = 42

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def detect_delimiter(path):
    with open(path, "rb") as f:
        raw_first_line = f.readline()
    line_str = raw_first_line.decode("utf-8", errors="replace").rstrip("\r\n")
    candidates = ["\t", ",", ";", "|"]
    counts = {c: line_str.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best, counts, line_str


def check_bom(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(codecs.BOM_UTF8):
        return "UTF-8 BOM present"
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        return "UTF-16 BOM present"
    return "No BOM"


def check_utf8_validity(path):
    """Stream-decode the whole file as strict UTF-8 without loading it all
    into memory at once. Returns (is_valid, error_message_or_None, bytes_checked).
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    total = 0
    chunk_size = 4 * 1024 * 1024
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as e:
                    return False, f"{e} (near byte offset {total})", total
                return True, None, total
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as e:
                return False, f"{e} (near byte offset {total})", total
            total += len(chunk)


def find_mojibake_examples(path, max_examples=3):
    """Scan the first portion of the file (as valid UTF-8) for lines
    containing non-ASCII characters, and show what they'd look like if a
    terminal/tool had incorrectly decoded the same UTF-8 bytes as cp1252 -
    this reproduces the 'LÃ©arning' / 'à¤°à¤¾à¤®...' style garbling the user saw.
    """
    examples = []
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            if any(ord(ch) > 127 for ch in line):
                clean = line.rstrip("\r\n")
                utf8_bytes = clean.encode("utf-8")
                try:
                    mis_decoded_cp1252 = utf8_bytes.decode("cp1252", errors="replace")
                except Exception:
                    mis_decoded_cp1252 = "<cp1252 decode failed>"
                examples.append(
                    {
                        "correct_utf8_text": clean[:120],
                        "bytes_hex": utf8_bytes[:60].hex(),
                        "mis_decoded_as_cp1252": mis_decoded_cp1252[:120],
                    }
                )
            if len(examples) >= max_examples:
                break
            if i > 500_000:
                break
    return examples


def extract_prefix(entity_id):
    if "-" in entity_id:
        return entity_id.split("-", 1)[0]
    return entity_id[:2] if entity_id else ""


def is_unusual_country(value):
    if value == "":
        return False  # handled separately as "missing"
    v = value.strip()
    if v.lower() in NULL_LIKE_STRINGS:
        return True
    if any(ch.isdigit() for ch in v):
        return True
    if len(v) > 30:
        return True
    if v != v.title() and v.isupper() and len(v) > 3:
        # long all-caps strings that aren't standard 2-letter codes are worth flagging
        return True
    return False


def inspect_file(source_name, filename, expected_prefix):
    path = os.path.join(DATA_DIR, filename)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f"Inspecting {source_name}: {filename} ({file_size_mb:.1f} MB)")
    print("=" * 70)

    delim, delim_counts, raw_header_line = detect_delimiter(path)
    bom_status = check_bom(path)
    print(f"Detected delimiter: {repr(delim)}  (char counts in header: {delim_counts})")
    print(f"BOM check: {bom_status}")

    is_valid_utf8, utf8_error, bytes_checked = check_utf8_validity(path)
    print(f"Full-file strict UTF-8 decode: {'VALID' if is_valid_utf8 else 'INVALID'} "
          f"({bytes_checked:,} bytes checked)")
    if not is_valid_utf8:
        print(f"  -> UTF-8 decode error: {utf8_error}")

    mojibake_examples = find_mojibake_examples(path)
    if mojibake_examples:
        print("Sample non-ASCII lines (proving file bytes are correct UTF-8,"
              " and reproducing the mojibake if mis-decoded as cp1252):")
        for ex in mojibake_examples:
            print(f"  - correct : {ex['correct_utf8_text']}")
            print(f"    if read as cp1252: {ex['mis_decoded_as_cp1252']}")

    # ---- Single streaming pass: exact counts + Counters + reservoir sampling ----
    row_count = 0
    structural_mismatch_rows = 0
    header = None

    entity_id_seen = set()
    duplicate_id_count = 0

    missing_name = missing_address = missing_country = 0
    prefix_counter = Counter()
    country_counter = Counter()
    unusual_country_examples = []

    name_len_sum = 0
    addr_len_sum = 0
    name_len_min = None
    name_len_max = 0
    addr_len_min = None
    addr_len_max = 0

    rng = random.Random(RANDOM_SEED)
    name_len_reservoir = []
    addr_len_reservoir = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delim, quoting=csv.QUOTE_NONE)
        header = next(reader)
        header_ok = header == EXPECTED_COLUMNS
        col_index = {c: i for i, c in enumerate(header)}
        idx_id = col_index.get("entity_id", 0)
        idx_name = col_index.get("business_name", 1)
        idx_addr = col_index.get("business_address", 2)
        idx_country = col_index.get("country", 3)

        for row in reader:
            row_count += 1
            if len(row) != len(header):
                structural_mismatch_rows += 1
                continue

            eid = row[idx_id]
            name = row[idx_name]
            addr = row[idx_addr]
            country = row[idx_country]

            if eid in entity_id_seen:
                duplicate_id_count += 1
            else:
                entity_id_seen.add(eid)

            prefix_counter[extract_prefix(eid)] += 1

            if name == "":
                missing_name += 1
            if addr == "":
                missing_address += 1
            if country == "":
                missing_country += 1
            else:
                country_counter[country] += 1
                if is_unusual_country(country) and len(unusual_country_examples) < 15:
                    unusual_country_examples.append((eid, country))

            nlen = len(name)
            alen = len(addr)
            name_len_sum += nlen
            addr_len_sum += alen
            name_len_min = nlen if name_len_min is None else min(name_len_min, nlen)
            name_len_max = max(name_len_max, nlen)
            addr_len_min = alen if addr_len_min is None else min(addr_len_min, alen)
            addr_len_max = max(addr_len_max, alen)

            if len(name_len_reservoir) < RESERVOIR_SIZE:
                name_len_reservoir.append(nlen)
                addr_len_reservoir.append(alen)
            else:
                j = rng.randint(0, row_count - 1)
                if j < RESERVOIR_SIZE:
                    name_len_reservoir[j] = nlen
                    addr_len_reservoir[j] = alen

    def pct(sorted_vals, p):
        if not sorted_vals:
            return None
        k = int(len(sorted_vals) * p)
        k = min(k, len(sorted_vals) - 1)
        return sorted_vals[k]

    name_len_reservoir.sort()
    addr_len_reservoir.sort()

    print(f"\nHeader matches expected {EXPECTED_COLUMNS}: {header_ok}")
    print(f"Actual header: {header}")
    print(f"Row count (data rows, excluding header): {row_count:,}")
    print(f"Structural mismatch rows (column count != header count): {structural_mismatch_rows:,}")

    print(f"\nDuplicate entity_id occurrences (full-file exact check): {duplicate_id_count:,}")
    print(f"Unique entity_id count: {len(entity_id_seen):,}")

    print(f"\nID prefix breakdown (top 10): {prefix_counter.most_common(10)}")
    expected_count = prefix_counter.get(expected_prefix, 0)
    other_prefixes = row_count - expected_count
    print(f"Rows with expected prefix '{expected_prefix}-': {expected_count:,} "
          f"({expected_count / row_count * 100:.2f}%)")
    if other_prefixes:
        print(f"Rows WITHOUT expected prefix: {other_prefixes:,}")

    print(f"\nMissing business_name (empty string): {missing_name:,} "
          f"({missing_name / row_count * 100:.3f}%)")
    print(f"Missing business_address (empty string): {missing_address:,} "
          f"({missing_address / row_count * 100:.3f}%)")
    print(f"Missing country (empty string): {missing_country:,} "
          f"({missing_country / row_count * 100:.3f}%)")

    print(f"\nDistinct non-empty country values: {len(country_counter):,}")
    print(f"Top 10 country values: {country_counter.most_common(10)}")
    if unusual_country_examples:
        print(f"Sample unusual country values (entity_id, country) [{len(unusual_country_examples)} shown]:")
        for eid, c in unusual_country_examples:
            print(f"  - {eid}: {c!r}")

    print(f"\nbusiness_name length (chars) - exact min/mean/max: "
          f"{name_len_min} / {name_len_sum / row_count:.1f} / {name_len_max}")
    print(f"business_name length percentiles (reservoir n={len(name_len_reservoir):,}): "
          f"p25={pct(name_len_reservoir, .25)} p50={pct(name_len_reservoir, .50)} "
          f"p75={pct(name_len_reservoir, .75)} p95={pct(name_len_reservoir, .95)} "
          f"p99={pct(name_len_reservoir, .99)}")

    print(f"\nbusiness_address length (chars) - exact min/mean/max: "
          f"{addr_len_min} / {addr_len_sum / row_count:.1f} / {addr_len_max}")
    print(f"business_address length percentiles (reservoir n={len(addr_len_reservoir):,}): "
          f"p25={pct(addr_len_reservoir, .25)} p50={pct(addr_len_reservoir, .50)} "
          f"p75={pct(addr_len_reservoir, .75)} p95={pct(addr_len_reservoir, .95)} "
          f"p99={pct(addr_len_reservoir, .99)}")

    return {
        "source_name": source_name,
        "row_count": row_count,
        "header": header,
        "header_ok": header_ok,
        "structural_mismatch_rows": structural_mismatch_rows,
        "duplicate_id_count": duplicate_id_count,
        "unique_id_count": len(entity_id_seen),
        "prefix_counter": prefix_counter,
        "missing_name": missing_name,
        "missing_address": missing_address,
        "missing_country": missing_country,
        "distinct_countries": len(country_counter),
    }


def main():
    print("Amazon ML Challenge 2026 - Business Entity Resolution")
    print("Role 2 (Candidate Generation / Blocking) - INITIAL DATA INSPECTION ONLY")
    print(f"Data directory: {DATA_DIR}")
    print("NOTE: no files are being modified; this is a read-only pass.\n")

    results = []
    for source_name, filename, expected_prefix in FILES:
        result = inspect_file(source_name, filename, expected_prefix)
        results.append(result)

    print(f"\n{'=' * 70}")
    print("CROSS-SOURCE SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['source_name']}: rows={r['row_count']:,}, header_ok={r['header_ok']}, "
              f"structural_mismatches={r['structural_mismatch_rows']:,}, "
              f"duplicate_ids={r['duplicate_id_count']:,}, "
              f"distinct_countries={r['distinct_countries']:,}")

    headers = [tuple(r["header"]) for r in results]
    print(f"\nAll headers identical across sources: {len(set(headers)) == 1}")


if __name__ == "__main__":
    main()

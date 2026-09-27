"""
ML Challenge 2026 -- Role 2: Candidate Generation
STAGE 2 (independent experiment): address-based blocking.

Retrieves plausible Source-2/Source-3 candidates for each Source-1 entity
using ONLY normalized business_address tokens -- no business_name, no
country filter, no ground truth. This module is intentionally independent
of exact_name_blocking.py (Stage 1) so the standalone contribution of
address information can be measured; it is never combined with Stage 1 here.

Blocking key design (see candidate_generation/analysis/inspect_address_tokens.py
and its output address_token_inspection_output.txt for the empirical basis):
a stride-15 sample of 688,016 Source-2/3 addresses (22,862 empty/null, 3.32%)
found:
  - The top 40 WORD tokens by document frequency were dominated by
    street-type abbreviations our own normalize_business_address produces
    ('rd' 19.5%, 'no' 17.2%, 'st' 13.6%, 'dr' 10.2%, 'ave' 8.5%, 'fl' 6.4%,
    'ln' 4.8%), directionals ('new','north','west','south','near'), and
    common city/state names ('delhi' 5.3%, 'mumbai' 3.7%, 'maharashtra' 3.5%,
    'texas' 2.9%). All 40 exceeded ~1.9% document frequency -- at full
    (10.3M-row) scale that is well over 100k postings each, too common and
    too low-signal to use as blocking keys.
  - Single-character word tokens (a, b, c, h, o) were also very common
    (2.0-3.5%) -- unit/block-letter fragments, not distinctive on their own.
  - A literal token 'null' appeared as embedded TEXT inside 2.66% of
    addresses -- a data artifact (not a missing-value field, the literal
    string "null" concatenated into the address), so it is also excluded.
  - NUMERIC tokens: 1-digit and 2-digit tokens occurred 185,626 and 221,589
    times respectively (very common -- floor/unit numbers), while tokens
    with 3+ digits were comparatively rare and far more likely to be a
    genuine house number, PIN/ZIP code, or highway number.

Resulting rule: a token is a usable blocking key only if it is (a) numeric
with >= MIN_NUMERIC_TOKEN_DIGITS digits, or (b) a word, length >= 2, not in
ADDRESS_WORD_STOPWORDS. A posting-list cap is kept as a safety net for any
token that slips past these filters.

CANDIDATE SELECTION IS AND, NOT OR (MIN_SHARED_TOKENS = 2):
A first version of this module unioned postings across ALL of a query's
qualifying tokens (OR semantics). At full (10.3M-row) scale this blew up:
tokens outside the empirically-derived top-40 stopword list (e.g. city-name
fragments like 'high'/'point' from "High Point, NC") were still common
enough that a 4-token address could union to thousands of low-signal
candidates per query, producing a 37.7GB candidate file (vs. 330MB for
Stage 1) and exhausting system memory. Requiring a candidate to share
>= MIN_SHARED_TOKENS tokens with the query (AND semantics) is far more
conservative: a genuine address match should agree on more than one
fragment (e.g. house number AND street name), and this cuts candidate-set
size by orders of magnitude. The trade-off, by design: an S1 entity whose
address yields fewer than 2 qualifying tokens can never get an address
candidate from this stage (see `generate_address_candidates`).

Memory approach: same as exact_name_blocking.py -- each source file is
streamed once; the two resulting index dicts (S2, S3) are the only large
resident structures, by design.
"""

import csv
import os
import re
import sys
from collections import Counter, defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preprocessing import normalize_business_address  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

_TOKEN_SPLIT_RE = re.compile(r"[,\s]+")

ADDRESS_WORD_STOPWORDS = frozenset({
    "rd", "no", "st", "dr", "ave", "new", "city", "fl", "nagar", "delhi",
    "ln", "north", "west", "mumbai", "maharashtra", "plot", "mh", "tx",
    "texas", "null", "pradesh", "bangalore", "tn", "ny", "york", "door",
    "carolina", "nc", "unit", "महाराष्ट्र", "south", "near", "dl", "park", "oh",
})

MIN_NUMERIC_TOKEN_DIGITS = 3
DEFAULT_MAX_POSTING_SIZE = 5000
DEFAULT_MIN_SHARED_TOKENS = 2


def address_tokens(normalized_address):
    """Split an already-normalized address into raw tokens on whitespace/
    comma. Hyphenated tokens (e.g. 'a-212') are kept whole -- the hyphen is
    meaningful in house/plot numbers."""
    if not normalized_address:
        return []
    return [t for t in _TOKEN_SPLIT_RE.split(normalized_address) if t]


def is_keepable_address_token(token):
    """True if `token` is informative enough to use as an address blocking
    key -- see module docstring for the empirical justification."""
    if any(ch.isdigit() for ch in token):
        digit_count = sum(ch.isdigit() for ch in token)
        return digit_count >= MIN_NUMERIC_TOKEN_DIGITS
    return len(token) >= 2 and token not in ADDRESS_WORD_STOPWORDS


def address_blocking_keys(normalized_address):
    """The subset of `address_tokens(normalized_address)` usable as blocking
    keys. Empty/missing addresses yield an empty list (no candidates)."""
    return [t for t in address_tokens(normalized_address) if is_keepable_address_token(t)]


def build_address_token_index(path, max_posting_size=DEFAULT_MAX_POSTING_SIZE,
                               progress_every=1_000_000, log=print):
    """Stream a source-style TSV and return
    {address_blocking_token: [entity_id, ...]}, capping any token's posting
    list at `max_posting_size` (first-seen ids kept, as in Stage 1)."""
    index = defaultdict(list)
    frozen_tokens = set()
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_addr = header.index("business_address")

        for row in reader:
            norm = normalize_business_address(row[idx_addr])
            for tok in address_blocking_keys(norm):
                if tok in frozen_tokens:
                    continue
                lst = index.setdefault(tok, [])
                lst.append(row[idx_id])
                if len(lst) >= max_posting_size:
                    frozen_tokens.add(tok)

            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  indexed {n:,} rows from {os.path.basename(path)}...")

    log(f"  done indexing {os.path.basename(path)}: {n:,} rows, "
        f"{len(index):,} distinct address blocking tokens "
        f"({len(frozen_tokens):,} capped at {max_posting_size:,} postings).")
    return dict(index)


def generate_address_candidates(source1_path, s2_index, s3_index,
                                 min_shared_tokens=DEFAULT_MIN_SHARED_TOKENS,
                                 progress_every=200_000, log=print):
    """Stream `source1_path`; for each row yield
    (entity_id, s2_candidate_ids, s3_candidate_ids).

    A candidate is included only if it shares >= `min_shared_tokens` of the
    query's address blocking tokens (AND semantics, not a union) -- see the
    module docstring for why this replaced a first, union-based version that
    produced unbounded candidate sets at full scale. No business_name, no
    country filter, no ground truth used.
    """
    with open(source1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_addr = header.index("business_address")

        n = 0
        for row in reader:
            eid = row[idx_id]
            norm = normalize_business_address(row[idx_addr])
            keys = address_blocking_keys(norm)

            s2_counts = Counter()
            s3_counts = Counter()
            for tok in keys:
                for cid in s2_index.get(tok, ()):
                    s2_counts[cid] += 1
                for cid in s3_index.get(tok, ()):
                    s3_counts[cid] += 1

            s2_cand = {cid for cid, cnt in s2_counts.items() if cnt >= min_shared_tokens}
            s3_cand = {cid for cid, cnt in s3_counts.items() if cnt >= min_shared_tokens}

            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  generated candidates for {n:,} S1 entities...")

            yield eid, s2_cand, s3_cand

        log(f"  done: generated candidates for {n:,} S1 entities.")

"""
ML Challenge 2026 -- Role 2: Candidate Generation / Blocking
PRODUCTION candidate pipeline (locked configuration).

Final candidate set for each Source-1 entity = UNION of three channels:

  1. EXACT NAME: every S2/S3 row whose normalize_business_name() equals the
     S1's (empty names never match).
  2. ADDRESS: qualifying tokens = address_blocking.address_blocking_keys()
     minus the 161 frozen stopwords (config/address_stopwords_161.txt), as a
     SET of DISTINCT tokens per address. Per-source inverted index (S2 and
     S3 separately), each token's posting list holding the first
     `posting_cap` (5,000) rows in file order. A target is retrieved when at
     least `min_shared_tokens` (2) DISTINCT query tokens hit it through the
     capped postings (each row is posted at most once per token, and each
     query token is counted once). Retrieved targets are ranked by token
     Jaccard |Q & C| / |Q | C| over the full distinct token sets, then by
     |Q & C| descending, then entity_id ascending; the first `top_k` (200)
     are kept.
  3. RARE NAME TOKEN: name tokens = normalize_business_name() split on
     whitespace, length >= 2. DF = number of S2+S3 rows (combined) whose
     name contains the token. The S1's single rarest token with
     rare_min_df <= DF <= rare_max_df (1..200; tie: token string) retrieves
     every S2/S3 row containing it whose country equals the S1 country
     (strip + lowercase). An S1 with an empty country gets no rare-name
     candidates. Country is used ONLY in this channel.

Candidate identity is the entity_id string (S2-/S3- prefixes preserved);
the union is a set of entity_ids, written sorted, so output is
deterministic. Ground truth is never read here.

Memory design (for ~10.3M target rows): source files are streamed row by
row with csv.reader. Indexes are built only for keys the selected Source-1
rows can query (their normalized names, address tokens and name tokens),
which is exact: posting caps and DFs are still computed over every target
row. Targets are referenced by integer row numbers in `array` postings; one
list maps row number -> entity_id.

All normalization comes from utils.preprocessing (shared contract); address
token filtering comes from address_blocking.py. Neither is modified.
"""

import csv
import heapq
import os
import sys
from array import array
from collections import Counter
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from candidate_generation.address_blocking import address_blocking_keys  # noqa: E402
from utils.preprocessing import normalize_business_address, normalize_business_name  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

DEFAULT_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "config", "address_stopwords_161.txt")
SOURCE_FIELDS = ("entity_id", "business_name", "business_address", "country")


@dataclass(frozen=True)
class PipelineConfig:
    min_shared_tokens: int = 2
    posting_cap: int = 5000
    top_k: int = 200
    rare_min_df: int = 1
    rare_max_df: int = 200


LOCKED_CONFIG = PipelineConfig()


# --------------------------------------------------------------------------
# tokenization (thin wrappers over the shared preprocessing)
# --------------------------------------------------------------------------
def load_address_stopwords(path=DEFAULT_STOPWORDS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip() and not line.startswith("#"))


def address_query_tokens(normalized_address, stopwords):
    """Distinct qualifying address tokens of an already-normalized address."""
    return frozenset(t for t in address_blocking_keys(normalized_address) if t not in stopwords)


def name_tokens(normalized_name):
    """Distinct name tokens (length >= 2) of an already-normalized name."""
    return frozenset(t for t in normalized_name.split() if len(t) >= 2) if normalized_name else frozenset()


def country_key(country):
    return (country or "").strip().lower()


def iter_source_rows(path):
    """Yield (entity_id, business_name, business_address, country) per row."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx = [header.index(c) for c in SOURCE_FIELDS]
        for row in reader:
            yield tuple(row[i] for i in idx)


def count_data_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        return sum(1 for _ in f)


def iter_source1_selection(path, sample_size=None, limit=None):
    """Source-1 rows to process. sample_size -> stride sampling identical to
    the bounded experiments (stride = total // sample_size, rows with
    index % stride == 0, first sample_size of them); limit -> first N rows;
    neither -> every row."""
    stride = max(1, count_data_rows(path) // sample_size) if sample_size else 1
    cap = sample_size or limit
    n = 0
    for i, row in enumerate(iter_source_rows(path)):
        if i % stride:
            continue
        yield row
        n += 1
        if cap and n >= cap:
            break


# --------------------------------------------------------------------------
# pass 1: Source-1 vocabulary (what the selected S1 rows can query)
# --------------------------------------------------------------------------
@dataclass
class S1Vocabulary:
    names: set
    address_tokens: set
    name_tokens: set
    n_rows: int


def collect_s1_vocabulary(s1_rows, stopwords):
    names, addr, ntok = set(), set(), set()
    n = 0
    for _, name, address, _ in s1_rows:
        nn = normalize_business_name(name)
        if nn:
            names.add(nn)
            ntok.update(name_tokens(nn))
        addr.update(address_query_tokens(normalize_business_address(address), stopwords))
        n += 1
    return S1Vocabulary(names, addr, ntok, n)


# --------------------------------------------------------------------------
# pass 2: target index over S2 then S3
# --------------------------------------------------------------------------
class TargetIndex:
    def __init__(self, n_sources):
        self.ids = []                                  # row -> entity_id
        self.country = array("H")                      # row -> country code
        self.country_codes = {}                        # country_key -> code
        self.exact = {}                                # normalized name -> array(rows)
        self.addr_postings = [dict() for _ in range(n_sources)]  # per source: token -> array(rows), capped
        self.addr_token_ids = {}                       # S1-vocab address token -> int id
        self.addr_token_count = array("H")             # row -> |distinct qualifying tokens|
        self.fwd_offsets = array("q", [0])             # row -> slice into fwd_tokens
        self.fwd_tokens = array("i")                   # S1-vocab token ids per row
        self.name_df = {}                              # S1-vocab name token -> combined DF
        self.name_postings = {}                        # token -> array(rows) while DF <= rare_max_df

    def _country_code(self, key):
        code = self.country_codes.get(key)
        if code is None:
            code = self.country_codes[key] = len(self.country_codes)
        return code


def build_target_index(target_paths, vocab, stopwords, cfg=LOCKED_CONFIG, log=print, progress_every=1_000_000):
    idx = TargetIndex(len(target_paths))
    for src, path in enumerate(target_paths):
        postings = idx.addr_postings[src]
        n = 0
        for eid, name, address, country in iter_source_rows(path):
            r = len(idx.ids)
            idx.ids.append(eid)
            idx.country.append(idx._country_code(country_key(country)))

            nn = normalize_business_name(name)
            if nn and nn in vocab.names:
                rows = idx.exact.get(nn)
                if rows is None:
                    rows = idx.exact[nn] = array("i")
                rows.append(r)

            q = address_query_tokens(normalize_business_address(address), stopwords)
            idx.addr_token_count.append(min(len(q), 65535))
            for tok in q:
                if tok not in vocab.address_tokens:
                    continue
                tid = idx.addr_token_ids.get(tok)
                if tid is None:
                    tid = idx.addr_token_ids[tok] = len(idx.addr_token_ids)
                idx.fwd_tokens.append(tid)
                post = postings.get(tok)
                if post is None:
                    post = postings[tok] = array("i")
                if len(post) < cfg.posting_cap:
                    post.append(r)
            idx.fwd_offsets.append(len(idx.fwd_tokens))

            for tok in name_tokens(nn) & vocab.name_tokens:
                d = idx.name_df.get(tok, 0) + 1
                idx.name_df[tok] = d
                if d <= cfg.rare_max_df:
                    rows = idx.name_postings.get(tok)
                    if rows is None:
                        rows = idx.name_postings[tok] = array("i")
                    rows.append(r)
                elif d == cfg.rare_max_df + 1:
                    del idx.name_postings[tok]

            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  indexed {n:,} rows from {os.path.basename(path)}")
        log(f"  done {os.path.basename(path)}: {n:,} rows")
    return idx


# --------------------------------------------------------------------------
# pass 3: per-S1 channels and union
# --------------------------------------------------------------------------
@dataclass
class S1Candidates:
    entity_id: str
    exact: frozenset
    address: frozenset
    rare: frozenset

    @property
    def union(self):
        """The final candidate list passed to Role 3: sorted entity_ids."""
        return sorted(self.exact | self.address | self.rare)


def exact_channel(idx, normalized_name):
    if not normalized_name:
        return frozenset()
    return frozenset(idx.ids[r] for r in idx.exact.get(normalized_name, ()))


def address_channel(idx, query_tokens, cfg=LOCKED_CONFIG):
    if len(query_tokens) < cfg.min_shared_tokens:
        return frozenset()
    hits = Counter()
    for postings in idx.addr_postings:
        for tok in query_tokens:
            hits.update(postings.get(tok, ()))
    retrieved = [r for r, c in hits.items() if c >= cfg.min_shared_tokens]
    if not retrieved:
        return frozenset()
    qids = {idx.addr_token_ids[t] for t in query_tokens if t in idx.addr_token_ids}
    nq = len(query_tokens)
    ranked = []
    for r in retrieved:
        shared = sum(1 for t in idx.fwd_tokens[idx.fwd_offsets[r]:idx.fwd_offsets[r + 1]] if t in qids)
        jaccard = shared / (nq + idx.addr_token_count[r] - shared)
        ranked.append((-jaccard, -shared, idx.ids[r]))
    return frozenset(eid for _, _, eid in heapq.nsmallest(cfg.top_k, ranked))


def select_rare_token(idx, tokens, cfg=LOCKED_CONFIG):
    eligible = [(idx.name_df[t], t) for t in tokens if cfg.rare_min_df <= idx.name_df.get(t, 0) <= cfg.rare_max_df]
    return min(eligible) if eligible else None


def rare_name_channel(idx, normalized_name, s1_country, cfg=LOCKED_CONFIG):
    ck = country_key(s1_country)
    code = idx.country_codes.get(ck)
    if not ck or code is None:
        return frozenset()
    sel = select_rare_token(idx, name_tokens(normalized_name), cfg)
    if sel is None:
        return frozenset()
    return frozenset(idx.ids[r] for r in idx.name_postings[sel[1]] if idx.country[r] == code)


def generate_candidates(s1_rows, idx, stopwords, cfg=LOCKED_CONFIG):
    """Yield S1Candidates for every selected S1 row, in input order."""
    for eid, name, address, country in s1_rows:
        nn = normalize_business_name(name)
        q = address_query_tokens(normalize_business_address(address), stopwords)
        yield S1Candidates(
            entity_id=eid,
            exact=exact_channel(idx, nn),
            address=address_channel(idx, q, cfg),
            rare=rare_name_channel(idx, nn, country, cfg),
        )

"""
ML Challenge 2026 -- Role 2: Candidate Generation / Blocking

Reusable, stdlib-only blocking library. Given Source-2/3 records, builds an
inverted index from normalized business-name tokens to entity IDs, then for
each Source-1 query returns a bounded candidate set by unioning the postings
of that query's RAREST tokens (rarest-token blocking).

Why rarest-token blocking (not "union of all tokens"):
  Business names share very common words even after removing legal-suffix
  stopwords (e.g. "retail", "services", "trading"). Unioning postings for
  *every* token in a name would pull in huge, mostly-irrelevant candidate
  sets and make the pairwise cost roughly (num_S1 * avg_candidates)
  unmanageable at this scale (2.2M x ~10.3M records). Restricting each query
  to its `max_query_tokens` least-frequent tokens keeps candidate sets small
  while still requiring the pair to share a distinctive word -- standard
  practice for token/canopy blocking in entity resolution.

Why a posting-list cap (`max_posting_size`):
  A handful of tokens (generic words that survive the stopword filter) can
  still appear in tens of thousands of names. Capping how many IDs a single
  token's posting list retains bounds both index memory and worst-case
  candidate-set size. IDs are kept in first-seen (file) order; once a
  token's list reaches the cap, further records with that token are simply
  not added to *that token's* list (they remain reachable via their other,
  presumably rarer, tokens).

Memory approach:
  - Source files are streamed one at a time (csv.reader over a text handle),
    never read() into memory.
  - The index itself (token -> list[entity_id]) is the only large resident
    structure, by design -- blocking fundamentally requires an in-memory
    lookup structure over the target records. Its size is bounded by
    `max_posting_size` and by dropping non-informative tokens.
"""

import csv
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.preprocessing import normalize_business_name  # noqa: E402

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

# Tokens dropped from blocking keys: too generic to be distinctive (legal
# suffixes our own normalization introduces, plus a couple of ultra-common
# connector words). NOT removed from the normalized text used elsewhere --
# this list only affects which tokens participate as blocking keys.
GENERIC_NAME_STOPWORDS = {
    "ltd", "inc", "llc", "llp", "pvt", "corp", "co", "and", "the", "of",
}

DEFAULT_MAX_POSTING_SIZE = 5000
DEFAULT_MAX_QUERY_TOKENS = 3


def name_blocking_tokens(normalized_name):
    """Split an already-normalized business name into candidate blocking
    tokens: whitespace-split words, length >= 2, minus generic stopwords."""
    if not normalized_name:
        return []
    return [
        tok for tok in normalized_name.split(" ")
        if len(tok) >= 2 and tok not in GENERIC_NAME_STOPWORDS
    ]


def build_name_token_index(sources, max_posting_size=DEFAULT_MAX_POSTING_SIZE,
                            progress_every=1_000_000, log=print):
    """Build an inverted index token -> [entity_id, ...] by streaming each
    (label, path) in `sources` once.

    Also returns an `id_country` dict (entity_id -> country) so callers can
    apply a country pre-filter without a second pass over the source files.

    Returns: (index, id_country, stats) where stats is a Counter of
    per-source row counts processed.
    """
    index = {}
    frozen_tokens = set()
    id_country = {}
    stats = Counter()

    for label, path in sources:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            idx_id = header.index("entity_id")
            idx_name = header.index("business_name")
            idx_country = header.index("country")

            for row in reader:
                eid = row[idx_id]
                id_country[eid] = row[idx_country]

                norm = normalize_business_name(row[idx_name])
                for tok in name_blocking_tokens(norm):
                    if tok in frozen_tokens:
                        continue
                    lst = index.setdefault(tok, [])
                    lst.append(eid)
                    if len(lst) >= max_posting_size:
                        frozen_tokens.add(tok)

                stats[label] += 1
                if progress_every and stats[label] % progress_every == 0:
                    log(f"  [{label}] indexed {stats[label]:,} rows...")

        log(f"  [{label}] done: {stats[label]:,} rows indexed.")

    log(f"Index built: {len(index):,} distinct blocking tokens "
        f"({len(frozen_tokens):,} capped at {max_posting_size:,} postings), "
        f"{len(id_country):,} entities.")
    return index, id_country, stats


def candidates_for_name(name, index, max_query_tokens=DEFAULT_MAX_QUERY_TOKENS):
    """Return the candidate entity_id set for a (raw) business name: the
    union of postings for that name's `max_query_tokens` rarest blocking
    tokens (by current posting-list length in `index`)."""
    norm = normalize_business_name(name)
    tokens = name_blocking_tokens(norm)
    if not tokens:
        return set()

    tokens_sorted = sorted(tokens, key=lambda t: (len(index.get(t, ())), t))
    chosen = tokens_sorted[:max_query_tokens]

    candidates = set()
    for tok in chosen:
        candidates.update(index.get(tok, ()))
    return candidates


def generate_candidates_for_source1(source1_path, index, id_country,
                                     max_query_tokens=DEFAULT_MAX_QUERY_TOKENS,
                                     country_filter=False,
                                     progress_every=200_000, log=print):
    """Stream `source1_path`, yielding (source1_entity_id, candidate_id_set)
    for every row. `country_filter`: if True, drop candidates whose stored
    country differs from the query's own country (only meaningful once
    verified safe -- see the empirical country-agreement check in the
    project notes)."""
    with open(source1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx_id = header.index("entity_id")
        idx_name = header.index("business_name")
        idx_country = header.index("country")

        n = 0
        for row in reader:
            eid = row[idx_id]
            name = row[idx_name]
            country = row[idx_country]

            candidates = candidates_for_name(name, index, max_query_tokens)
            if country_filter and country:
                candidates = {c for c in candidates if id_country.get(c) == country}

            n += 1
            if progress_every and n % progress_every == 0:
                log(f"  generated candidates for {n:,} S1 entities...")

            yield eid, candidates

        log(f"  done: generated candidates for {n:,} S1 entities.")


def write_candidate_pairs_tsv(pairs_iter, out_path):
    """Write (source1_entity_id, candidate_id_set) pairs to a TSV matching
    the project's candidate_pairs.tsv contract: header
    'source1_entity_id\\tcandidate_entity_ids', comma-separated IDs, empty
    when there are no candidates."""
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, candidates in pairs_iter:
            f.write(f"{eid}\t{','.join(sorted(candidates))}\n")

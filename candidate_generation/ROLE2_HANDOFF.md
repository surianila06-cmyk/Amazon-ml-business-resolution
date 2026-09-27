# Role 2 Handoff — Candidate Generation / Blocking

Status: training candidate set generated and verified. Test candidates and
`candidate_pairs.tsv` have **not** been generated.

## 1. Objective

For every Source-1 (S1) entity, produce a small set of candidate Source-2 (S2)
and Source-3 (S3) entity IDs that is likely to contain its true matches. Role 3
scores these candidates and makes the final match decision. Role 2 does not
decide matches; it only narrows the search space and aims for high recall at a
manageable volume.

## 2. Dataset sizes (data rows, excluding header)

| file | rows |
|---|---|
| `dataset/train/train_source1.tsv` | 2,206,821 |
| `dataset/train/train_source2.tsv` | 5,034,616 |
| `dataset/train/train_source3.tsv` | 5,285,603 |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 |
| `dataset/test/test_source1.tsv` | 1,732,544 (line count) |
| `dataset/test/test_source2.tsv` | 4,887,273 (line count) |
| `dataset/test/test_source3.tsv` | 5,082,316 (line count) |

Source columns: `entity_id, business_name, business_address, country`.

Train ground truth: 2,083,574 S1 have at least one true match, and there are
7,638,365 true pairs.

## 3. Shared preprocessing contract

All normalization comes from `utils/preprocessing.py`, which Role 2 does not
modify:

- `normalize_business_name(name)`: NFKC, lowercase, whitespace/quote cleanup,
  conservative legal-suffix normalization (e.g. Private Limited → `pvt ltd`,
  LLC variants → `llc`), `&` → `and`, and removal of other punctuation
  (apostrophes and hyphens are kept). Non-Latin scripts are preserved.
- `normalize_business_address(address)`: NFKC, lowercase, whitespace cleanup,
  and unambiguous street-type abbreviations only (street → `st`, road → `rd`,
  etc.). Commas, apostrophes and hyphens are kept. All digits are preserved.
- Null-like values (`""`, `null`, `none`, `nan`, `n/a`, ...) normalize to an
  empty string.

Country is compared only after `strip().lower()`.

## 4. Channel 1 — exact-name blocking

- Key: `normalize_business_name(business_name)`.
- Candidates: every S2/S3 row whose normalized name equals the S1's normalized
  name.
- An empty normalized name never matches.
- No country restriction and no cap.

## 5. Channel 2 — address blocking (locked)

**Tokens**

- `address_blocking.address_blocking_keys(normalize_business_address(addr))`
  splits the address on whitespace or commas. Hyphenated tokens are kept
  whole.
- A token is kept if either:
  - it contains digits and has at least 3 digits, or
  - it is a word of length ≥ 2.
- The 161 frozen stopwords in `candidate_generation/config/address_stopwords_161.txt`
  are then removed.
- Tokens are treated as a **set of distinct tokens** per address.

**Indexing**

- There is a separate inverted index per source (S2 and S3).
- Each token's posting list is capped at the **first 5,000 rows in file
  order**, per source.

**Retrieval**

- A target is retrieved when **≥ 2 distinct** query tokens hit it through the
  capped postings.

**Ranking**

1. Token Jaccard `|Q∩C| / |Q∪C|`, computed on the full distinct token sets,
   descending.
2. Shared-token count, descending.
3. `entity_id`, ascending.

**Output**

- The **top 200** targets per S1 are kept.

## 6. Channel 3 — rare-name-token blocking (locked)

- **Name tokens:** the normalized name split on whitespace, keeping tokens of
  length ≥ 2.
- **Document frequency (DF):** the number of S2+S3 rows (combined) whose name
  contains the token.
- **Token selection:** the S1's single rarest token with **1 ≤ DF ≤ 200**. Ties
  are broken by the token string.
- **Candidates:** every S2/S3 row containing that token **whose country equals
  the S1 country**.
- An S1 with an empty country gets no rare-name candidates.
- Country is used **only** in this channel.

## 7. Final architecture

```
candidates(S1) = exact-name  ∪  address top-200  ∪  rare-name
```

- Candidates are identified by `entity_id`, with the `S2-`/`S3-` prefixes
  preserved.
- Each row is written as a sorted, de-duplicated list.
- Ground truth is never read during generation; it is used only for scoring.
- **Code:**
  - `candidate_generation/candidate_pipeline.py`: the locked configuration
    `PipelineConfig(min_shared_tokens=2, posting_cap=5000, top_k=200, rare_min_df=1, rare_max_df=200)`.
  - `candidate_generation/run_candidate_pipeline.py`: the CLI, which is
    streaming, checkpointed and resumable.

## 8. Full training results

**Volume:**

| metric | exact-name | address | rare-name | **UNION** |
|---|---|---|---|---|
| candidate pairs | 23,402,481 | 187,742,011 | 53,757,182 | **260,807,896** |
| mean / S1 | 10.60 | 85.07 | 24.36 | **118.18** |
| median / S1 | 1 | 36 | 0 | **99** |
| max / S1 | 997 | 200 | 200 | **1,196** |
| zero-candidate % | 27.57% | 2.80% | 66.03% | **0.483%** (10,655 S1) |

**Recall** (`evaluate_candidates.py` against the train ground truth):

| scope | pair recall | entity coverage |
|---|---|---|
| S2 | 88.4046% (3,265,329 / 3,693,619) | 92.9013% (1,782,846 / 1,919,076) |
| S3 | 86.4503% (3,410,243 / 3,944,746) | 92.1535% (1,788,280 / 1,940,545) |
| **Combined** | **87.3953%** (6,675,572 / 7,638,365) | **97.4277%** (2,029,978 / 2,083,574) |

The candidate set is 99.998855% smaller than brute-force pairing
(2,206,821 × 10,320,219).

Per-channel recall was measured on the 20K and 100K samples only. The full run
was written without per-channel diagnostics.

## 9. Stability across scales

All runs use the same production code and configuration. The samples use
deterministic stride sampling of S1: stride 110 for 20K and stride 22 for 100K.
The 20K sample is a subset of the 100K sample, which is a subset of the full
set.

| metric | 20K | 100K | full (2,206,821) |
|---|---|---|---|
| union pairs | 2,331,992 | 11,816,508 | 260,807,896 |
| mean / median cand. per S1 | 116.60 / 95 | 118.17 / 99 | 118.18 / 99 |
| max cand. per S1 | 604 | 604 | 1,196 |
| zero-candidate % | 0.460% | 0.477% | 0.483% |
| union pair recall | 87.1805% | 87.3178% | 87.3953% |
| union entity coverage | 97.3468% | 97.4120% | 97.4277% |

Channel pair recall (20K → 100K):

| channel | 20K | 100K |
|---|---|---|
| exact-name | 22.65% | 22.69% |
| address | 77.96% | 77.88% |
| rare-name | 27.46% | 28.31% |

Every 20K and 100K output row is byte-identical to the corresponding row of the
full run.

## 10. Runtime and memory

These are single-process CPython runs on Windows with 16 GB RAM.

| stage | 20K | 100K | full |
|---|---|---|---|
| S1 vocabulary pass | 4 s | 13 s | 150 s |
| S2/S3 index pass | 987 s | 553 s | 704 s |
| candidate generation | 62 s | 270 s | 8,985 s |
| total | 1,053 s | 836 s | 9,844 s (2.73 h) |
| peak working set | 1,142 MB | 1,259 MB | 1,869 MB |
| output size | 30.3 MB | 153.6 MB | 3,389,897,080 B (3.39 GB) |

The full run's generation time includes two machine-suspension pauses of
about 18 and 31 minutes. Without those pauses, generation took about 6,150 s.
The full run's working set held at about 1.52 GB during generation, and system
free RAM stayed at 6.3 GB or more.

Checkpoints were written every 50,000 S1 rows. No resume was needed
(`resumed_from_row = 0`), and no `.partial` or checkpoint files remain.

## 11. Integrity checks (all passed on the full run)

- The header is `source1_entity_id\tcandidate_entity_ids`.
- There are 2,206,821 rows, in Source-1 file order. Every S1 appears exactly
  once, with none missing or extra.
- There are no duplicate candidate IDs within a row.
- Every candidate ID starts with `S2-` or `S3-`.
- Candidate IDs are sorted in every row (deterministic output).
- **Union check:** for every S1, the row length equals
  `|exact ∪ address ∪ rare|` as recorded by the pipeline
  (`train_candidate_pairs.channel_counts.tsv`), and it is consistent with each
  channel's size.
- **Set-level union check (100K pilot):** each pilot row was verified as
  `sorted(exact ∪ address ∪ rare)` using per-channel diagnostics. All 100,000
  pilot rows are byte-identical in the full output.
- The union pair count in `summary.json` equals an independent recount.
- No production or experiment Python file was modified. SHA-256 hashes were
  checked before and after the runs.

## 12. Training candidate file

```
candidate_generation/output/train_candidate_pairs.tsv
```

Companion files in the same folder:

- `train_candidate_pairs.channel_counts.tsv`: per-S1 channel sizes, no ground
  truth.
- `train_candidate_pairs.summary.json`
- `train_candidate_pairs.run_log.txt`
- `train_candidate_pairs_eval_log.txt`

## 13. Interface contract for Role 3

**Input file:** `candidate_generation/output/train_candidate_pairs.tsv`

- UTF-8, tab-separated, `\n` line endings.
- Header: `source1_entity_id\tcandidate_entity_ids`.
- One row per S1 entity (all 2,206,821). Rows with zero candidates are kept,
  with an empty second field.

**Candidate IDs**

- `candidate_entity_ids` is a comma-separated, sorted, de-duplicated list of
  `entity_id` values.
- Every ID is from Source 2 (`S2-…`) or Source 3 (`S3-…`), and they are mixed
  in one list.

**Rules for Role 3**

- The final matches for an S1 **must be a subset of that S1's candidates**.
  Role 3 must not add pairs outside the candidate set.
- Role 3 should join candidates to the source files by `entity_id` for
  features, and use `utils/preprocessing.py` for any normalized text.

## 14. Known limitation

Candidate pair recall on train is **87.3953%**. The remaining 962,793 true
pairs (12.6047%) are not in the candidate set, so **Role 3 cannot recover
them**. This recall is the upper bound on the pair recall of any downstream
matcher.

Entity coverage is 97.4277%, so 53,596 S1 entities with true matches have no
true match among their candidates.

## 15. External data

**No external data lookup was used.** Candidate generation reads only the
provided local train TSV files and the frozen stopword file. The code has no
network access.

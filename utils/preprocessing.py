#!/usr/bin/env python3
"""
ML Challenge 2026 -- Shared Preprocessing Contract

Stdlib-only text normalization shared by every role (candidate generation,
matching, evaluation) so all downstream code compares text the same way.

This module ONLY produces normalized string representations. It never
modifies the original ``business_name`` / ``business_address`` fields --
callers are expected to keep the raw values around and add the normalized
ones as new fields/columns alongside them.

Public functions:
    normalize_business_name(name)     -> str
    normalize_business_address(addr)  -> str
    base_clean(text)                  -> str   (shared helper, also public
                                                  since other roles may want
                                                  the generic cleanup step
                                                  without name/address-specific
                                                  rules)

Design notes (why these choices):
  * Unicode-aware, not ASCII-folding: this dataset mixes English/Latin-script
    names/addresses with Devanagari (Hindi) business names (seen during data
    inspection, e.g. source2 records like "राम मार्केटिंग प्राइवेट लिमिटेड").
    Transliterating or stripping non-Latin scripts would destroy information
    needed to match those records, so normalization only lowercases and
    reshapes punctuation/whitespace -- it never removes/replaces non-Latin
    characters. `str.lower()` and `unicodedata.normalize` are Unicode-aware
    and are no-ops on scripts without a case distinction (e.g. Devanagari).
  * NFKC normalization folds full-width/compatibility characters and
    combining sequences into a canonical form, which reduces spurious
    mismatches between visually-identical strings without touching which
    script is used.
  * Legal-suffix / abbreviation normalization is limited to a short, high
    confidence list of unambiguous English business/address terms (e.g.
    "Private Limited" -> "pvt ltd", "Street" -> "st"). Ambiguous
    abbreviations (e.g. "St" could mean "Saint" or "Street") are
    deliberately NOT mapped, to avoid silently conflating different real
    meanings -- conservative by design, per the team's preprocessing
    contract.
  * Numeric content (house numbers, PIN/ZIP codes, highway numbers, unit
    numbers) is never stripped or altered -- only surrounding punctuation
    and whitespace are normalized.
"""

import math
import re
import unicodedata

__all__ = ["base_clean", "normalize_business_name", "normalize_business_address"]

# ---------------------------------------------------------------------------
# Null-like value handling
# ---------------------------------------------------------------------------

_NULL_LIKE_STRINGS = {
    "", "null", "none", "nan", "n/a", "na", "-", "--", "unknown", "?", "missing",
}


def _is_null_like(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text.lower() in _NULL_LIKE_STRINGS


# ---------------------------------------------------------------------------
# Shared low-level cleanup
# ---------------------------------------------------------------------------

# Curly/typographic quote and dash variants -> plain ASCII equivalents, so
# "O'Brien" / "O’Brien" / "OʼBrien" all compare equal, and en/em
# dashes behave like a plain hyphen. This does NOT touch non-Latin scripts.
_PUNCT_TRANSLATION = str.maketrans(
    {
        "‘": "'", "’": "'", "‛": "'", "′": "'",
        "`": "'", "´": "'", "ʼ": "'",
        "“": '"', "”": '"', "„": '"', "″": '"',
        "–": "-", "—": "-", "−": "-",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_symbols(text, keep):
    """Replace characters that are not letters/digits/marks/whitespace/`keep`
    with a space.

    Deliberately NOT implemented with a `\\w`-based regex character class:
    Python's `re` module's `\\w` does not include Unicode *combining marks*
    (category Mn/Mc/Me), which scripts such as Devanagari rely on for vowel
    signs (matras). A `[^\\w\\s...]` class would silently strip those marks
    and corrupt non-Latin text -- exactly the kind of script damage this
    module must avoid. Iterating by Unicode category keeps letters (L*),
    numbers (N*) and marks (M*) intact regardless of script.
    """
    out = []
    for ch in text:
        if ch.isspace() or ch in keep:
            out.append(ch)
        elif unicodedata.category(ch)[0] in ("L", "N", "M"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def base_clean(text):
    """Generic, field-agnostic cleanup shared by name/address normalization.

    Steps: null-like -> "", Unicode NFKC normalization, quote/dash
    standardization, lowercasing, whitespace collapsing/trimming.
    Does not remove punctuation or apply any domain-specific abbreviation
    rules -- see ``normalize_business_name`` / ``normalize_business_address``
    for those.
    """
    if _is_null_like(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_TRANSLATION)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------------------------

# Ordered (longer/more specific phrases first) so e.g. "private limited"
# collapses in one step instead of first being caught by a generic
# "limited" -> "ltd" rule and left as "private ltd".
_NAME_SUFFIX_PATTERNS = [(re.compile(p), repl) for p, repl in [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\blimited liability company\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bl\.?l\.?c\.?\b", "llc"),
    (r"\bl\.?l\.?p\.?\b", "llp"),
    (r"\bincorporated\b", "inc"),
    (r"\binc\.?\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\.?\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\.?\b", "ltd"),
    (r"\bpvt\.?\b", "pvt"),
    (r"\bprivate\b", "pvt"),
    (r"\bco\.?\b", "co"),
]]

_AMPERSAND_RE = re.compile(r"\s*&\s*")

# Characters kept as-is (besides letters/digits/marks/whitespace, see
# `_strip_symbols`) when normalizing business names: apostrophes (meaningful,
# e.g. "mcdonald's") and hyphens (meaningful in hyphenated names).
_NAME_PUNCT_KEEP = {"'", "-"}


def normalize_business_name(name):
    """Normalize a business_name value for matching/blocking.

    - Unicode NFKC + lowercase + whitespace/quote cleanup (`base_clean`)
    - Conservative legal-suffix normalization (Inc./Incorporated -> inc,
      LLC variants -> llc, Private Limited -> pvt ltd, etc.)
    - "&" -> "and"
    - Remaining punctuation/symbols removed (apostrophes and hyphens kept)
    - Non-Latin scripts (e.g. Devanagari) are preserved as-is
    """
    text = base_clean(name)
    if not text:
        return ""

    for pattern, repl in _NAME_SUFFIX_PATTERNS:
        text = pattern.sub(repl, text)

    text = _AMPERSAND_RE.sub(" and ", text)
    text = _strip_symbols(text, _NAME_PUNCT_KEEP)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Business address normalization
# ---------------------------------------------------------------------------

# Only unambiguous, single-meaning address terms are mapped. Deliberately
# excluded: "st" (Street vs Saint), "ct" (Court vs Circuit / Connecticut),
# and directionals like N/S/E/W are left as full words to avoid colliding
# with unrelated single letters elsewhere in an address.
_ADDRESS_ABBR_PATTERNS = [(re.compile(p), repl) for p, repl in [
    (r"\bapartment\b", "apt"),
    (r"\bavenue\b", "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bbuilding\b", "bldg"),
    (r"\bdrive\b", "dr"),
    (r"\bfloor\b", "fl"),
    (r"\bhighway\b", "hwy"),
    (r"\blane\b", "ln"),
    (r"\bmountain\b", "mtn"),
    (r"\bparkway\b", "pkwy"),
    (r"\bplace\b", "pl"),
    (r"\broad\b", "rd"),
    (r"\bstreet\b", "st"),
    (r"\bsuite\b", "ste"),
    (r"\bterrace\b", "ter"),
]]

# Characters kept as-is (besides letters/digits/marks/whitespace, see
# `_strip_symbols`) when normalizing addresses: commas (meaningful component
# separators), apostrophes, and hyphens (meaningful in hyphenated house
# numbers like "A-212"). House numbers, PIN/ZIP codes and highway numbers
# are preserved automatically since digits are never stripped.
_ADDRESS_PUNCT_KEEP = {",", "'", "-"}
_COMMA_SPACING_RE = re.compile(r"\s*,\s*")


def normalize_business_address(address):
    """Normalize a business_address value for matching/blocking.

    - Unicode NFKC + lowercase + whitespace/quote cleanup (`base_clean`)
    - Conservative, unambiguous street-type abbreviation normalization
      (Street -> st, Avenue -> ave, Highway -> hwy, etc.)
    - Punctuation normalized; commas kept as component separators with
      consistent spacing ("a ,b" / "a,b" / "a , b" -> "a, b")
    - All numeric content (house numbers, PIN/ZIP codes, highway numbers,
      unit numbers) is preserved exactly, only surrounding punctuation and
      whitespace are normalized
    - Non-Latin scripts are preserved as-is
    """
    text = base_clean(address)
    if not text:
        return ""

    for pattern, repl in _ADDRESS_ABBR_PATTERNS:
        text = pattern.sub(repl, text)

    text = _strip_symbols(text, _ADDRESS_PUNCT_KEEP)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _COMMA_SPACING_RE.sub(", ", text)
    text = text.strip(", ").strip()
    return text


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

_NAME_TEST_CASES = [
    # (input, expected_output, note)
    ("Orelee's Barbershop", "orelee's barbershop", "English name, apostrophe preserved"),
    ("B+ Retail Inc.", "b retail inc", "punctuation ('+', '.') normalized, suffix kept"),
    ("Prime Money LLC", "prime money llc", "legal suffix already canonical"),
    ("XYZ Private Limited", "xyz pvt ltd", "legal suffix normalization (multi-word)"),
    ("ABC Corp.", "abc corp", "legal suffix normalization with trailing period"),
    ("A & B Traders", "a and b traders", "ampersand normalization"),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड",
     "non-Latin (Devanagari) script preserved unchanged (no ASCII folding)"),
    ("LLC Moncada Léarning Center", "llc moncada léarning center",
     "accented Latin character preserved, only lowercased"),
    (None, "", "None -> empty string"),
    ("", "", "empty string -> empty string"),
    (float("nan"), "", "float NaN -> empty string"),
    ("NULL", "", "null-like string -> empty string"),
    ("N/A", "", "null-like string -> empty string"),
    ("   ", "", "whitespace-only -> empty string"),
]

_ADDRESS_TEST_CASES = [
    ("1795 Westchester Drive, High Point, NC", "1795 westchester dr, high point, nc",
     "house number + street abbreviation preserved/normalized"),
    ("17560 Ellis Road, Tahlequah, OK", "17560 ellis rd, tahlequah, ok",
     "road abbreviation, house number preserved"),
    ("123 Main Street, Suite 400", "123 main st, ste 400",
     "street + suite abbreviations, unit number preserved"),
    ("US Highway 101", "us hwy 101", "highway number preserved"),
    ("560001", "560001", "bare PIN/ZIP code preserved exactly"),
    ("A-212, Malhotra Complex", "a-212, malhotra complex",
     "hyphenated house/plot number preserved"),
    (None, "", "None -> empty string"),
    ("", "", "empty string -> empty string"),
    ("unknown", "", "null-like string -> empty string"),
]


def _run_self_tests():
    failures = []

    print("=== normalize_business_name ===")
    for raw, expected, note in _NAME_TEST_CASES:
        actual = normalize_business_name(raw)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            failures.append(("name", raw, expected, actual))
        print(f"[{status}] {raw!r} -> {actual!r}   ({note})")

    print("\n=== normalize_business_address ===")
    for raw, expected, note in _ADDRESS_TEST_CASES:
        actual = normalize_business_address(raw)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            failures.append(("address", raw, expected, actual))
        print(f"[{status}] {raw!r} -> {actual!r}   ({note})")

    print(f"\n{len(_NAME_TEST_CASES) + len(_ADDRESS_TEST_CASES) - len(failures)} / "
          f"{len(_NAME_TEST_CASES) + len(_ADDRESS_TEST_CASES)} tests passed.")

    if failures:
        print("\nFAILURES:")
        for kind, raw, expected, actual in failures:
            print(f"  [{kind}] input={raw!r} expected={expected!r} actual={actual!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except AttributeError:
        pass
    _run_self_tests()

"""Source-bound lexical relevance matching, including serialized table formatting.

This scores literal evidence coverage, not semantic support or answer correctness.
Only an identified table and an available original physical page permit removal of
the parser's `` | `` cell separators. Numbers and all source punctuation survive.
"""

import re
from collections.abc import Mapping

_TOKEN = re.compile(r"[+-]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?:[%％])?"
                    r"|[A-Za-z_]+|[^\s]")


def _compact(text):
    return re.sub(r"\s+", "", text)


def _tokens_and_rows(text, *, table=False):
    tokens, rows, boundaries = [], [], set()
    for number, line in enumerate(text.splitlines()):
        # Tokenize each cell independently: e.g. 1 | 23 must never become 123.
        cells = line.split(" | ") if table else [line]
        for column, cell in enumerate(cells):
            matches = list(_TOKEN.finditer(cell))
            if matches and rows and (rows[-1] != number or table and column > 0):
                boundaries.add(len(tokens))
            previous = None
            for match in matches:
                if not table and previous is not None and match.start() > previous.end():
                    boundaries.add(len(tokens))
                tokens.append(match[0])
                rows.append(number)
                previous = match
    return tokens, rows, boundaries


def _occurrences(tokens, needle):
    if not needle:
        return
    size = len(needle)
    for index in range(len(tokens) - size + 1):
        if tokens[index] == needle[0] and tokens[index:index + size] == needle:
            yield index


def _row_breaks(rows, start, size):
    return {offset for offset in range(1, size)
            if rows[start + offset] != rows[start + offset - 1]}


def _relative_boundaries(boundaries, start, size):
    return {offset for offset in range(1, size) if start + offset in boundaries}


def _table_match(text, quote, original_page):
    wanted, _, _ = _tokens_and_rows(quote)
    source_tokens, source_rows, source_gaps = _tokens_and_rows(original_page)
    source_breaks = [(_row_breaks(source_rows, start, len(wanted)),
                      _relative_boundaries(source_gaps, start, len(wanted)))
                     for start in _occurrences(source_tokens, wanted)]
    if not source_breaks:
        return False
    candidate_tokens, candidate_rows, candidate_cells = _tokens_and_rows(text, table=True)
    for start in _occurrences(candidate_tokens, wanted):
        candidate_breaks = _row_breaks(candidate_rows, start, len(wanted))
        cell_breaks = _relative_boundaries(candidate_cells, start, len(wanted))
        # A wrapped source cell may span more physical lines than its serialized
        # row. The reverse is unsafe: never splice separate candidate rows into
        # a single source row merely because their tokens happen to concatenate.
        # Cell boundaries must also occur between source tokens; Chinese text
        # from different cells cannot be silently glued into a source word.
        if any(candidate_breaks.issubset(lines) and cell_breaks.issubset(gaps)
               for lines, gaps in source_breaks):
            return True
    return False


def match_gold(candidate, gold, page_texts=None):
    """Return covered gold IDs after exact report/physical-page constraints.

    ``page_texts`` is ``{report_id: {pdf_page_number: raw_text}}``; string page
    keys from JSON are accepted. When absent, retain the previous whitespace-only
    string comparison. When present, table matches require the quote on that
    exact source page, ordered cell tokens and compatible source line breaks.
    No fuzzy matching, punctuation stripping or semantic correctness is implied.
    """
    chunk = candidate.chunk
    pages = {ref["pdf_page_number"] for ref in chunk["page_refs"]}
    table = chunk.get("metadata", {}).get("kind", chunk.get("kind")) == "table"
    matched = set()
    for index, unit in enumerate(gold):
        quote = unit.get("quote")
        number = unit.get("pdf_page_number")
        if (unit.get("report_id") != chunk["report_id"] or number not in pages
                or not isinstance(quote, str) or not quote.strip()):
            continue
        if table and page_texts is not None:
            source_pages = page_texts.get(chunk["report_id"], {})
            if not isinstance(source_pages, Mapping):
                continue
            original = source_pages.get(number, source_pages.get(str(number)))
            covered = isinstance(original, str) and _table_match(chunk["text"], quote, original)
        else:
            covered = _compact(quote) in _compact(chunk["text"])
        if covered:
            matched.add(unit.get("id", f"g{index}"))
    return matched

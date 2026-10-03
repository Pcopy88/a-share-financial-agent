"""Bounded child chunks with original character/page mappings."""
import hashlib
import re
from collections.abc import Iterable
from typing import Protocol

from annual_report_agent.schemas.report import Chunk, PageRef, ParentBlock


class TokenCounter(Protocol):
    version: str

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class ChunkingError(ValueError):
    pass


def _references(parent: ParentBlock, start: int, end: int) -> list[PageRef]:
    segments = parent.metadata.get("segments", [])
    if not segments:
        return parent.page_refs
    refs = [PageRef.model_validate(s["page_ref"]) for s in segments
            if s["start"] < end and s["end"] > start]
    return list({r.model_dump_json(): r for r in refs}.values())


def _make(parent, tokenizer, text, start, end, ordinal, *, metadata=None):
    identity = f"{parent.id}:{tokenizer.version}:{start}:{end}:{ordinal}:{text}"
    return Chunk(
        id=hashlib.sha256(identity.encode()).hexdigest()[:32],
        report_id=parent.report_id, parent_id=parent.id, text=text,
        page_refs=_references(parent, start, end), section_path=parent.section_path,
        question_number=parent.question_number, report_periods=parent.report_periods,
        chunk_token_count=tokenizer.count(text), tokenizer_version=tokenizer.version,
        index_version=parent.index_version,
        metadata={"kind": parent.kind, "start_char": start, "end_char": end,
                  **(metadata or {})},
    )


def _text_chunks(parent, tokenizer, size, overlap):
    text = parent.text
    start, ordinal = 0, 0
    while start < len(text):
        remaining = text[start:]
        offsets = tokenizer.offsets(remaining)
        if not offsets:
            break
        end = len(text) if len(offsets) <= size else start + offsets[size - 1][1]
        # Prefer a paragraph or sentence end near the budget, preserving content.
        if end < len(text):
            boundaries = [m.end() for m in re.finditer(r"\n\n|[。！？!?]\s*", text[start:end])]
            eligible = [b for b in boundaries if b >= (end - start) // 2]
            if eligible:
                end = start + eligible[-1]
        while end > start and tokenizer.count(text[start:end]) > size:
            end -= 1
        if end <= start:
            raise ChunkingError(f"TOKEN_BOUNDARY_UNSPLITTABLE:{parent.id}")
        value = text[start:end].strip()
        if value:
            yield _make(parent, tokenizer, value, start, end, ordinal)
            ordinal += 1
        if end >= len(text):
            break
        used = tokenizer.offsets(text[start:end])
        if overlap and len(used) > overlap:
            next_start = start + used[-overlap][0]
        else:
            next_start = end
        # Offsets can coincide for Unicode pieces. Never loop over the same span.
        start = next_start if next_start > start else end
        while start < len(text) and text[start].isspace():
            start += 1


def _row_text(row):
    return " | ".join((cell or "").replace("\n", " ").strip() for cell in row)


def _table_prefix(parent, headers, columns=None):
    title = parent.metadata.get("table_header") or parent.section_path
    unit = parent.metadata.get("unit")
    selected = headers if columns is None else [
        [row[c] if c < len(row) else "" for c in columns] for row in headers]
    return "\n".join(v for v in [title, f"单位：{unit}" if unit else "",
                                  *map(_row_text, selected)] if v)


def _fragment_row(parent, row, row_index, headers, tokenizer, size, columns=None):
    """Repeat short fields; preserve long cell offsets and original column indices."""
    columns = list(range(len(row))) if columns is None else columns
    cells = [(row[c] or "") for c in columns]
    prefix = _table_prefix(parent, headers, columns)

    def render(values):
        return "\n".join([prefix, _row_text(values)]).strip()

    def details(focused=None, start=0, end=0):
        return {"row_indices": [row_index], "row_fragment": True,
                "header_column_indices": columns,
                "header_source_parent_id": parent.id,
                "cell_fragments": [
                    {"column_index": column, "start_char": start if k == focused else 0,
                     "end_char": end if k == focused else len(cells[k])}
                    for k, column in enumerate(columns)]}

    if tokenizer.count(render(cells)) <= size:
        yield render(cells), details()
        return
    largest = max(range(len(cells)), key=lambda n: tokenizer.count(cells[n]))
    skeleton = cells.copy()
    skeleton[largest] = ""
    # If columns/header consume the budget, select their full individual headers.
    if tokenizer.count(render(skeleton)) >= size:
        if len(columns) > 2:
            for column in columns[1:]:
                yield from _fragment_row(parent, row, row_index, headers, tokenizer, size,
                                         [columns[0], column])
            return
        raise ChunkingError(f"TABLE_COLUMN_CONTEXT_TOO_LONG:{parent.id}:row={row_index}")
    text, start = cells[largest], 0
    while start < len(text):
        low, high, fit = start + 1, len(text), start
        while low <= high:
            mid = (low + high) // 2
            current = cells.copy()
            current[largest] = text[start:mid]
            if tokenizer.count(render(current)) <= size:
                fit = mid
                low = mid + 1
            else:
                high = mid - 1
        if fit == start:
            raise ChunkingError(f"TABLE_CELL_UNSPLITTABLE:{parent.id}:row={row_index}")
        if fit < len(text):
            boundaries = [m.end() for m in re.finditer(r"\n|[。！？!?]|\s+", text[start:fit])]
            eligible = [b for b in boundaries if b >= (fit - start) // 2]
            if eligible:
                fit = start + eligible[-1]
        current = cells.copy()
        current[largest] = text[start:fit]
        yield render(current), details(largest, start, fit)
        start = fit


def _table_chunks(parent, tokenizer, size):
    rows = parent.metadata.get("rows", [])
    headers = parent.metadata.get("header_rows", [])
    if not rows:
        raise ChunkingError(f"TABLE_ROWS_MISSING:{parent.id}")
    prefix = _table_prefix(parent, headers)
    # Only skip the leading rows if these are genuinely present on this page.
    header_count = len(headers) if rows[:len(headers)] == headers else 0
    pending, row_ids, ordinal = [], [], 0
    for index, row in enumerate(rows[header_count:], header_count):
        rendered = _row_text(row)
        single = "\n".join([prefix, rendered]).strip()
        if tokenizer.count(single) > size:
            if pending:
                yield _make(parent, tokenizer, "\n".join([prefix, *pending]).strip(),
                            0, len(parent.text), ordinal, metadata={"row_indices": row_ids})
                ordinal += 1
                pending, row_ids = [], []
            for fragment, metadata in _fragment_row(parent, row, index, headers, tokenizer, size):
                yield _make(parent, tokenizer, fragment, 0, len(parent.text), ordinal,
                            metadata=metadata)
                ordinal += 1
            continue
        combined = "\n".join([prefix, *pending, rendered]).strip()
        if pending and tokenizer.count(combined) > size:
            yield _make(parent, tokenizer, "\n".join([prefix, *pending]).strip(),
                        0, len(parent.text), ordinal, metadata={"row_indices": row_ids})
            ordinal += 1
            pending, row_ids = [], []
        pending.append(rendered)
        row_ids.append(index)
    if pending or (prefix and ordinal == 0):
        value = "\n".join([prefix, *pending]).strip()
        if tokenizer.count(value) > size:
            raise ChunkingError(f"TABLE_HEADER_TOO_LONG:{parent.id}")
        yield _make(parent, tokenizer, value, 0, len(parent.text), ordinal,
                    metadata={"row_indices": row_ids})


def chunk_parents(parents: Iterable[ParentBlock], tokenizer: TokenCounter,
                  chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    chunks = []
    for parent in parents:
        if not parent.text.strip():
            continue
        generated = (_table_chunks(parent, tokenizer, chunk_size) if parent.kind == "table"
                     else _text_chunks(parent, tokenizer, chunk_size, overlap))
        chunks.extend(generated)
    return chunks

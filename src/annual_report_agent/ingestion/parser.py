"""Parse local PDFs without treating extraction success as financial verification."""
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from annual_report_agent.ingestion.chunker import TokenCounter, chunk_parents
from annual_report_agent.ingestion.tokenizer import FixedTokenizer
from annual_report_agent.schemas.report import Page, PageRef, ParentBlock, ParsedReport, Report

PARSER_VERSION = "pdfplumber-structure-v4"
_PAGE_LABEL = re.compile(r"^(?:第\s*)?(\d+(?:[-－]\d+)*)(?:\s*页)?(?:\s*共\s*\d+\s*页)?$")
_STATEMENT = re.compile(r"(合并|母公司)(资产负债表|利润表|现金流量表|所有者权益变动表)")
_HEADING = re.compile(r"^(第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|"
                      r"[（(][一二三四五六七八九十\d]+[）)]|问题\s*\d+|\d+[、．]\s*\D)")
_QUESTION = re.compile(r"^(?:问题\s*|第)(\d+)(?:[、.．：:]|题|\s)")


class PDFParseError(RuntimeError):
    pass


def _compact(value):
    return re.sub(r"\s+", "", value)


def _identify(report, text):
    compact = _compact(text)
    if report.doc_type == "other":
        if "半年度报告" in compact:
            report.doc_type = "semiannual_report"
        elif "年度报告" in compact:
            report.doc_type = "annual_report"
        elif "问询" in compact and "回复" in compact:
            report.doc_type = "inquiry_response"
        elif "问询函" in compact:
            report.doc_type = "inquiry_letter"
        elif "招股说明书" in compact:
            report.doc_type = "prospectus"
    if report.doc_type in ("semiannual_report", "annual_report"):
        match = re.search(r"(20\d{2})年(?:半)?年度报告", compact)
        if match and report.fiscal_year is None:
            report.fiscal_year = int(match[1])
        if report.fiscal_year and not report.report_periods:
            year = report.fiscal_year
            half = report.doc_type == "semiannual_report"
            report.report_periods = [{"label": f"{year}H1" if half else str(year),
                                      "period_start": f"{year}-01-01",
                                      "period_end": f"{year}-06-30" if half else f"{year}-12-31"}]
    if report.inquiry_round is None and report.doc_type.startswith("inquiry_"):
        match = re.search(r"第([一二三四五六七八九十\d]+)轮(?:审核)?问询", compact)
        if match:
            raw = match[1]
            report.inquiry_round = int(raw) if raw.isdecimal() else {
                "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(raw)


def _header_rows(rows):
    result = []
    for row in rows[:3]:
        # A page-border detector can collapse many prose rows into one cell.
        # The presence of a year or the word 项目 does not make it a header.
        if any(len(_compact(cell or "")) > 120 for cell in row):
            break
        text = _compact(" ".join(c or "" for c in row))
        if any(word in text for word in ("项目", "期末余额", "期初余额", "本期发生额",
                                         "上期发生额", "本报告期", "上年同期")) or (
                not result and re.search(r"20\d{2}年", text)):
            result.append(row)
        else:
            break
    return result


def _inside(line, bbox):
    cx = (line["x0"] + line["x1"]) / 2
    cy = (line["top"] + line["bottom"]) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _stable(report, kind, page, ordinal, text):
    raw = f"{report.id}:{report.sha256}:{PARSER_VERSION}:{kind}:{page}:{ordinal}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class _LayoutTable:
    bbox: tuple
    rows: list
    layout_metadata: dict

    def extract(self):
        return self.rows


def _word_rows(words):
    """Group baselines, allowing small font/bold offsets without joining adjacent rows."""
    groups = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not groups or word["top"] - groups[-1][0]["top"] > 2.5:
            groups.append([word])
        else:
            groups[-1].append(word)
    return [sorted(group, key=lambda item: item["x0"]) for group in groups]


def _cluster(values, tolerance=2):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _horizontal_spans(page):
    edges = [edge for edge in page.edges if edge["orientation"] == "h" and edge["x1"] > edge["x0"]]
    spans = []
    for top in _cluster([edge["top"] for edge in edges], 1.5):
        intervals = sorted((edge["x0"], edge["x1"]) for edge in edges if abs(edge["top"] - top) < 1.6)
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + 3:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        spans.extend((start, top, end) for start, end in merged if end - start > page.width * .4)
    return spans


def _vertical_separators(page, left, top, right, bottom):
    edges = [edge for edge in page.edges if edge["orientation"] == "v"
             and left + 2 < edge["x0"] < right - 2 and edge["bottom"] > top and edge["top"] < bottom]
    result = []
    for x in _cluster([edge["x0"] for edge in edges], 2):
        intervals = sorted((max(top, edge["top"]), min(bottom, edge["bottom"]))
                           for edge in edges if abs(edge["x0"] - x) < 2.1)
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        if sum(end - start for start, end in merged) >= (bottom - top) * .45:
            result.append(x)
    return result


_NUMERAL = re.compile(r"[+-]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?")
_NOTE = re.compile(r"[一二三四五六七八九十]+[、,，]\d+(?:[（(]\d+[）)])?")


def _text_column_bounds(header, body):
    """Infer gutters only from matching period headings and stable amount alignment."""
    years = [word for word in header if re.search(r"(?:19|20)\d{2}", word["text"])]
    numeric = [word for group in body for word in group if _NUMERAL.fullmatch(_compact(word["text"]))]
    anchors = _cluster([word["x1"] for word in numeric], 3)
    if len(anchors) != len(years) or len(years) < 2:
        return None
    spans = []
    note_header = [word for word in header if "附注" in _compact(word["text"])]
    first_amount = min(word["x0"] for word in years)
    notes = [word for group in body for word in group if _NOTE.fullmatch(_compact(word["text"]))]
    labels = [word for group in body for word in group if word["x0"] < first_amount
              and not _NOTE.fullmatch(_compact(word["text"]))
              and not _NUMERAL.fullmatch(_compact(word["text"]))]
    item_header = [word for word in header if word["x1"] < (note_header or years)[0]["x0"]]
    if not labels or not item_header:
        return None
    spans.append((min(w["x0"] for w in labels + item_header), max(w["x1"] for w in labels + item_header)))
    if note_header:
        spans.append((min(w["x0"] for w in notes + note_header), max(w["x1"] for w in notes + note_header)))
    for anchor, year in zip(anchors, years, strict=True):
        values = [word for word in numeric if abs(word["x1"] - anchor) <= 3]
        if len(values) < 2:
            return None
        spans.append((min(year["x0"], min(w["x0"] for w in values)),
                      max(year["x1"], max(w["x1"] for w in values))))
    if any(first[1] >= second[0] for first, second in zip(spans, spans[1:], strict=False)):
        return None
    return [spans[0][0] - 2, *[(a[1] + b[0]) / 2 for a, b in zip(spans, spans[1:], strict=False)],
            spans[-1][1] + 2]


def _recover_open_tables(page, existing, lines):
    """Recover open-border/borderless statement tables, never by company or amount."""
    if not any(_STATEMENT.search(_compact(line["text"])) for line in lines):
        return existing
    groups = _word_rows(page.extract_words())
    horizontals = _horizontal_spans(page)
    recovered = []
    for index, header in enumerate(groups):
        heading = _compact("".join(word["text"] for word in header))
        if "项目" not in heading or len(re.findall(r"(?:19|20)\d{2}", heading)) < 2:
            continue
        top = min(word["top"] for word in header)
        header_bottom = max(word["bottom"] for word in header)
        # Preserve successful fully ruled extraction instead of replacing all tables.
        valid_existing = False
        for table in existing:
            if table.bbox[1] - 3 <= top <= table.bbox[3]:
                rows = table.extract() or []
                if (rows and any("项目" in _compact(cell or "") for cell in rows[0])
                        and max(len(row) for row in rows) >= 3
                        and table.bbox[0] <= min(word["x0"] for word in header) + 3
                        and table.bbox[2] >= max(word["x1"] for word in header) - 3
                        and not any((cell or "").count("\n") > 3 for row in rows for cell in row)):
                    valid_existing = True
        if valid_existing:
            continue
        limit = page.height * .94
        for group in groups[index + 1:]:
            text = _compact("".join(word["text"] for word in group))
            if (_STATEMENT.search(text) or "法定代表人" in text or "主管会计" in text
                    or ("项目" in text and len(re.findall(r"(?:19|20)\d{2}", text)) >= 2)):
                limit = min(word["top"] for word in group)
                break
        nearby = [span for span in horizontals if top - 20 <= span[1] <= header_bottom + 2]
        border = max(nearby, key=lambda span: span[2] - span[0]) if nearby else None
        if border:
            left, _, right = border
            bottoms = [span[1] for span in horizontals if header_bottom + 3 < span[1] < limit
                       and span[0] <= left + 3 and span[2] >= right - 3]
            bottom = max(bottoms) if bottoms else limit
        else:
            left, right, bottom = 0, page.width, limit
        body = [group for group in groups[index + 1:] if min(w["top"] for w in group) < bottom
                and max(w["bottom"] for w in group) <= bottom + 2]
        verticals = _vertical_separators(page, left, top, right, bottom) if border else []
        bounds = [left, *verticals, right] if len(verticals) >= 2 else _text_column_bounds(header, body)
        if bounds is None:
            continue
        row_words = [header, *body]
        rows, row_bboxes, cell_bboxes = [], [], []
        uncertain = False
        for group in row_words:
            cells = [[] for _ in range(len(bounds) - 1)]
            for word in group:
                center = (word["x0"] + word["x1"]) / 2
                column = next((i for i in range(len(bounds) - 1) if bounds[i] <= center < bounds[i + 1]), None)
                if column is None:
                    continue
                if word["x0"] < bounds[column] - 2 or word["x1"] > bounds[column + 1] + 2:
                    uncertain = True
                cells[column].append(word)
            if not any(cells):
                continue
            rows.append([" ".join(word["text"] for word in cell) if cell else None for cell in cells])
            row_bboxes.append([min(w["x0"] for w in group), min(w["top"] for w in group),
                               max(w["x1"] for w in group), max(w["bottom"] for w in group)])
            cell_bboxes.append([[min(w["x0"] for w in cell), min(w["top"] for w in cell),
                                 max(w["x1"] for w in cell), max(w["bottom"] for w in cell)]
                                if cell else None for cell in cells])
        if not rows or len(rows) < 3:
            continue
        periods = [i for i, value in enumerate(rows[0]) if re.search(r"(?:19|20)\d{2}", value or "")]
        if len(periods) < 2 or len({_compact(rows[0][i]) for i in periods}) != len(periods):
            uncertain = True
        for column in periods:
            values = [_compact(row[column] or "") for row in rows[1:]]
            if sum(bool(_NUMERAL.fullmatch(value)) for value in values) < 2:
                uncertain = True
            if any(value and not _NUMERAL.fullmatch(value) and value not in {"-", "—", "--", "不适用"}
                   for value in values):
                uncertain = True
        if "项目" not in _compact(rows[0][0] or ""):
            uncertain = True
        bbox = (bounds[0], top - .1, bounds[-1], max(box[3] for box in row_bboxes) + .1)
        recovered.append(_LayoutTable(bbox, rows, {
            "row_bboxes": row_bboxes, "cell_bboxes": cell_bboxes, "column_boundaries": bounds,
            "table_detection": "vertical_edges_text_rows" if verticals else "header_numeric_gutters",
            "structure_status": "needs_review" if uncertain else "candidate",
        }))
    if not recovered:
        return existing
    retained = [table for table in existing if not any(
        table.bbox[0] < replacement.bbox[2] and table.bbox[2] > replacement.bbox[0]
        and table.bbox[1] < replacement.bbox[3] and table.bbox[3] > replacement.bbox[1]
        for replacement in recovered)]
    return retained + recovered


def parse_report(report: Report, tokenizer: TokenCounter | None = None,
                 chunk_size: int = 500, overlap: int = 80, *,
                 page_numbers: list[int] | None = None,
                 split_chunks: bool = True) -> ParsedReport:
    """Parse all or selected 1-based physical pages. Missing tokenizer never auto-downloads.

    Tables remain separate per page; continuation headers are inherited only across
    adjacent pages with equal column counts. Exact row/cell extraction is a candidate,
    not a verified FinancialFact. Original page text always remains available.
    """
    if split_chunks:
        tokenizer = tokenizer if tokenizer is not None else FixedTokenizer.load()
    output_report = report.model_copy(deep=True)
    pages, parents, warnings = [], [], []
    try:
        pdf = pdfplumber.open(Path(report.path))
    except Exception as exc:
        raise PDFParseError(f"PARSE_FAILED:{type(exc).__name__}") from exc
    with pdf:
        output_report.page_count = len(pdf.pages)
        selected = sorted(set(page_numbers or range(1, len(pdf.pages) + 1)))
        if any(n < 1 or n > len(pdf.pages) for n in selected):
            raise ValueError("page_numbers must be valid 1-based PDF pages")
        identity_text = "\n".join((p.extract_text() or "") for p in pdf.pages[:3])
        _identify(output_report, identity_text)
        # Collect raw pages once; repeated boundary text is removed only from blocks.
        prepared = []
        edge_frequency = Counter()
        for number in selected:
            page = pdf.pages[number - 1]
            raw, lines = "", []
            try:
                raw = page.extract_text() or ""
                lines = page.extract_text_lines(return_chars=False)
                tables = page.find_tables()
                edges = {_compact(line["text"]) for line in lines
                         if line["top"] < page.height * .085
                         or line["bottom"] > page.height * .93}
                edge_frequency.update(edges)
                prepared.append((number, page, raw, lines, tables, None))
            except Exception as exc:
                prepared.append((number, page, raw, lines, [], type(exc).__name__))
        section, question = "", None
        previous_table, previous_page = None, None
        statement, scope, unit, title_ref = "", None, None, None
        for number, page, raw, lines, tables, error in prepared:
            if previous_page is not None and number != previous_page + 1:
                previous_table = None
                statement, scope, unit, title_ref = "", None, None, None
            previous_page = number
            printed = None
            for line in lines:
                match = _PAGE_LABEL.fullmatch(_compact(line["text"]))
                if match and line["top"] > page.height * .90:
                    printed = match[1]
            status = "failed" if error else ("parsed" if raw.strip() else "needs_ocr")
            pages.append(Page(report_id=report.id, pdf_page_number=number,
                              printed_page_label=printed, raw_text=raw,
                              parse_status=status, error=error))
            if status != "parsed":
                warnings.append(f"PAGE_{status.upper()}:{number}")
                previous_table = None
                continue
            tables = _recover_open_tables(page, tables, lines)
            cleaned = []
            for line in lines:
                boundary = (line["top"] < page.height * .085
                            or line["bottom"] > page.height * .93)
                text = _compact(line["text"])
                if (boundary and not _STATEMENT.search(text) and "单位" not in text
                        and (edge_frequency[text] >= 2 or _PAGE_LABEL.fullmatch(text))):
                    continue
                cleaned.append(line)
            events = [(line["top"], "line", line) for line in cleaned
                      if not any(_inside(line, t.bbox) for t in tables)]
            events += [(t.bbox[1], "table", t) for t in tables]
            events.sort(key=lambda event: event[0])
            buffer = []

            def flush_text(buffer, number, printed, section, question):
                if not buffer:
                    return
                text = "\n".join(line["text"] for line in buffer)
                bbox = (min(x["x0"] for x in buffer), min(x["top"] for x in buffer),
                        max(x["x1"] for x in buffer), max(x["bottom"] for x in buffer))
                ref = PageRef(pdf_page_number=number, printed_page_label=printed, bbox=bbox)
                parents.append(ParentBlock(
                    id=_stable(report, "text", number, len(parents), text),
                    report_id=report.id, section_path=section, question_number=question,
                    report_periods=[], text=text, page_refs=[ref], metadata={
                        "segments": [{"start": 0, "end": len(text),
                                      "page_ref": ref.model_dump()}],
                        "parser_version": PARSER_VERSION}))
                buffer.clear()

            for _, kind, value in events:
                if kind == "line":
                    text = value["text"].strip()
                    statement_match = _STATEMENT.search(_compact(text))
                    if statement_match:
                        statement = statement_match[0]
                        scope = "consolidated" if statement_match[1] == "合并" else "parent"
                        title_ref = PageRef(pdf_page_number=number, printed_page_label=printed,
                                            bbox=(value["x0"], value["top"], value["x1"], value["bottom"]))
                        unit = None
                        previous_table = None
                    unit_match = re.search(r"单位[：:]\s*(人民币)?\s*(亿元|万元|元|千元)", text)
                    if unit_match:
                        unit = unit_match[2]
                    if len(text) <= 120 and _HEADING.match(text):
                        flush_text(buffer, number, printed, section, question)
                        if not statement_match:
                            statement, scope, unit, title_ref = "", None, None, None
                            previous_table = None
                        section = text
                        found = _QUESTION.match(text)
                        if found:
                            question = found[1]
                    buffer.append(value)
                    continue
                flush_text(buffer, number, printed, section, question)
                rows = value.extract()
                if not rows:
                    warnings.append(f"TABLE_EMPTY:{number}")
                    continue
                rows = [[cell.strip() if isinstance(cell, str) else cell for cell in row]
                        for row in rows]
                headers = _header_rows(rows)
                columns = max(map(len, rows))
                ref = PageRef(pdf_page_number=number, printed_page_label=printed, bbox=value.bbox)
                inherited = False
                header_refs = [ref.model_dump()] if headers else []
                # Never inherit over an unseen page or a new statement title/header.
                if (not headers and previous_table and previous_table["page"] == number - 1
                        and columns == previous_table["columns"]
                        and statement == previous_table["table_header"]):
                    headers = previous_table["header_rows"]
                    header_refs = previous_table["header_page_refs"]
                    inherited = bool(headers)
                current_title = statement or section
                uncertain_structure = (len(rows) <= 2 and any(
                    len(_compact(cell or "")) > 400 for row in rows for cell in row))
                layout_metadata = getattr(value, "layout_metadata", {})
                uncertain_structure |= layout_metadata.get("structure_status") == "needs_review"
                if uncertain_structure:
                    warnings.append(f"TABLE_STRUCTURE_UNCERTAIN:{number}")
                table_text = "\n".join(filter(None, [current_title,
                    f"单位：{unit}" if unit else "",
                    *(" | ".join((c or "").replace("\n", " ") for c in row) for row in rows)]))
                metadata = {"rows": rows, "bbox": list(value.bbox), "scope": scope,
                            "table_header": current_title, "unit": unit,
                            "header_rows": headers, "header_page_refs": header_refs,
                            "title_page_refs": [title_ref.model_dump()] if title_ref else [],
                            "header_inherited": inherited, "parser_version": PARSER_VERSION,
                            "structure_status": "needs_review" if uncertain_structure else "candidate",
                            "scope_verification": "candidate" if scope else "unknown", **layout_metadata}
                parents.append(ParentBlock(
                    id=_stable(report, "table", number, len(parents), table_text),
                    report_id=report.id, kind="table", section_path=section,
                    question_number=question, report_periods=[], text=table_text,
                    page_refs=[ref], metadata=metadata))
                previous_table = {"page": number, "columns": columns,
                                  "table_header": statement, "header_rows": headers,
                                  "header_page_refs": header_refs}
            flush_text(buffer, number, printed, section, question)
    chunks = chunk_parents(parents, tokenizer, chunk_size, overlap) if split_chunks else []
    output_report.status = "parsed" if all(p.parse_status == "parsed" for p in pages) else "partial"
    if not parents:
        output_report.status = "needs_ocr" if any(p.parse_status == "needs_ocr" for p in pages) else "failed"
    output_report.capabilities["text_search"] = {
        "status": "parsed_not_indexed" if parents else "unavailable",
        "reason": "Embedding/index publication is a separate step"}
    output_report.capabilities.setdefault("verified_financial_facts", {
        "status": "unverified", "reason": "PDF extraction is not financial verification"})
    return ParsedReport(report=output_report, pages=pages, parents=parents, chunks=chunks,
                        warnings=warnings)

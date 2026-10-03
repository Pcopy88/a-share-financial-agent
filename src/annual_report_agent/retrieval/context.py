"""Expand only selected parent text and preserve the pages actually sent."""

import hashlib

from annual_report_agent.schemas.answer import Evidence


def estimate_tokens(text: str) -> int:
    # Explicit conservative character estimate, not BGE tokens or cloud billing.
    return max(1, len(text))


def evidence_id(version: str, chunk_id: str, role: str, quote: str) -> str:
    return "e-" + hashlib.sha256(f"{version}:{chunk_id}:{role}:{quote}".encode()).hexdigest()[:24]


def expand_context(candidates, repository, budget_tokens: int = 6000, top_k: int = 5) -> list[Evidence]:
    evidence, seen_quotes, selected_parents = [], set(), {}
    used = 0
    for candidate in candidates[:top_k]:
        chunk, version = candidate.chunk, candidate.index_version
        report_metadata = repository.build(version)["report_metadata"]
        quote = chunk["text"]
        if quote in seen_quotes:
            continue
        cost = estimate_tokens(quote)
        if used + cost > budget_tokens:
            continue
        used += cost
        seen_quotes.add(quote)
        evidence.append(Evidence(id=evidence_id(version, chunk["id"], "primary", quote),
            report_id=chunk["report_id"], parent_id=chunk["parent_id"], chunk_ids=[chunk["id"]],
            quote=quote, page_refs=chunk["page_refs"], index_version=version,
            section_path=chunk.get("section_path", ""), metadata=report_metadata | chunk.get("metadata", {}) |
            {"token_count_method": "character_estimate", "question_number": chunk.get("question_number")}))
        selected_parents.setdefault((version, chunk["parent_id"]), []).append(chunk)
    for (version, parent_id), chunks in selected_parents.items():
        parent = repository.get("parent", f"{version}:{parent_id}")
        report_metadata = repository.build(version)["report_metadata"]
        if not parent:
            raise ValueError("missing original parent context")
        text, refs, partial = parent["text"], parent["page_refs"], False
        if text in seen_quotes:
            continue
        if estimate_tokens(text) > budget_tokens - used:
            segments = parent.get("metadata", {}).get("segments", [])
            if not segments:
                continue  # Never label an arbitrary truncated table as complete.
            start = min(c.get("metadata", {}).get("start_char", 0) for c in chunks)
            end = max(c.get("metadata", {}).get("end_char", len(parent["text"])) for c in chunks)
            selected = [s for s in segments if s["start"] < end and s["end"] > start]
            if not selected:
                continue
            text = "\n".join(parent["text"][s["start"]:s["end"]] for s in selected)
            refs = [s["page_ref"] for s in selected]
            partial = True
        if text in seen_quotes or used + estimate_tokens(text) > budget_tokens:
            continue
        used += estimate_tokens(text)
        seen_quotes.add(text)
        evidence.append(Evidence(id=evidence_id(version, parent_id, "context", text),
            report_id=parent["report_id"], parent_id=parent_id,
            chunk_ids=[c["id"] for c in chunks], role="context", quote=text,
            page_refs=refs, index_version=version, section_path=parent.get("section_path", ""),
            metadata=report_metadata | {"partial_context": partial, "token_count_method": "character_estimate",
                      "question_number": parent.get("question_number")}))
    return evidence

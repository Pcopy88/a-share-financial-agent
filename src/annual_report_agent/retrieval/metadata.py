"""Apply the same explicit allow-list to every retrieval path."""


def matches_filters(report: dict, chunk: dict, filters: dict) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        if key in {"question_number", "section_path"}:
            actual = chunk.get(key)
        else:
            actual = report.get(key)
        if actual != expected:
            return False
    return True

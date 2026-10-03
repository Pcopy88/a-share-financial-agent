"""Reciprocal rank fusion combines ranks, never incompatible raw scores."""


def rrf(rankings: list[list], k: int = 60, limit: int = 30) -> list:
    scores, candidates = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            key = (item.index_version, item.chunk["id"])
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            candidates[key] = item
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [candidates[key].model_copy(update={"score": scores[key], "retrieval": "rrf"})
            for key in ordered]

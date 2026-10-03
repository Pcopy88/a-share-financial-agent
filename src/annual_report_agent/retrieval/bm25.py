"""One corpus per effective query scope, with Chinese financial terminology."""

import re

import jieba
from rank_bm25 import BM25Okapi

TOKENIZER = jieba.Tokenizer()
TERMS = ("归母净利润", "归属于母公司股东", "经营活动现金流量净额", "营业收入", "营业成本")
DICTIONARY_VERSION = {"implementation": "jieba-search-v1", "jieba": jieba.__version__, "terms": TERMS}
for _term in TERMS:
    TOKENIZER.add_word(_term)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKENIZER.lcut_for_search(text) if re.search(r"\w", token)]


def rank(question: str, candidates: list, limit: int = 30) -> list:
    if not candidates:
        return []
    corpus = [tokenize(candidate.chunk["text"]) for candidate in candidates]
    if not any(corpus):
        return []
    query = tokenize(question)
    scores = BM25Okapi(corpus).get_scores(query)
    # Zero can be a valid IDF on a tiny corpus. Require lexical overlap instead of score > 0.
    matches = [i for i, tokens in enumerate(corpus) if set(tokens).intersection(query)]
    matches.sort(key=lambda i: (-float(scores[i]), candidates[i].chunk["id"]))
    return [candidates[i].model_copy(update={"score": float(scores[i]), "retrieval": "bm25"})
            for i in matches[:limit]]

"""Batched, recoverable encoding and atomic publication of versioned collections."""

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError
from pydantic import BaseModel

from annual_report_agent.retrieval.bm25 import DICTIONARY_VERSION
from annual_report_agent.retrieval.bm25 import rank as bm25_rank
from annual_report_agent.retrieval.fusion import rrf
from annual_report_agent.retrieval.metadata import matches_filters
from annual_report_agent.storage.chroma_path import native_chroma_path


class Candidate(BaseModel):
    chunk: dict
    index_version: str
    score: float = 0
    retrieval: str = "dense"


def as_dict(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


class _UnhealthyCollection(ValueError):
    """A collection no longer matches its committed materialization records."""


class KnowledgeIndex:
    def __init__(self, repository, path: Path, embedder):
        self.repo = repository
        self.embedder = embedder
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(native_chroma_path(self.path)), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.usage_records: list[dict] = []

    def close(self):
        self.client.close()

    @staticmethod
    def _validate_ids(collection, identifiers):
        if identifiers:
            saved = collection.get(ids=list(identifiers), include=[])["ids"]
            if set(saved) != set(identifiers):
                raise _UnhealthyCollection("persistent vector collection is incomplete")

    def _probe_collection(self, collection, chunk, profile):
        # Exercise the vector index itself. ID reads alone do not load HNSW.
        # Existing document vectors make this a local check with no paid calls.
        vector = self.repo.cached_embedding(profile, "document", chunk["text"])
        if vector is None:
            stored = collection.get(ids=[chunk["id"]], include=["embeddings"])
            embeddings = stored.get("embeddings")
            if embeddings is None or len(embeddings) != 1:
                raise _UnhealthyCollection("persistent vector collection has no probe vector")
            vector = list(embeddings[0])
        found = collection.query(query_embeddings=[vector],
            where={"chunk_id": chunk["id"]}, n_results=1, include=[])
        if found["ids"] != [[chunk["id"]]]:
            raise _UnhealthyCollection("persistent vector collection failed query validation")

    def _build_collection(self, source_identity, report, chunks, profile, mode):
        active = self.repo.active_index(report["id"])
        version = source_identity
        if active and active.get("source_identity", active["index_version"]) == source_identity:
            version = active["index_version"]
        recovery_of = None
        chunk_map = {chunk["id"]: chunk for chunk in chunks}
        # Deterministic recovery links resume an interrupted repair. They also
        # preserve every old version used by a run snapshot; never delete one.
        for _ in range(32):
            saved = self.repo.build(version)
            if saved:
                try:
                    collection = self.client.get_collection(version, embedding_function=None)
                    encoded = saved["encoded_ids"]
                    self._validate_ids(collection, encoded)
                    if encoded:
                        self._probe_collection(collection, chunk_map[encoded[0]], profile)
                    elif saved["build_status"] == "ready":
                        raise _UnhealthyCollection("ready collection has no committed vectors")
                    return version, collection
                except (ChromaError, _UnhealthyCollection):
                    recovery_of = version
                    version = "idx-r-" + hashlib.sha256(
                        f"{source_identity}:{recovery_of}".encode()).hexdigest()[:40]
                    continue
            self.repo.save_build(version, report["id"], list(chunk_map), profile, mode,
                                 tokenizer_version=chunks[0]["tokenizer_version"],
                                 distance_metric="cosine", report_metadata=report,
                                 source_identity=source_identity, recovery_of=recovery_of)
            collection = self.client.get_or_create_collection(
                version, embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
            return version, collection
        raise _UnhealthyCollection("INDEX_RECOVERY_LIMIT: too many damaged recovery versions")

    def build(self, report, parents, chunks) -> dict:
        report, parents, chunks = as_dict(report), list(map(as_dict, parents)), list(map(as_dict, chunks))
        if not chunks:
            raise ValueError("EMPTY_INDEX: cannot publish an index without searchable chunks")
        parent_map = {parent["id"]: parent for parent in parents}
        if len(parent_map) != len(parents):
            raise ValueError("parent IDs must be unique")
        for chunk in chunks:
            parent = parent_map.get(chunk["parent_id"])
            if not parent or parent["report_id"] != report["id"] or chunk["report_id"] != report["id"]:
                raise ValueError("chunk parent/report association is invalid")
            if not chunk["text"].strip() or not chunk["page_refs"]:
                raise ValueError("chunk must contain text and page provenance")
        profile = self.embedder.profile.profile_id
        source_metadata = {key: value for key, value in report.items()
                           if key not in {"status", "path", "filename", "capabilities"}}
        identity = {"report": source_metadata, "parents": parents, "chunks": chunks,
                    "profile": profile, "schema": "2", "distance_metric": "cosine",
                    "chunk_config": {"chunk_size": 500, "overlap": 80},
                    "lexical_config": DICTIONARY_VERSION, "index_types": ["dense", "bm25"]}
        source_identity = "idx-" + hashlib.sha256(json.dumps(
            identity, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:40]
        # Source identity excludes processing status and local storage location.
        mode = self.embedder.profile.mode
        version, collection = self._build_collection(source_identity, report, chunks, profile, mode)
        for parent in parents:
            self.repo.put("parent", f"{version}:{parent['id']}", parent, report["id"])
        for chunk in chunks:
            self.repo.put("chunk", f"{version}:{chunk['id']}", chunk, report["id"])
        already = set(self.repo.build(version)["encoded_ids"])
        remaining = [c for c in chunks if c["id"] not in already]
        for start in range(0, len(remaining), 10):
            batch = remaining[start:start + 10]
            vectors = {c["id"]: self.repo.cached_embedding(profile, "document", c["text"])
                       for c in batch}
            missing = [c for c in batch if vectors[c["id"]] is None]
            if missing:
                result = self.embedder.encode_batch(
                    [c["text"] for c in missing], [c["id"] for c in missing], "document")
                if result.input_ids != [c["id"] for c in missing]:
                    raise ValueError("encoder output IDs mismatch")
                if len(result.ordered_vectors) != len(missing):
                    raise ValueError("incomplete embedding batch")
                for chunk, vector in zip(missing, result.ordered_vectors, strict=True):
                    self.repo.cache_embedding(profile, "document", chunk["text"], vector)
                    vectors[chunk["id"]] = vector
                self.usage_records.append(dict(phase="document_embedding", mode=mode,
                    request_id=result.request_id, total_tokens=result.usage_total_tokens,
                    attempts=result.attempts, latency=result.latency,
                    model=self.embedder.profile.model_id,
                    count=len(missing), profile=profile))
            collection.upsert(ids=[c["id"] for c in batch],
                              embeddings=[vectors[c["id"]] for c in batch],
                              metadatas=[{"chunk_id": c["id"], "report_id": report["id"]} for c in batch])
            for chunk in batch:
                self.repo.mark_chunk_encoded(version, chunk["id"])
        self._validate_ids(collection, [c["id"] for c in chunks])
        self._probe_collection(collection, chunks[0], profile)
        self.repo.publish_index(version)
        self.repo.put("report", report["id"], report)
        return self.repo.build(version)

    def snapshot(self, report_ids: list[str]) -> list[dict]:
        manifests = [self.repo.active_index(report_id) for report_id in report_ids]
        if any(manifest is None for manifest in manifests):
            raise ValueError("INDEX_NOT_READY: selected report has no active index")
        return manifests

    def retrieve(self, task, manifests: list[dict] | None = None,
                 mode: str = "dense", limit: int = 30) -> list[Candidate]:
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        manifests = manifests if manifests is not None else self.snapshot(task.report_ids)
        filters = task.hard_filters.model_dump(exclude_none=True)
        eligible, by_version = [], {}
        for manifest in manifests:
            if manifest["report_id"] not in task.report_ids:
                continue
            if manifest["embedding_profile_hash"] != self.embedder.profile.profile_id:
                raise ValueError("INDEX_PROFILE_MISMATCH: rebuild under current encoder")
            version = manifest["index_version"]
            items = []
            for chunk_id in manifest["chunk_ids"]:
                chunk = self.repo.get("chunk", f"{version}:{chunk_id}")
                if not chunk:
                    raise ValueError("INDEX_CORRUPTED: missing chunk")
                if matches_filters(manifest["report_metadata"], chunk, filters):
                    item = Candidate(chunk=chunk, index_version=version)
                    items.append(item)
                    eligible.append(item)
            by_version[version] = items
        if not eligible:
            return []
        lexical = bm25_rank(task.query_text, eligible, limit) if mode != "dense" else []
        dense = []
        if mode != "bm25":
            result = self.embedder.encode_batch([task.rewritten_query], [task.subtask_id], "query")
            self.usage_records.append(dict(phase="query_embedding", mode=self.embedder.profile.mode,
                request_id=result.request_id, total_tokens=result.usage_total_tokens, count=1,
                attempts=result.attempts, latency=result.latency, model=self.embedder.profile.model_id,
                profile=self.embedder.profile.profile_id))
            for version, candidates in by_version.items():
                if not candidates:
                    continue
                mapping = {item.chunk["id"]: item for item in candidates}
                collection = self.client.get_collection(version, embedding_function=None)
                found = collection.query(query_embeddings=result.ordered_vectors,
                    where={"chunk_id": {"$in": list(mapping)}},
                    n_results=min(limit, len(mapping)), include=["distances"])
                for identifier, distance in zip(found["ids"][0], found["distances"][0], strict=True):
                    dense.append(mapping[identifier].model_copy(update={"score": 1 - distance}))
            dense.sort(key=lambda c: (-c.score, c.chunk["id"]))
            dense = dense[:limit]
        if mode == "hybrid":
            return rrf([lexical, dense], limit=limit)
        return lexical if mode == "bm25" else dense

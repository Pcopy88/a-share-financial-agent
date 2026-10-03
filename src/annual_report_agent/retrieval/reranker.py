"""Pinned local cross-encoder. Scores rank relevance, never answer confidence."""

import math
from pathlib import Path
from time import perf_counter

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class RerankerError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


class LocalReranker:
    def __init__(self, backend, *, max_length=1024, batch_size=2, profile=None):
        self.backend = backend
        self.max_length = max_length
        self.batch_size = batch_size
        self.profile = profile or {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                                   "validation": "injected_backend"}
        self.last_run = {}

    @classmethod
    def from_config(cls, settings, *, allow_download=False):
        try:
            import torch
            from FlagEmbedding import FlagReranker
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RerankerError("RERANK_DEPENDENCY_MISSING",
                                "先安装 uv sync --extra rerank") from error
        device = settings.rerank_device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RerankerError("RERANK_DEVICE_UNAVAILABLE", "CUDA 不可用，请检查驱动或选择 cpu")
        try:
            path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                cache_dir=str(Path(settings.data_dir) / "models"),
                allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model"],
                local_files_only=not allow_download)
            backend = FlagReranker(path, use_fp16=device.startswith("cuda"), devices=[device],
                                   batch_size=settings.rerank_batch_size,
                                   max_length=settings.rerank_max_length)
        except Exception as error:
            raise RerankerError("RERANK_LOAD_FAILED",
                "本地权重未就绪或设备不兼容；运行 models prepare-reranker 查看详细检查") from error
        return cls(backend, max_length=settings.rerank_max_length,
                   batch_size=settings.rerank_batch_size,
                   profile={"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": device,
                            "precision": "fp16" if device.startswith("cuda") else "fp32",
                            "max_length": settings.rerank_max_length,
                            "batch_size": settings.rerank_batch_size})

    def rerank(self, question, candidates):
        if not candidates:
            return []
        pairs = [[question, candidate.chunk["text"]] for candidate in candidates]
        for index, (query, passage) in enumerate(pairs):
            encoded = self.backend.tokenizer(query, passage, truncation=False,
                                            add_special_tokens=True)
            if len(encoded["input_ids"]) > self.max_length:
                raise RerankerError("RERANK_INPUT_TOO_LONG",
                    f"候选 {candidates[index].chunk['id']} 超长，请缩短查询或重建更小的结构块")
        start = perf_counter()
        try:
            scores = self.backend.compute_score(pairs, batch_size=self.batch_size,
                max_length=self.max_length, query_max_length=self.max_length, normalize=True)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [scores]
            scores = [float(value) for value in scores]
        except Exception as error:
            raise RerankerError("RERANK_INFERENCE_FAILED", "本地重排失败，未自动切换排名") from error
        if len(scores) != len(candidates) or not all(math.isfinite(value) for value in scores):
            raise RerankerError("RERANK_INVALID_OUTPUT", "分数数量不符或存在无效数值")
        ranked = [candidate.model_copy(update={"score": score, "retrieval": "cross_encoder"})
                  for candidate, score in zip(candidates, scores, strict=True)]
        self.last_run = self.profile | {"pairs": len(pairs), "latency_seconds": perf_counter()-start}
        return sorted(ranked, key=lambda candidate: -candidate.score)

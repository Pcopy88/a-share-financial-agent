"""Run mock checks by default; --live explicitly opts into billable cloud probes."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from annual_report_agent.config import Settings
from annual_report_agent.generation.cloud_client import DeepSeekClient
from annual_report_agent.retrieval.embedding_client import EmbeddingClient, MockEmbeddingClient
from annual_report_agent.schemas.model import EmbeddingProfile


class ProbeOutput(BaseModel):
    status: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="执行收费云端请求，需自行配置两家Key")
    parser.add_argument("--output-dir", default="outputs/model-probes")
    args = parser.parse_args()
    settings = Settings()
    mode = "live" if args.live else "mock"
    result = {"mode": mode, "live_verified": False, "checked_at": datetime.now(UTC).isoformat()}
    output = Path(args.output_dir).resolve() / (mode + "-" + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.live:
            missing = settings.missing_cloud_config()
            if missing:
                raise ValueError("待配置环境变量：" + ", ".join(missing))
            embedder = EmbeddingClient(api_key=settings.dashscope_api_key.get_secret_value(),
                profile=EmbeddingProfile(endpoint=settings.embedding_url, model_id=settings.embedding_model))
            generator = DeepSeekClient(api_key=settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url, model=settings.deepseek_model)
        else:
            embedder = MockEmbeddingClient()
            # Exercise the real response parser over an explicit fake network boundary.
            response = {"id": "mock-probe", "model": "mock-deepseek-contract", "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                 "content": '{"status":"mock_ok"}'}}], "usage": {"total_tokens": 0}}
            generator = DeepSeekClient(api_key="mock-no-real-key", base_url="https://mock.invalid",
                http_client=httpx.Client(transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json=response))))
        documents = embedder.encode_batch(["公司营业收入增长。", "报告披露了原材料风险。"],
                                          ["doc-1", "doc-2"], "document")
        query = embedder.encode_batch(["公司有哪些风险？"], ["query-1"], "query")
        client = chromadb.PersistentClient(str(output / "chroma"),
                                          settings=ChromaSettings(anonymized_telemetry=False))
        collection = client.create_collection("probe", embedding_function=None,
                                               configuration={"hnsw": {"space": "cosine"}})
        collection.add(ids=documents.input_ids, embeddings=documents.ordered_vectors)
        found = collection.query(query_embeddings=query.ordered_vectors, n_results=1)
        generated = generator.generate_json("Return json only: {\"status\":\"ok\"}",
                                              "请按给定 json 格式返回 status。", max_tokens=64)
        validated = ProbeOutput.model_validate(generated.data)
        result.update(embedding_dimension=len(documents.ordered_vectors[0]),
                      document_count=len(documents.input_ids),
                      chroma_query_succeeded=bool(found["ids"][0]),
                      generation_schema_succeeded=bool(validated.status),
                      embedding_request_ids=[documents.request_id, query.request_id],
                      generation_request_id=generated.request_id,
                      embedding_usage=[documents.usage_total_tokens, query.usage_total_tokens],
                      generation_usage=generated.usage if args.live else None,
                      live_verified=args.live, status="passed")
    except Exception as exc:
        message = str(exc)
        for secret in (settings.deepseek_api_key.get_secret_value(), settings.dashscope_api_key.get_secret_value()):
            if secret:
                message = message.replace(secret, "***")
        result.update(status="failed", error=message)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

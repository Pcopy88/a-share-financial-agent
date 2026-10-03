"""Cross-process restart beyond HNSW's default persistence threshold."""

import json
import os
import subprocess
import sys


def test_unicode_path_persists_1100_vectors_across_independent_processes(tmp_path):
    directory = tmp_path / "中文' & 数据"
    script = r'''
import json, sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
from annual_report_agent.storage.chroma_path import native_chroma_path
path = native_chroma_path(Path(sys.argv[1]))
client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
try:
    collection = client.get_or_create_collection('restart-proof', embedding_function=None)
    if sys.argv[2] == 'write':
        collection.add(ids=[str(i) for i in range(1100)],
            embeddings=[[float(i % 7 + 1), 1., 2., 3., 4., 5., 6., 7.] for i in range(1100)])
    result = collection.query(query_embeddings=[[1.,1.,2.,3.,4.,5.,6.,7.]],n_results=3)
    print(json.dumps({'count':collection.count(), 'hits':len(result['ids'][0]), 'path':path}))
finally:
    client.close()
'''
    environment = os.environ | {"PYTHONIOENCODING": "utf-8"}
    for phase in ("write", "read"):
        result = subprocess.run([sys.executable, "-c", script, str(directory), phase],
                                capture_output=True, text=True, encoding="utf-8",
                                env=environment, timeout=60, check=True)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        assert data["count"] == 1100 and data["hits"] == 3
        if os.name == "nt":
            assert data["path"].isascii()
    assert list(directory.rglob("header.bin")), "native binary persistence must occur"

"""Pinned CPU tokenizer only. Never load embedding or generative weights."""
import hashlib
from pathlib import Path

import httpx
from tokenizers import Tokenizer

TOKENIZER_REPOSITORY = "BAAI/bge-m3"
TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
TOKENIZER_SHA256 = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"


class TokenizerUnavailable(RuntimeError):
    pass


class FixedTokenizer:
    def __init__(self, path: Path):
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != TOKENIZER_SHA256:
            raise TokenizerUnavailable(f"TOKENIZER_HASH_MISMATCH:{path}")
        self._tokenizer = Tokenizer.from_file(str(path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        self.revision = TOKENIZER_REVISION
        self.sha256 = actual
        self.version = f"{TOKENIZER_REPOSITORY}@{self.revision}:sha256:{actual}"

    @classmethod
    def load(cls, cache_dir: str | Path = "data/tokenizer", *,
             allow_download: bool = False) -> "FixedTokenizer":
        path = Path(cache_dir) / TOKENIZER_REVISION / "tokenizer.json"
        if not path.is_file():
            if not allow_download:
                raise TokenizerUnavailable(
                    f"TOKENIZER_UNAVAILABLE:{path}; explicitly download the pinned tokenizer first")
            url = (f"https://huggingface.co/{TOKENIZER_REPOSITORY}/resolve/"
                   f"{TOKENIZER_REVISION}/tokenizer.json")
            try:
                response = httpx.get(url, follow_redirects=True, timeout=60)
                response.raise_for_status()
                content = response.content
            except httpx.HTTPError as exc:
                raise TokenizerUnavailable("TOKENIZER_UNAVAILABLE:download failed") from exc
            if hashlib.sha256(content).hexdigest() != TOKENIZER_SHA256:
                raise TokenizerUnavailable("TOKENIZER_HASH_MISMATCH:download rejected")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.partial")
            temporary.write_bytes(content)
            temporary.replace(path)
        return cls(path)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self._tokenizer.encode(text, add_special_tokens=False).offsets

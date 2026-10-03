"""Prepare source-bound annual-report evaluation material without cloud calls.

Downloads are limited to six explicitly selected official URLs. Only development
documents are text-extracted by this script; held-out labels require a later freeze.
Run: .venv/Scripts/python.exe scripts/prepare_annual_eval.py --download --extract-development
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    {"id": "moutai_2023", "company": "贵州茅台酒股份有限公司", "stock_code": "600519", "year": 2023,
     "split": "development", "disclosure_date": "2024-04-03",
     "url": "https://dataclouds.cninfo.com.cn/shgonggao/2024/2024-04-03/93d5007bf0fa11ee84b5fa163e26e5de.pdf"},
    {"id": "moutai_2024", "company": "贵州茅台酒股份有限公司", "stock_code": "600519", "year": 2024,
     "split": "development", "disclosure_date": "2025-04-03",
     "url": "https://static.cninfo.com.cn/finalpage/2025-04-03/1222993920.PDF"},
    {"id": "gree_2023", "company": "珠海格力电器股份有限公司", "stock_code": "000651", "year": 2023,
     "split": "development", "disclosure_date": "2024-04-30",
     "url": "https://static.cninfo.com.cn/finalpage/2024-04-30/1219928418.PDF"},
    {"id": "gree_2024", "company": "珠海格力电器股份有限公司", "stock_code": "000651", "year": 2024,
     "split": "development", "disclosure_date": "2025-04-28",
     "url": "https://static.cninfo.com.cn/finalpage/2025-04-28/1223330631.PDF"},
    {"id": "fuyao_2023", "company": "福耀玻璃工业集团股份有限公司", "stock_code": "600660", "year": 2023,
     "split": "test", "disclosure_date": "2024-03-16",
     "url": "https://www.fuyaogroup.com/upfiles/investor/202403/1710498588937.pdf"},
    {"id": "fuyao_2024", "company": "福耀玻璃工业集团股份有限公司", "stock_code": "600660", "year": 2024,
     "split": "test", "disclosure_date": "2025-03-19",
     "url": "https://static.cninfo.com.cn/finalpage/2025-03-19/1222834034.PDF"},
]
HOSTS = {"static.cninfo.com.cn", "dataclouds.cninfo.com.cn", "www.sse.com.cn", "static.sse.com.cn", "www.fuyaogroup.com"}


def download_known_source(url):
    """Fixed trusted sources only; use the workstation's configured proxy/TLS route.

    This preparation helper is not the application's arbitrary-candidate downloader.
    The workstation resolves external hosts into its proxy's 198.18/15 Fake-IP range;
    the production downloader intentionally continues to reject non-public DNS.
    """
    limit = 100 * 1024 * 1024

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            return None

    opener = build_opener(NoRedirect)
    for _ in range(4):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.port not in (None, 443):
            raise ValueError("unexpected official-source redirect")
        deadline = time.monotonic() + 60
        from urllib.error import HTTPError
        request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
        try:
            response = opener.open(request, timeout=20)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                url = urljoin(url, exc.headers.get("Location", ""))
                continue
            raise
        with response:
            if response.status in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200:
                raise ValueError(f"source HTTP status {response.status}")
            parts, size = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("source download deadline exceeded")
                part = response.read1(min(65536, limit - size + 1))
                if not part:
                    break
                size += len(part)
                if size > limit:
                    raise ValueError("PDF size exceeds 100 MiB")
                parts.append(part)
            data = b"".join(parts)
            if not data.startswith(b"%PDF-"):
                raise ValueError("source did not return PDF bytes")
            return data, url
    raise ValueError("too many source redirects")


def prepare(download=False, extract_development=False):
    folder = ROOT / "data/samples/annual"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "evals/annual_sample_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    previous_by_id = {entry["id"]: entry for entry in previous.get("samples", [])}
    results = []
    for sample in SAMPLES:
        path = folder / (sample["id"] + ".pdf")
        entry = {**previous_by_id.get(sample["id"], {}), **sample}
        entry["source_url"] = sample["url"]
        entry["fiscal_year"] = sample["year"]
        entry["report_periods"] = [{"label": str(sample["year"]),
                                    "period_start": f"{sample['year']}-01-01",
                                    "period_end": f"{sample['year']}-12-31"}]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing selected source: {sample['id']}")
            data, final_url = download_known_source(sample["url"])
            with path.open("xb") as stream:
                stream.write(data)
            entry["downloaded_at"] = datetime.now(UTC).isoformat()
            entry["resolved_source_url"] = final_url
        raw = path.read_bytes()
        with pdfplumber.open(path) as pdf:
            cover = (pdf.pages[0].extract_text() or "").replace(" ", "").replace("\n", "")
            if str(sample["year"]) not in cover or sample["company"] not in cover or "摘要" in cover:
                raise ValueError(f"source cover identity mismatch: {sample['id']}")
            if len(pdf.pages) > 1000:
                raise ValueError("source exceeds 1000-page project limit")
            entry.update({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                          "size_bytes": len(raw), "page_count": len(pdf.pages),
                          "title": f"{sample['company']}{sample['year']}年年度报告", "doc_type": "annual_report",
                          "identity_verification": "assistant_checked_cover_text_and_official_source",
                          "human_reviewed": False,
                          "label_status": entry.get("label_status", "not_prepared")})
            if extract_development and sample["split"] == "development":
                destination = ROOT / "tmp/annual_eval" / sample["id"]
                destination.mkdir(parents=True, exist_ok=True)
                pages = [page.extract_text() or "" for page in pdf.pages]
                (destination / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
                (destination / "extracted.txt").write_text(
                    "\n\n".join(f"PDF_PAGE_{i + 1}\n{text}" for i, text in enumerate(pages)), encoding="utf-8")
        results.append(entry)
        print(json.dumps({"id": entry["id"], "sha256": entry["sha256"],
                          "pages": entry["page_count"], "split": entry["split"]}, ensure_ascii=False), flush=True)
        manifest = {"schema_version": "1.0", "purpose": "formal_annual_evaluation_sources",
                    "prepared_at": datetime.now(UTC).isoformat(), "samples": results,
                    "scope": "Selected 3 companies x 2 annual reports; not exhaustive market coverage",
                    "labeling": {**previous.get("labeling", {}), "development_target": 20, "test_target": 30,
                                 "test_labels_require_rules_freeze": True,
                                 "human_reviewed": False, "gold_complete": False}}
        (ROOT / "evals/annual_sample_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract-development", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.download, arguments.extract_development)

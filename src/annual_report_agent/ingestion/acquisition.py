"""Candidate search and explicitly selected CNInfo PDF acquisition.

AKShare's disclosure API returns detail-page URLs, not PDF download URLs:
https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_disclosure_cninfo.py

First integration: Shanghai/Shenzhen A-share annual full reports on CNInfo. Dynamic
detail pages without an explicit PDF link fail with PDF_LINK_UNRESOLVED; local upload
remains available. Search never chooses a revision or downloads PDFs for the user.
"""

import hashlib
import html
import http.client
import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic as _monotonic
from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd
from pydantic import BaseModel

from annual_report_agent.finance.akshare_adapter import AdapterError, call_akshare, market_symbol

_ALLOWED_HOSTS = frozenset({"www.cninfo.com.cn", "static.cninfo.com.cn"})
_ANNOUNCEMENT_COLUMNS = {"代码", "简称", "公告标题", "公告时间", "公告链接"}
MAX_PDF_BYTES = 100 * 1024 * 1024


class AcquisitionError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}; 可改用本地 PDF 上传")


class AnnouncementCandidate(BaseModel):
    id: str
    stock_code: str
    company: str
    report_year: int
    title: str
    detail_url: str
    disclosure_date: date
    revision: str
    pdf_url: str | None = None
    source_api: str = "stock_zh_a_disclosure_report_cninfo"


class DownloadResult(BaseModel):
    candidate_id: str
    path: str
    source_url: str
    pdf_url: str
    sha256: str
    downloaded_at: datetime
    disclosure_date: date
    report_year: int
    stock_code: str
    revision: str
    size_bytes: int


@dataclass(frozen=True)
class HTTPPayload:
    status: int
    headers: dict[str, str]
    body: bytes


def _validate_url(url: str):
    if not isinstance(url, str) or any(ord(character) < 32 for character in url) or "\\" in url:
        raise AcquisitionError("UNSAFE_URL")
    try:
        parts = urlsplit(url)
        valid = (parts.scheme == "https" and parts.hostname in _ALLOWED_HOSTS
                 and parts.port in (None, 443) and not parts.username and not parts.password)
    except ValueError as exc:
        raise AcquisitionError("UNSAFE_URL") from exc
    if not valid:
        raise AcquisitionError("UNSAFE_URL", "HTTPS CNInfo hosts only")
    return parts


def _resolve_dns(host: str) -> list[str]:
    return list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))


def _public_address(url: str, resolver) -> str:
    parts = _validate_url(url)
    try:
        addresses = resolver(parts.hostname)
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise AcquisitionError("UNSAFE_ADDRESS", "DNS result includes non-public address")
    except (OSError, ValueError) as exc:
        if isinstance(exc, AcquisitionError):
            raise
        raise AcquisitionError("UNSAFE_ADDRESS", "DNS validation failed") from exc
    return addresses[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, timeout=20, context=ssl.create_default_context())
        self.validated_address = address

    def connect(self):
        # Pin the already-validated address while verifying TLS against the hostname.
        raw = socket.create_connection((self.validated_address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


def _request(url: str, address: str, max_bytes: int) -> HTTPPayload:
    parts = _validate_url(url)
    connection = _PinnedHTTPSConnection(parts.hostname, address)
    deadline = _monotonic() + 60
    try:
        target = urlunsplit(("", "", parts.path or "/", parts.query, ""))
        connection.request("GET", target, headers={"Host": parts.hostname,
                                                   "User-Agent": "AnnualReportAgent/0.1",
                                                   "Accept-Encoding": "identity"})
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        if response.status in {301, 302, 303, 307, 308}:
            return HTTPPayload(response.status, headers, b"")
        length = headers.get("content-length")
        if length and (not length.isdecimal() or int(length) > max_bytes):
            raise AcquisitionError("DOWNLOAD_TOO_LARGE")
        chunks, size = [], 0
        while True:
            if _monotonic() > deadline:
                raise AcquisitionError("DOWNLOAD_TIMEOUT", "total response deadline exceeded")
            # read1 performs at most one underlying read, so trickled bytes cannot keep
            # read(n) waiting indefinitely without returning to the deadline check.
            part = response.read1(min(65536, max_bytes - size + 1))
            if not part:
                break
            size += len(part)
            if size > max_bytes:
                raise AcquisitionError("DOWNLOAD_TOO_LARGE")
            chunks.append(part)
        return HTTPPayload(response.status, headers, b"".join(chunks))
    except (OSError, http.client.HTTPException) as exc:
        raise AcquisitionError("SOURCE_UNAVAILABLE", type(exc).__name__) from exc
    finally:
        connection.close()


def _fetch(url: str, *, transport, resolver, max_bytes: int, max_redirects: int = 3):
    for hop in range(max_redirects + 1):
        address = _public_address(url, resolver)
        response = transport(url, address, max_bytes)
        headers = {key.lower(): value for key, value in response.headers.items()}
        if response.status in {301, 302, 303, 307, 308}:
            if hop == max_redirects or not headers.get("location"):
                raise AcquisitionError("REDIRECT_LIMIT")
            url = urljoin(url, headers["location"])
            continue
        if response.status != 200:
            raise AcquisitionError("DOWNLOAD_HTTP_ERROR", str(response.status))
        if len(response.body) > max_bytes:
            raise AcquisitionError("DOWNLOAD_TOO_LARGE")
        return response.body, url
    raise AcquisitionError("REDIRECT_LIMIT")


def search_announcements(stock_code: str, year: int, *, ak_client=None) -> list[AnnouncementCandidate]:
    """Return full-report candidates, preserving original/revised versions for selection.

    Search disclosure dates from the report year through the second following year;
    later revisions outside this explicit window are not guaranteed to be discovered.
    """
    try:
        market_symbol(stock_code)
    except AdapterError as exc:
        raise AcquisitionError(exc.code, str(exc)) from exc
    if not isinstance(year, int) or not 1990 <= year <= date.today().year:
        raise AcquisitionError("INVALID_REPORT_YEAR")
    params = {"symbol": stock_code, "market": "沪深京", "keyword": f"{year}年",
              "category": "年报", "start_date": f"{year}0101",
              "end_date": min(date.today(), date(year + 2, 12, 31)).strftime("%Y%m%d")}
    try:
        frame = (call_akshare("stock_zh_a_disclosure_report_cninfo", params)[0] if ak_client is None
                 else ak_client.stock_zh_a_disclosure_report_cninfo(**params))
    except AdapterError as exc:
        raise AcquisitionError(exc.code, str(exc)) from exc
    except Exception as exc:
        raise AcquisitionError("SOURCE_UNAVAILABLE", type(exc).__name__) from exc
    if not isinstance(frame, pd.DataFrame) or not _ANNOUNCEMENT_COLUMNS.issubset(frame.columns):
        raise AcquisitionError("FIELD_SCHEMA_CHANGED", "announcement columns changed")
    if len(frame) > 500:
        raise AcquisitionError("TOO_MANY_CANDIDATES", "narrow the disclosure range")
    candidates = {}
    for row in frame.to_dict(orient="records"):
        if str(row["代码"]).zfill(6) != stock_code:
            continue
        title = html.unescape(re.sub(r"<[^>]*>", "", str(row["公告标题"]))).strip()
        compact = re.sub(r"\s+", "", title)
        if (not re.search(rf"{year}年(?:年度|度)?报告", compact)
                or any(word in compact for word in ("摘要", "半年度", "更正公告", "更正说明", "补充公告"))):
            continue
        url = str(row["公告链接"]).strip()
        # AKShare currently formats fixed official detail links with HTTP. Upgrade only
        # that exact documented hostname; never follow an arbitrary insecure redirect.
        if url.startswith("http://www.cninfo.com.cn/"):
            url = "https://" + url[len("http://"):]
        _validate_url(url)
        try:
            disclosed = date.fromisoformat(str(row["公告时间"])[:10])
        except ValueError as exc:
            raise AcquisitionError("FIELD_SCHEMA_CHANGED", "invalid disclosure date") from exc
        identifier = hashlib.sha256(url.encode()).hexdigest()[:24]
        candidates[identifier] = AnnouncementCandidate(
            id=identifier, stock_code=stock_code, company=str(row["简称"]), report_year=year,
            title=title, detail_url=url, disclosure_date=disclosed,
            revision="revised" if any(word in compact for word in ("更正后", "修订", "更新后")) else "original",
        )
    return sorted(candidates.values(), key=lambda item: (item.disclosure_date, item.id), reverse=True)


class _PDFLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in {"a", "iframe", "embed", "object"}:
            self.links.extend(value for key, value in attrs if key in {"href", "src", "data"} and value)


def _pdf_link(body: bytes, detail_url: str) -> str:
    text = body.decode("utf-8", errors="replace")
    parser = _PDFLinks()
    parser.feed(text)
    links = parser.links + re.findall(r'["\']adjunctUrl["\']\s*:\s*["\']([^"\']+)["\']', text)
    found = set()
    for value in links:
        value = html.unescape(value).replace("\\/", "/")
        if value.startswith("finalpage/"):
            value = "https://static.cninfo.com.cn/" + value
        resolved = urljoin(detail_url, value)
        if urlsplit(resolved).path.lower().endswith(".pdf"):
            _validate_url(resolved)
            found.add(resolved)
    if len(found) != 1:
        raise AcquisitionError("PDF_LINK_UNRESOLVED" if not found else "AMBIGUOUS_PDF_LINK")
    return found.pop()


def download_candidate(candidate: AnnouncementCandidate, destination: str | Path, *,
                       selected: bool = False, max_bytes: int = MAX_PDF_BYTES,
                       transport=None, resolver=None) -> DownloadResult:
    """Download only a user-selected candidate; the caller must persist that selection.

    Each hop requires HTTPS, an exact allowlisted hostname, public-only DNS results,
    and a TLS connection pinned to one checked IP. Redirects and file size are bounded.
    Injected transport/resolver are for source-boundary tests, not user configuration.
    """
    if not selected:
        raise AcquisitionError("SELECTION_REQUIRED", "choose one announcement version first")
    if not 0 < max_bytes <= MAX_PDF_BYTES:
        raise AcquisitionError("INVALID_DOWNLOAD_LIMIT")
    transport, resolver = transport or _request, resolver or _resolve_dns
    initial_url = candidate.pdf_url or candidate.detail_url
    direct_pdf = urlsplit(initial_url).path.lower().endswith(".pdf")
    body, final_url = _fetch(initial_url, transport=transport, resolver=resolver,
                             max_bytes=max_bytes if direct_pdf else min(max_bytes, 2 * 1024 * 1024))
    if not direct_pdf and not body.startswith(b"%PDF-"):
        pdf_url = _pdf_link(body, final_url)
        body, final_url = _fetch(pdf_url, transport=transport, resolver=resolver, max_bytes=max_bytes)
    if not body.startswith(b"%PDF-"):
        raise AcquisitionError("INVALID_PDF", "missing PDF magic bytes")
    digest = hashlib.sha256(body).hexdigest()
    target = Path(destination)
    if target.suffix.lower() != ".pdf":
        target = target / (digest + ".pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(body)
    except FileExistsError:
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise AcquisitionError("DESTINATION_EXISTS", "refusing to overwrite different content") from None
    return DownloadResult(
        candidate_id=candidate.id, path=str(target.resolve()), source_url=candidate.detail_url,
        pdf_url=final_url, sha256=digest, downloaded_at=datetime.now(UTC),
        disclosure_date=candidate.disclosure_date, report_year=candidate.report_year,
        stock_code=candidate.stock_code, revision=candidate.revision, size_bytes=len(body),
    )

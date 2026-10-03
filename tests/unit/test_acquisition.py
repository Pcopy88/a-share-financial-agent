from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def announcement_rows():
    return pd.DataFrame([
        {"代码": "300666", "简称": "样例", "公告标题": title, "公告时间": "2025-04-01 12:00:00",
         "公告链接": f"http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300666&announcementId={i}"}
        for i, title in enumerate(["2024年年度报告", "2024年年度报告（更正后）",
                                   "2024年年度报告摘要", "2023年年度报告",
                                   "关于2024年年度报告的更正公告", "2024年半年度报告"], 1)
    ])


def test_search_filters_content_year_summary_and_keeps_revised_full_report():
    from annual_report_agent.ingestion.acquisition import search_announcements

    def source(**kwargs):
        assert kwargs["symbol"] == "300666"
        assert kwargs["start_date"] <= "20250401" <= kwargs["end_date"]
        return announcement_rows()

    candidates = search_announcements("300666", 2024,
                                      ak_client=SimpleNamespace(stock_zh_a_disclosure_report_cninfo=source))
    assert len(candidates) == 2
    assert {item.revision for item in candidates} == {"original", "revised"}
    assert all(item.report_year == 2024 and item.disclosure_date.year == 2025 for item in candidates)
    assert all(item.pdf_url is None for item in candidates)
    assert all(item.detail_url.startswith("https://www.cninfo.com.cn/") for item in candidates)


def candidate(url="https://www.cninfo.com.cn/new/disclosure/detail?announcementId=1"):
    from annual_report_agent.ingestion.acquisition import AnnouncementCandidate

    return AnnouncementCandidate(id="candidate-1", stock_code="300666", company="样例",
                                 report_year=2024, title="2024年年度报告", detail_url=url,
                                 disclosure_date="2025-04-01", revision="original")


def public_resolver(host):
    return ["8.8.8.8"]


def test_download_needs_selection_and_does_not_treat_detail_html_as_pdf(tmp_path):
    from annual_report_agent.ingestion.acquisition import AcquisitionError, download_candidate

    with pytest.raises(AcquisitionError, match="SELECTION_REQUIRED"):
        download_candidate(candidate(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_selected_candidate_resolves_html_to_pdf_and_records_hash(tmp_path):
    from annual_report_agent.ingestion.acquisition import HTTPPayload, download_candidate

    body = b"%PDF-1.7\nsynthetic test bytes\n%%EOF"

    def transport(url, address, max_bytes):
        if "disclosure/detail" in url:
            return HTTPPayload(200, {}, b'<a href="https://static.cninfo.com.cn/finalpage/2025-04-01/1.PDF">PDF</a>')
        return HTTPPayload(200, {"content-type": "application/pdf"}, body)

    result = download_candidate(candidate(), tmp_path, selected=True,
                                transport=transport, resolver=public_resolver)
    assert Path(result.path).read_bytes() == body
    assert len(result.sha256) == 64
    assert result.size_bytes == len(body)
    assert result.source_url == candidate().detail_url
    assert result.pdf_url.endswith("1.PDF")


@pytest.mark.parametrize("url", [
    "file:///C:/secret.pdf", "http://static.cninfo.com.cn/a.pdf", "https://localhost/a.pdf",
    "https://127.0.0.1/a.pdf", "https://static.cninfo.com.cn.evil.example/a.pdf",
    "https://user:pass@static.cninfo.com.cn/a.pdf", "https://static.cninfo.com.cn:8443/a.pdf",
])
def test_unsafe_url_is_rejected_before_transport(url, tmp_path):
    from annual_report_agent.ingestion.acquisition import AcquisitionError, download_candidate

    def no_request(*args):
        pytest.fail("unsafe URL reached network transport")

    with pytest.raises(AcquisitionError, match="UNSAFE_URL"):
        download_candidate(candidate(url), tmp_path, selected=True,
                           transport=no_request, resolver=public_resolver)


def test_redirect_to_private_address_is_rejected_at_next_hop(tmp_path):
    from annual_report_agent.ingestion.acquisition import (
        AcquisitionError,
        HTTPPayload,
        download_candidate,
    )

    visited = []

    def transport(url, address, max_bytes):
        visited.append(url)
        return HTTPPayload(302, {"location": "https://static.cninfo.com.cn/a.pdf"}, b"")

    def resolver(host):
        return ["8.8.8.8"] if host == "www.cninfo.com.cn" else ["10.0.0.1"]

    with pytest.raises(AcquisitionError, match="UNSAFE_ADDRESS"):
        download_candidate(candidate(), tmp_path, selected=True, transport=transport, resolver=resolver)
    assert len(visited) == 1


@pytest.mark.parametrize(("body", "limit", "code"), [
    (b"<html>not a pdf</html>", 100, "INVALID_PDF"),
    (b"%PDF-" + b"x" * 30, 20, "DOWNLOAD_TOO_LARGE"),
])
def test_pdf_magic_and_size_are_enforced_without_partial_file(body, limit, code, tmp_path):
    from annual_report_agent.ingestion.acquisition import (
        AcquisitionError,
        HTTPPayload,
        download_candidate,
    )

    with pytest.raises(AcquisitionError, match=code):
        download_candidate(candidate("https://static.cninfo.com.cn/a.pdf"), tmp_path,
                           selected=True, max_bytes=limit, resolver=public_resolver,
                           transport=lambda *_: HTTPPayload(200, {}, body))
    assert not list(tmp_path.iterdir())


def test_unresolved_dynamic_detail_fails_explicitly_instead_of_guessing_pdf_url(tmp_path):
    from annual_report_agent.ingestion.acquisition import (
        AcquisitionError,
        HTTPPayload,
        download_candidate,
    )

    with pytest.raises(AcquisitionError, match="PDF_LINK_UNRESOLVED"):
        download_candidate(candidate(), tmp_path, selected=True, resolver=public_resolver,
                           transport=lambda *_: HTTPPayload(200, {}, b"<html>dynamic content</html>"))


def test_announcement_schema_drift_is_explicit():
    from annual_report_agent.ingestion.acquisition import AcquisitionError, search_announcements

    source = SimpleNamespace(stock_zh_a_disclosure_report_cninfo=lambda **_: pd.DataFrame([{"title": "changed"}]))
    with pytest.raises(AcquisitionError, match="FIELD_SCHEMA_CHANGED"):
        search_announcements("300666", 2024, ak_client=source)


def test_search_accepts_common_year_degree_title_without_duplicate_year_word():
    from annual_report_agent.ingestion.acquisition import search_announcements

    frame = announcement_rows().iloc[:1].copy()
    frame.loc[frame.index[0], "公告标题"] = "样例：2024年度报告"
    source = SimpleNamespace(stock_zh_a_disclosure_report_cninfo=lambda **_: frame)
    assert len(search_announcements("300666", 2024, ak_client=source)) == 1


def test_redirect_cycle_is_bounded(tmp_path):
    from annual_report_agent.ingestion.acquisition import (
        AcquisitionError,
        HTTPPayload,
        download_candidate,
    )

    requests = []

    def transport(url, *_):
        requests.append(url)
        return HTTPPayload(302, {"location": url}, b"")

    with pytest.raises(AcquisitionError, match="REDIRECT_LIMIT"):
        download_candidate(candidate(), tmp_path, selected=True, resolver=public_resolver, transport=transport)
    assert len(requests) == 4


def test_mixed_public_and_private_dns_answer_is_rejected(tmp_path):
    from annual_report_agent.ingestion.acquisition import AcquisitionError, download_candidate

    with pytest.raises(AcquisitionError, match="UNSAFE_ADDRESS"):
        download_candidate(candidate(), tmp_path, selected=True,
                           resolver=lambda _: ["8.8.8.8", "127.0.0.1"])


def test_download_does_not_overwrite_existing_different_content(tmp_path):
    from annual_report_agent.ingestion.acquisition import (
        AcquisitionError,
        HTTPPayload,
        download_candidate,
    )

    target = tmp_path / "existing.pdf"
    target.write_bytes(b"user-original")
    with pytest.raises(AcquisitionError, match="DESTINATION_EXISTS"):
        download_candidate(candidate("https://static.cninfo.com.cn/new.pdf"), target, selected=True,
                           resolver=public_resolver, transport=lambda *_: HTTPPayload(200, {}, b"%PDF-1.7 new"))
    assert target.read_bytes() == b"user-original"


def test_slow_stream_has_total_deadline(monkeypatch):
    import annual_report_agent.ingestion.acquisition as module

    class Response:
        status = 200

        def getheaders(self):
            return []

        def read(self, *_):
            return b"%PDF-1"  # An endless trickle must terminate before the size limit.

        read1 = read

    class Connection:
        def __init__(self, *_):
            pass

        def request(self, *_, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(module, "_PinnedHTTPSConnection", Connection)
    ticks = iter([0, 0, 61, 62, 63])
    monkeypatch.setattr(module, "_monotonic", lambda: next(ticks), raising=False)
    with pytest.raises(module.AcquisitionError, match="DOWNLOAD_TIMEOUT"):
        module._request("https://static.cninfo.com.cn/a.pdf", "8.8.8.8", 100)

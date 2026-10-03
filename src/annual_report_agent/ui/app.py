"""Four-page local UI. Every read/write goes through the API."""


import httpx
import streamlit as st

from annual_report_agent.config import Settings

st.set_page_config(page_title="财报资料分析", page_icon="📄", layout="wide")
STATUS = {"queued": "排队中", "running": "处理中", "succeeded": "完成", "failed": "失败",
          "interrupted": "已中断", "answered": "已回答", "partial": "部分完成",
          "insufficient_evidence": "资料不足", "needs_clarification": "需要补充问题条件",
          "unsupported": "暂不支持", "not_executed": "未执行", "ready": "可查询",
          "uploaded": "待确认", "metadata_confirmed": "待建库"}


def api(method, path, **kwargs):
    response = httpx.request(method, Settings().api_url.rstrip("/") + path, timeout=30, **kwargs)
    if response.is_error:
        try:
            data = response.json()
            error = data.get("error", data.get("detail", data))
        except ValueError:
            error = f"HTTP {response.status_code}"
        raise RuntimeError(str(error))
    return response.json()


def show_task():
    identifier = st.session_state.get("task_id")
    if not identifier:
        return
    task = api("GET", f"/api/tasks/{identifier}")
    status = task["status"]
    st.info(f"任务：{STATUS.get(status, status)}")
    with st.expander("任务编号"):
        st.code(identifier)
    if task.get("error_code"):
        st.error(f"{task['error_code']}：{task.get('error_message', task.get('message', ''))}")
    if st.button("刷新任务状态"):
        st.rerun()
    if task.get("resource_type") in {"analysis", "run", "report"}:
        st.session_state["run_id"] = task["resource_id"]


def remember_task(response):
    st.session_state["task_id"] = response["task_id"]
    if response.get("run_id"):
        st.session_state["run_id"] = response["run_id"]
    st.success("任务已提交。使用刷新按钮查看处理进度。")


def report_label(report):
    status = STATUS.get(report["status"], report["status"])
    return f"{report.get('company') or '待确认公司'} · {report['filename']} · {status}"


def report_selection(reports, key):
    options = {report["id"]: report_label(report) for report in reports}
    return st.multiselect("选择本次使用的资料", options, format_func=options.get, key=key)


def document_page(reports):
    st.header("资料管理")
    st.caption("先上传并核对资料身份，再解析建库。文件中的文字不会被当作系统指令。")
    with st.form("upload"):
        file = st.file_uploader("PDF 文件", type=["pdf"])
        upload = st.form_submit_button("上传")
    if upload:
        if file is None:
            st.warning("请先选择 PDF。")
        else:
            result = api("POST", "/api/reports/upload",
                         files={"file": (file.name, file.getvalue(), "application/pdf")})
            st.session_state["selected_report"] = result.get("report_id", result.get("id"))
            st.success("已上传，请核对下方信息。")
            st.rerun()
    if reports:
        by_id = {report["id"]: report for report in reports}
        selected = st.selectbox("已上传资料", by_id, format_func=lambda value: report_label(by_id[value]),
                                key="selected_report")
        report = by_id[selected]
        types = ["annual_report", "semiannual_report", "prospectus", "inquiry_letter",
                 "inquiry_response", "other"]
        names = ["年报", "半年报", "招股说明书", "问询函", "问询回复", "其他"]
        with st.form("metadata"):
            company = st.text_input("公司全称", value=report.get("company") or "")
            stock_code = st.text_input("股票代码（不确定可留空）", value=report.get("stock_code") or "")
            doc_type = st.selectbox("文档类型", types, format_func=lambda value: names[types.index(value)],
                index=types.index(report.get("doc_type", "other")))
            year = st.text_input("报告年度（例如 2026，不是披露日期）",
                                 value=str(report.get("fiscal_year") or ""))
            inquiry_round = st.text_input("问询轮次（不确定留空）", value=str(report.get("inquiry_round") or ""))
            revision = st.text_input("文件版本（例如 更正后；不确定留空）", value=report.get("revision") or "")
            original_periods = report.get("report_periods", [])
            first_period = original_periods[0] if original_periods else {}
            period_start = st.text_input("主报告期起始日期（例如 2026-01-01，未知留空）",
                value=first_period.get("period_start", first_period.get("start", "")) or "")
            period_end = st.text_input("主报告期结束日期（例如 2026-06-30，未知留空）",
                value=first_period.get("period_end", first_period.get("end", "")) or "")
            confirmed = st.checkbox("我已核对公司、文件类型和期间")
            save = st.form_submit_button("确认资料身份")
        if save:
            periods = list(original_periods)
            if period_start or period_end:
                from datetime import date
                date.fromisoformat(period_start)
                date.fromisoformat(period_end)
                period = {"period_start": period_start, "period_end": period_end}
                periods = [period, *periods[1:]]
            update = {"company": company or None, "stock_code": stock_code or None,
                "doc_type": doc_type, "fiscal_year": int(year) if year else None,
                "inquiry_round": int(inquiry_round) if inquiry_round else None,
                "revision": revision or None, "report_periods": periods,
                "confirmed": confirmed}
            api("PATCH", f"/api/reports/{selected}/metadata", json=update)
            st.success("身份已保存。")
        if st.button("解析并建立知识库", type="primary"):
            remember_task(api("POST", f"/api/reports/{selected}/prepare"))
        with st.expander("查看处理状态、失败页和能力范围"):
            st.json(api("GET", f"/api/reports/{selected}"))
    with st.expander("按股票代码获取全年年报"):
        with st.form("announcement-search"):
            code = st.text_input("沪深股票代码", max_chars=6)
            year = st.number_input("年报年度", 1990, 2100, 2024)
            search = st.form_submit_button("查询公告候选")
        if search:
            remember_task(api("POST", "/api/announcements/search", json={"stock_code": code, "year": year}))
        candidates = api("GET", "/api/announcements")["candidates"]
        if candidates:
            by_id = {item["id"]: item for item in candidates}
            candidate_id = st.selectbox("选择要下载的原版或修订版", by_id,
                format_func=lambda value: f"{by_id[value]['title']} · {by_id[value]['disclosure_date']}")
            st.caption(by_id[candidate_id]["detail_url"])
            if st.button("下载选中公告"):
                remember_task(api("POST", f"/api/announcements/{candidate_id}/download"))
    show_task()


def used_financial_sources(run, answer):
    """Read accepted, sent facts from this run, never the latest report fact list."""
    allowed_reports = set(run.get("request", {}).get("report_ids", []))
    accepted = {result["subtask_id"]: result for result in answer["subtask_results"]
                if result["status"] in {"answered", "partial"}}
    calculations = {item["id"]: item for item in answer.get("calculations", [])}
    shown = set()
    for trace in run.get("traces", []):
        result = accepted.get(trace.get("task", {}).get("subtask_id"))
        if result is None:
            continue
        used = {identifier for claim in result.get("claims", [])
                for identifier in claim.get("fact_ids", [])}
        for calculation in trace.get("calculations", []):
            if calculation["id"] in calculations:
                used.update(calculations[calculation["id"]].get("operand_fact_ids", []))
        for fact in trace.get("facts", []):
            identity = (fact["report_id"], fact["id"])
            if (fact["id"] not in used or fact.get("validation_status") != "verified"
                    or fact["report_id"] not in allowed_reports or identity in shown):
                continue
            provenance = fact.get("provenance", {})
            refs = list(provenance.get("page_refs", []))
            if not refs:
                pages = provenance.get("pdf_page_numbers") or [provenance.get("pdf_page_number")]
                refs = [{"pdf_page_number": page} for page in pages]
            refs = [ref for ref in refs if type(ref.get("pdf_page_number")) is int
                    and ref["pdf_page_number"] > 0]
            if not refs:
                sources = set(fact.get("source_evidence_ids", []))
                refs = [ref for evidence in trace.get("evidence", [])
                        if evidence["id"] in sources and evidence["report_id"] == fact["report_id"]
                        for ref in evidence.get("page_refs", [])
                        if type(ref.get("pdf_page_number")) is int and ref["pdf_page_number"] > 0]
            shown.add(identity)
            yield fact, refs


def show_answer(run):
    answer = run.get("answer")
    if not answer:
        st.info(f"分析状态：{run.get('status', '等待执行')}")
        return
    if answer.get("mode") == "mock":
        st.warning("模拟流程结果：用于验证功能连接，不代表 DeepSeek 或向量模型的真实效果。")
    level = {"high": "较完整", "medium": "一般", "low": "有限"}.get(answer["evidence_level"], "未知")
    st.caption(f"结果状态：{STATUS.get(answer['status'], answer['status'])} · 证据完整程度：{level}（不是正确率）")
    for result in answer["subtask_results"]:
        st.subheader(f"{result['subtask_id']} · {STATUS.get(result['status'], result['status'])}")
        for claim in result["claims"]:
            st.write(claim["text"])
            if claim["evidence_ids"]:
                st.caption("引用：" + "、".join(claim["evidence_ids"]))
        if result.get("reason"):
            st.info(result["reason"])
        if result["warnings"]:
            readable = [warning.split(": ", 1)[-1] for warning in result["warnings"]]
            st.caption("；".join(readable))
    if answer.get("calculations"):
        with st.expander("程序计算结果与操作数编号"):
            st.dataframe(answer["calculations"], width="stretch")
    sources = list(used_financial_sources(run, answer))
    if sources:
        with st.expander("核对财务原值与来源"):
            for fact, refs in sources:
                label = fact.get("provenance", {}).get("row_label") or fact["metric_code"]
                scope = {"consolidated": "合并报表", "parent": "母公司报表"}.get(fact["scope"], fact["scope"])
                period = (f"{fact['period_start']} 至 {fact['period_end']}" if fact.get("period_start")
                          else f"截至 {fact['period_end']}")
                st.text(f"{fact['company']} · {label}\n"
                        f"原值：{fact.get('raw_value') or '未记录'} {fact.get('raw_unit') or '原表单位未记录'}\n"
                        f"期间：{period} · 口径：{scope}")
                for page in dict.fromkeys(ref["pdf_page_number"] for ref in refs):
                    st.button(f"打开财务原值 PDF 第 {page} 页",
                              key=f"fact-{run['run_id']}-{fact['report_id']}-{fact['id']}-{page}",
                              on_click=open_evidence, args=(fact["report_id"], page))
    with st.expander("核对引用原文"):
        for evidence in answer["citations"]:
            st.markdown(f"**{evidence['id']} · {evidence['role']}**")
            st.text(evidence["quote"])
            for ref in evidence["page_refs"]:
                page = ref["pdf_page_number"]
                st.button(f"打开 PDF 第 {page} 页", key=f"{evidence['id']}-{page}",
                          on_click=open_evidence, args=(evidence["report_id"], page))


def open_evidence(report_id, page):
    st.session_state["evidence_report"] = report_id
    st.session_state["evidence_page"] = page
    st.session_state["page"] = "证据与指标"


def question_page(reports):
    st.header("分析问答")
    selected = report_selection(reports, "question_reports")
    profiles = api("GET", "/api/model-profiles")
    with st.form("question"):
        profile = st.selectbox("生成模型配置", profiles["profiles"],
                              index=list(profiles["profiles"]).index(profiles["default"]))
        question = st.text_area("要分析什么", placeholder="例如：2026年上半年收入增长多少？公司如何解释变化？")
        sent = st.form_submit_button("开始分析", type="primary")
    if sent:
        if not selected or not question.strip():
            st.warning("请选择资料并输入问题。")
        else:
            remember_task(api("POST", "/api/query", json={"question": question,
                "report_ids": selected, "model_profile": profile}))
    show_task()
    if st.session_state.get("run_id"):
        show_answer(api("GET", f"/api/runs/{st.session_state['run_id']}"))


def evidence_page(reports):
    st.header("证据与指标")
    if not reports:
        st.info("请先上传资料。")
        return
    by_id = {report["id"]: report for report in reports}
    identifier = st.selectbox("资料", by_id, format_func=lambda value: report_label(by_id[value]),
                              key="evidence_report")
    report = by_id[identifier]
    left, right = st.columns([3, 2])
    with left:
        page = st.number_input("PDF 物理页码", min_value=1, max_value=max(1, report.get("page_count", 1)),
                               key="evidence_page")
        response = httpx.get(f"{Settings().api_url}/api/reports/{identifier}/pages/{page}", timeout=30)
        if response.status_code == 200:
            st.image(response.content)
        else:
            st.info("该页尚不可读取，请先完成解析或查看处理状态。")
    with right:
        facts = api("GET", f"/api/reports/{identifier}/facts")["facts"]
        st.caption("只有 verified 的事实可进入财务计算；pending 不等于文件没有披露。")
        if facts:
            st.dataframe([{key: fact.get(key) for key in ("metric_code", "value", "unit", "period_start",
                "period_end", "scope", "validation_status")} for fact in facts], width="stretch")
            with st.expander("查看原值、来源与核验记录"):
                st.json(facts)
        else:
            st.info("尚无结构化财务事实，仍可使用文字检索。")
        if st.button("查询 AKShare 候选值并核对"):
            remember_task(api("POST", f"/api/reports/{identifier}/financial-import"))
    show_task()


def evaluation_page(reports):
    st.header("评测与报告")
    selected = report_selection(reports, "report_reports")
    if st.button("生成简版分析报告", type="primary"):
        if selected:
            remember_task(api("POST", "/api/reports/generate", json={"question": "生成资料分析报告",
                "report_ids": selected, "task_type": "report"}))
        else:
            st.warning("请选择资料。")
    run_id = st.session_state.get("run_id")
    if run_id:
        run = api("GET", f"/api/runs/{run_id}")
        if run.get("answer"):
            markdown = httpx.get(f"{Settings().api_url}/api/runs/{run_id}/export", timeout=30)
            if markdown.status_code == 200:
                st.download_button("下载 Markdown 报告", markdown.content,
                                   file_name=f"analysis-{run_id}.md", mime="text/markdown")
            show_answer(run)
    st.divider()
    registry = api("GET", "/api/evaluation-datasets")
    with st.form("evaluation"):
        dataset = st.selectbox("题集", registry["datasets"])
        experiment = st.selectbox("实验", ["R0", "R1", "R2", "R3", "M2", "M1", "Q0", "Q1", "C0", "C1"])
        split = st.selectbox("分组", ["development", "test"])
        frozen_run = st.text_input("冻结输入的评测编号（M1/Q0/Q1/C0/C1 必填）")
        repetitions = st.number_input("重复次数", 1, 3, 1)
        submitted = st.form_submit_button("运行评测")
    st.caption("模拟模式只验证评测程序。文字含义与结论支持需要独立复核；未标注完整证据不报告完整 Recall。")
    if submitted:
        remember_task(api("POST", "/api/evaluations", json={"dataset": dataset,
            "experiment": experiment, "split": split, "repetitions": repetitions,
            "frozen_run_id": frozen_run or None}))
    for evaluation in api("GET", "/api/evaluations")["evaluations"]:
        with st.expander(f"{evaluation['id']} · {evaluation.get('experiment')} · {evaluation.get('mode')}"):
            st.json(evaluation)
            rows = httpx.get(f"{Settings().api_url}/api/evaluations/{evaluation['id']}/rows", timeout=30)
            if rows.status_code == 200:
                st.download_button("下载逐题结果", rows.content, file_name=f"{evaluation['id']}.jsonl",
                                   key=evaluation["id"])
    show_task()


def main():
    st.title("财报资料分析")
    try:
        health = api("GET", "/api/health")
        mode = health.get("mode", "unknown")
        st.sidebar.caption(f"当前模式：{mode}")
        if mode == "mock":
            st.sidebar.warning("模拟模式：云端模型尚未参与回答。")
        page = st.sidebar.radio("功能", ["资料管理", "分析问答", "证据与指标", "评测与报告"], key="page")
        reports = api("GET", "/api/reports")["reports"]
        {"资料管理": document_page, "分析问答": question_page,
         "证据与指标": evidence_page, "评测与报告": evaluation_page}[page](reports)
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"操作未完成：{error}")
        st.info("请确认后端已启动；配置和任务错误可在后端记录中查看。")


if __name__ == "__main__":
    main()

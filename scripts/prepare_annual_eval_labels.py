"""Rebuild assistant-reviewed development labels, never held-out labels.

The manually transcribed cells below are checked against PDF text, independently of
the application extractor. They are evaluation references, never application facts
or retrieval input. Only selected pages received visual review; no human review is
claimed. All numeric expected results use the production Decimal calculator.
"""

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from annual_report_agent.finance.calculator import FORMULA_OPERANDS, calculate_metric
from annual_report_agent.schemas.finance import FinancialFact

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["current_assets", "total_assets", "current_liabilities", "total_liabilities",
           "revenue", "operating_cost", "net_profit", "parent_net_profit", "operating_cash_flow"]
LABELS = ["流动资产合计", "资产总计", "流动负债合计", "负债合计", "营业收入", "营业成本",
          "净利润", "归属于母公司股东的净利润", "经营活动产生的现金流量净额"]
# Raw cells in current/prior column order, manually transcribed from identified pages.
CELLS = {
    "moutai_2023": [
        "225,172,517,821.28|216,611,435,672.92", "272,699,660,092.25|254,500,826,096.02",
        "48,697,611,501.20|49,065,668,798.38", "49,043,190,797.43|49,562,744,832.16",
        "147,693,604,994.14|124,099,843,771.99", "11,867,273,851.78|10,093,468,616.63",
        "77,521,476,277.80|65,376,039,957.88", "74,734,071,550.75|62,717,467,870.12",
        "66,593,247,721.09|36,698,595,830.03"],
    "moutai_2024": [
        "251,726,674,636.66|225,172,517,821.28", "298,944,579,918.70|272,699,660,092.25",
        "56,515,990,618.96|48,697,611,501.20", "56,933,264,798.10|49,043,190,797.43",
        "170,899,152,276.34|147,693,604,994.14", "13,789,482,367.98|11,867,273,851.78",
        "89,334,728,025.90|77,521,476,277.80", "86,228,146,421.62|74,734,071,550.75",
        "92,463,692,168.43|66,593,247,721.09"],
    "gree_2023": [
        "228,141,309,470.74|255,140,038,972.46", "368,053,902,576.37|355,070,769,579.70",
        "200,983,299,755.92|216,371,936,815.59", "247,407,749,159.93|253,194,721,565.51",
        "203,979,266,387.09|188,988,382,706.68", "141,625,549,746.95|139,784,387,882.78",
        "27,719,022,913.59|23,011,344,353.11", "29,017,387,604.18|24,506,623,782.46",
        "56,398,426,354.17|28,668,435,921.27"],
    "gree_2024": [
        "224,802,908,651.88|228,141,309,470.74", "368,031,704,522.86|368,053,902,576.37",
        "201,125,441,381.86|200,983,299,755.92", "226,518,009,574.89|247,407,749,159.93",
        "189,163,654,064.64|203,979,266,387.09", "133,496,119,623.36|143,953,487,220.05",
        "32,371,069,287.35|27,719,022,913.59", "32,184,570,372.28|29,017,387,604.18",
        "29,369,250,570.66|56,398,426,354.17"],
}
PAGE_NUMBERS = {"moutai_2023": [59,59,60,60,63,63,64,64,67],
                "moutai_2024": [59,59,60,60,63,63,64,64,67],
                "gree_2023": [114,114,115,115,116,116,116,116,117],
                "gree_2024": [111,111,112,112,113,113,113,113,114]}
VISUAL_PAGES = {"moutai_2023": {63}, "moutai_2024": {63,64},
                "gree_2023": {116}, "gree_2024": {111,113}}


def compact(value):
    return re.sub(r"\s+", "", value)


def make_reference(sample, pages):
    sid, year = sample["id"], sample["year"]
    facts, formula_facts = [], {}
    for i, (metric, label, pair, page) in enumerate(zip(
            METRICS, LABELS, CELLS[sid], PAGE_NUMBERS[sid], strict=True)):
        for prior, raw in enumerate(pair.split("|")):
            table = "合并资产负债表" if i < 4 else "合并现金流量表" if i == 8 else "合并利润表"
            start = None if i < 4 else f"{year-prior}-01-01"
            # Preserve the actual beginning-of-year column, rather than rewriting it.
            end = f"{year}-01-01" if i < 4 and prior and sid.startswith("gree") else f"{year-prior}-12-31"
            column = end.replace("-", "年", 1).replace("-", "月", 1) + "日" if i < 4 else f"{year-prior}年度"
            column = re.sub(r"月0([1-9])日", r"月\1日", column).replace("年01月", "年1月")
            assert raw in pages[page-1], (sid, metric, raw, page)
            # Inspect the original row and require the intended label nearby as well.
            text = compact(pages[page-1])
            where = text.index(raw)
            assert label in text[max(0, where-120):where], (sid, metric, page)
            fact_id = f"{metric}_{'previous' if prior else 'current'}"
            restated = sid == "gree_2024" and metric == "operating_cost" and prior
            item = {
                "id": fact_id, "metric_code": metric, "company": sample["company"],
                "row_label": label, "value": str(Decimal(raw.replace(',', ''))), "raw_value": raw,
                "currency": "CNY", "unit": "元", "raw_unit": "元", "scope": "consolidated",
                "period_kind": "instant" if i < 4 else "duration", "period_start": start,
                "period_end": end, "period_label": end if i < 4 else str(year-prior), "report_id": sid,
                "restatement_status": "restated" if restated else None,
                "source": {"table": table, "pdf_page_number": page, "printed_page_label": str(page),
                           "column_label": column, "raw_value_quote": raw},
                "reference_review_status": "text_and_selected_page_visual_checked_by_assistant",
                "text_verified": True, "visual_verified": page in VISUAL_PAGES[sid],
                "application_validation_status": "not_run", "human_reviewed": False,
            }
            if restated:
                item["restatement_evidence"] = {"pdf_page_number": 152,
                    "note": "解释第18号保证类质量保证由销售费用转入营业成本；2023年营业成本追溯增加2,327,937,473.10元。"}
            facts.append(item)
            # Ephemeral verified operands only to calculate an offline reference result.
            # These objects are not persisted as application FinancialFacts.
            formula_facts[(metric, prior)] = FinancialFact(
                id=f"reference:{sid}:{fact_id}", metric_code=metric, value=Decimal(item["value"]),
                raw_value=raw, raw_unit="元", company=sample["company"], period_start=start,
                period_end=end, scope="consolidated", report_id=sid, source_id=f"reference:{sid}:page:{page}",
                source_evidence_ids=[f"reference:{sid}:page:{page}"], validation_status="verified",
                restatement_status=item["restatement_status"],
                review_record={"method": "offline_assistant_reference_only", "human_reviewed": False},
            )
    calculations = {}
    for fid, mapping in FORMULA_OPERANDS.items():
        operands = {key: formula_facts[(metric, int(key.startswith("previous_")))]
                    for key, metric in mapping.items()}
        calculations[fid] = calculate_metric(fid, operands).model_dump(mode="json")
    return {
        "schema_version": "1.0", "purpose": "development_reference_only", "report_id": sid,
        "company": sample["company"], "source_sha256": sample["sha256"], "source_path": sample["path"],
        "source_url": sample["url"], "human_reviewed": False,
        "source_boundary": "官方原PDF文本及指定页面图像由assistant核对；不是人工复核、模型评测通过或自动财务提取准确率证明。",
        "scope_boundary": "覆盖八个白名单公式需要的九项合并财务科目及比较列，不代表已标注所有会计科目。",
        "period_rules": ["利润及现金流均为全年期间；资产负债保留原列时点。",
                         "同比使用同一报告内比较列；报告间重述差异不得自动覆盖。",
                         "营业收入与营业总收入不同；净利润与归母净利润不同。"],
        "facts": facts, "reference_calculations": calculations,
        "visual_reviewed_pdf_pages": sorted(VISUAL_PAGES[sid]),
    }


def build_development():
    manifest_path = ROOT / "evals/annual_sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = {s["id"]: s for s in manifest["samples"] if s["split"] == "development"}
    assert set(samples) == set(CELLS)  # No held-out company may enter this generator.
    texts = {sid: json.loads((ROOT / "tmp/annual_eval" / sid / "pages.json").read_text(encoding="utf-8"))
             for sid in samples}
    references = {}
    for sid, sample in samples.items():
        reference = make_reference(sample, texts[sid])
        references[sid] = reference
        (ROOT / "evals" / f"{sid}_reference.json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    cases = []

    def add(sid, category, question, points, quotes=(), formulas=(), status="answered", forbidden=()):
        sample = samples[sid]
        evidence = []
        for page, quote in quotes:
            assert compact(quote) in compact(texts[sid][page-1]), (sid, page, quote)
            evidence.append({"report_id": sid, "pdf_page_number": page, "printed_page_label": str(page), "quote": quote})
        expected = []
        for formula in formulas:
            calc = references[sid]["reference_calculations"][formula]
            expected.append(calc)
            for fact_id in calc["operand_fact_ids"]:
                label_id = fact_id.split(":")[-1]
                fact = next(f for f in references[sid]["facts"] if f["id"] == label_id)
                evidence.append({"report_id": sid, "pdf_page_number": fact["source"]["pdf_page_number"],
                                 "quote": fact["raw_value"], "metric_code": fact["metric_code"],
                                 "reference_fact_id": label_id})
        evidence = list({json.dumps(e, sort_keys=True, ensure_ascii=False): e for e in evidence}.values())
        route = "data" if category == "boundary" else category
        case = {
            "id": f"ANNUAL-DEV-{len(cases)+1:02d}", "question": question, "report_ids": [sid],
            "split": "development", "category": category, "route": route,
            "filters": {"company": sample["company"], "doc_type": "annual_report"},
            "gold_evidence": evidence, "gold_complete": False,
            "gold_scope": "核对的必要证据子集，未穷举同义段落或重复表格；统计coverage不称完整Recall。",
            "expected_status": status, "expected_points": points, "expected_calculations": expected,
            "expected_behavior": "保持指定报告、期间、合并口径与单位；仅据原文解释，计算引用白名单公式与实际证据。",
            "label_review": {"reviewer": "assistant", "human_reviewed": False, "review_date": "2026-10-03"},
            "source_verified": {"text_verified": True, "method": "逐页文本、原金额及quote连续匹配；selected pages视觉核对",
                                "visual_verified_pages": sorted({e['pdf_page_number'] for e in evidence} & VISUAL_PAGES[sid])},
            "run_status": "not_run", "forbidden_claims": list(forbidden),
            "numeric_display": {"decimal_places": 2, "rounding": "ROUND_HALF_UP", "absolute_tolerance": "0.005"},
        }
        if category == "hybrid":
            # Retrieval queries must come from the question, never from answer quotes.
            semantic_question = question.split("并", 1)[-1]
            case["canonical_subtasks"] = [{"query_text": f"{sample['company']}{sample['year']}年报 {semantic_question}",
                                           "route": "semantic"}]
        else:
            case["canonical_query"] = question
        cases.append(case)

    add("moutai_2023", "semantic", "贵州茅台2023年报中，直销与批发代理分别包括哪些销售渠道？",
        ["直销包括自营和i茅台等数字营销平台。", "批发代理包括社会经销商、商超、电商等。"],
        [(7, "直销渠道指自营和“i茅台”等数字营销平台渠道，批发代理渠道指社会经销商、商超、电商等渠道。")])
    add("moutai_2023", "semantic", "贵州茅台2023年报披露的高粱和小麦采购模式有什么区别？",
        ["高粱采用公司+地方政府+供应商+合作社或农户。", "小麦采用公司+供应商+合作社或农场。"],
        [(7, "茅台酒用高粱采取“公司+地方政府+供应商+合作社或农户”的模式，小麦采取“公司+供应商+合作社或农场”的模式")])
    add("moutai_2024", "semantic", "贵州茅台2024年报如何概括五大核心竞争力，报告期是否发生重大变化？",
        ["品质、品牌、工艺、环境、文化。", "报告期未发生重大变化。"],
        [(8, "拥有“品质、品牌、工艺、环境、文化”组成的“五大核心竞争力”。报告期内公司核心竞争力未发生重大变化。")])
    add("moutai_2024", "semantic", "按贵州茅台2024年报分产品表，茅台酒和其他系列酒披露的毛利率各是多少？",
        ["茅台酒94.06%；其他系列酒79.87%。", "这是分产品表的已披露数，不是集团整体毛利率。"],
        [(9, "茅台酒 145,928,075,955.31 8,662,079,388.78 94.06"),
         (9, "24,683,762,096.71 4,967,916,424.11 79.87")],
        forbidden=["不能把产品口径毛利率当作合并整体毛利率。"])
    add("gree_2023", "semantic", "格力电器2023年报所述新零售渠道如何结合线上和线下？",
        ["线上线下双线融合，以30个区域、超3万家网点及第三方电商官方旗舰店为基础。", "以数字化打通采购、物流、销售和服务。"],
        [(12, "以30个区域、超3万家线下网点、第三方电商平台官方旗舰店为基础，建立覆盖全国的双线销售网络。"),
         (12, "通过数字化信息系统建设，打通采购、物流、销售、服务各环节")])
    add("gree_2023", "semantic", "格力电器2023年报认为宏观经济波动如何影响其产品销售？",
        ["影响消费者收入和支出，也影响企业战略收缩和扩展的决策，进而影响消费品和工业制品销售。"],
        [(39, "公司产品覆盖消费品和工业制品两大领域，宏观经济变化影响消费者收入和支出的变化，影响企业战略收缩和扩展的决策，会对公司产品销售产生影响。")])
    add("gree_2024", "semantic", "格力电器2024年报提到的全新渠道战略品牌是什么，有什么目的？",
        ["董明珠健康家。", "整合多品类家电、满足多样化需求，提供家居生活解决方案。"],
        [(12, "推出全新渠道战略品牌“董明珠健康家”，整合多品类家电产品，满足消费者多样化需求，为消费者提供全面优质的家居生活解决方案。")])
    add("gree_2024", "semantic", "格力电器2024年报为何调整2023年的营业成本和销售费用？",
        ["解释第18号：不属于单项履约义务的保证类质量保证，由销售费用转入营业成本并追溯调整。", "2023年营业成本增加2,327,937,473.10元，销售费用等额减少。"],
        [(77, "本公司计提不属于单项履约义务的保证类质量保证原计入“销售费用”"),
         (77, "列报于利润表“营业成本”项目中，并进行追溯调整。"),
         (77, "销售费用 -2,327,937,473.10"), (77, "营业成本 2,327,937,473.10")],
        forbidden=["不得描述成2023年实际销售活动发生了新增成本或原始报表造假。"])
    add("moutai_2023", "data", "按贵州茅台2023年报合并利润表，2023年营业收入同比增长多少？保留两位小数。",
        ["使用营业收入，而不是营业总收入；分母为同表2022年度比较列。"], formulas=["revenue_yoy"])
    add("moutai_2024", "data", "按贵州茅台2024年报合并口径，2024年归母净利润同比增长多少？",
        ["归母净利润与合并净利润应区分。"], formulas=["parent_net_profit_yoy"])
    add("moutai_2024", "data", "按贵州茅台2024年报合并利润表，净利率（净利润除以营业收入）是多少？",
        ["使用包含少数股东损益的合并净利润，结果为百分数。"], formulas=["net_margin"])
    add("gree_2023", "data", "按格力电器2023年报合并口径，2023年末资产负债率和流动比率分别是多少？",
        ["两个比率均使用2023年12月31日時点；流动比率为倍数。"], formulas=["debt_asset_ratio", "current_ratio"])
    add("gree_2024", "data", "按格力电器2024年报的可比合并利润表，2024年毛利率和毛利率同比百分点变化分别是多少？保留四位小数。",
        ["用2024年报内经追溯调整的2023年成本比较列。", "毛利率变化用百分点，不用增长百分比。"],
        formulas=["gross_margin", "gross_margin_yoy_pp"], forbidden=["不得使用2023年报原列营业成本141,625,549,746.95。"])
    cases[-1]["numeric_display"] = {"decimal_places": 4, "rounding": "ROUND_HALF_UP", "absolute_tolerance": "0.00005"}
    add("gree_2024", "data", "按格力电器2024年报合并口径，2024年经营现金流收入比是多少？",
        ["经营活动现金流量净额除以同年营业收入，结果以百分数列示。"], formulas=["operating_cash_flow_to_revenue"])
    add("moutai_2023", "hybrid", "计算贵州茅台2023年合并毛利率，并说明年报对营业成本变动的解释。",
        ["按合并营业收入和营业成本计算整体毛利率。", "公司披露销量增加、生产成本增加及产品结构变化。"],
        [(9, "营业成本变动原因说明：主要是本期销量增加、生产成本增加及产品结构变化。")], ["gross_margin"])
    add("moutai_2024", "hybrid", "计算贵州茅台2024年营业收入同比，并结合年报说明收入增长原因。",
        ["营业收入同比采用2024年报同表比较列。", "披露原因是销量增加及茅台酒主要产品销售价格调整。"],
        [(8, "营业收入变动原因说明：主要是本期销量增加及茅台酒主要产品销售价格调整。")], ["revenue_yoy"])
    add("gree_2023", "hybrid", "计算格力电器2023年经营现金流收入比，并说明年报对经营现金流同比变动的解释。",
        ["经营现金流收入比使用全年合并口径。", "同比增长主要因为销售商品、提供劳务收到的现金增加。"],
        [(28, "经营活动产生的现金流量净额同比变动96.73%，主要系销售商品、提供劳务收到的现金增加所致。")], ["operating_cash_flow_to_revenue"])
    add("gree_2024", "hybrid", "计算格力电器2024年合并净利率，并说明经营现金流同比下降的已披露原因。",
        ["净利率使用合并净利润除以营业收入。", "经营现金流同比下降主要系渠道模式及市场变化等因素；不能据此断言舞弊或利润虚假。"],
        [(35, "经营活动产生的现金流量净额同比变动-47.93%，主要系渠道模式及市场变化等因素所致。")], ["net_margin"])
    add("moutai_2024", "boundary", "只依据这份贵州茅台2024年报，计算2025年全年营业收入同比。",
        ["2025全年不在所选报告已披露全年期间内，应明确缺少2025全年营业收入，不能预测充当实际数。"],
        status="insufficient_evidence", forbidden=["不得使用2024数字充当2025。", "不得隐式获取其他报告扩大用户范围。"])
    add("gree_2024", "boundary", "计算格力电器2024年的EBITDA利润率，并给出你采用的公式。",
        ["EBITDA利润率不在当前八个受控公式范围，应标明不支持，不能临时发明公式或把净利率代替EBITDA利润率。"],
        status="unsupported", forbidden=["不得调用任意模型生成代码。", "不得换成净利率后冒称EBITDA利润率。"])
    assert Counter(c["category"] for c in cases) == {"semantic": 8, "data": 6, "hybrid": 4, "boundary": 2}
    assert len(cases) == 20
    (ROOT / "evals/annual_development_v1.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    manifest["labeling"]["development_prepared"] = 20
    manifest["labeling"]["test_prepared"] = 0
    for s in manifest["samples"]:
        if s["id"] in samples:
            s["label_status"] = "assistant_development_reference_prepared"
            s["reference_path"] = f"evals/{s['id']}_reference.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"development_cases": len(cases), "distribution": dict(Counter(c['category'] for c in cases)),
                      "reference_facts": sum(len(r['facts']) for r in references.values()),
                      "formulas": sorted(FORMULA_OPERANDS), "heldout_labels_prepared": 0}, ensure_ascii=False))


if __name__ == "__main__":
    build_development()

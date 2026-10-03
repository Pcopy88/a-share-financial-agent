"""Prepare held-out labels only after rules_freeze_v1.json; never modify rules.

Independent label source: manually transcribed original PDF cells and quotations.
Application parsing is run only after labels are fixed, for an observed comparison.
No reference input is supplied to the parser. Reference-backed calculator evaluation
is an assisted-input track, not an end-to-end extraction score.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from annual_report_agent.finance.calculator import FORMULA_OPERANDS, calculate_metric
from annual_report_agent.finance.pdf_facts import extract_pdf_facts, verify_against_reference
from annual_report_agent.ingestion.parser import PARSER_VERSION, parse_report
from annual_report_agent.schemas.finance import FinancialFact
from annual_report_agent.schemas.report import Report

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["current_assets", "total_assets", "current_liabilities", "total_liabilities",
           "revenue", "operating_cost", "net_profit", "parent_net_profit", "operating_cash_flow"]
LABELS = ["流动资产合计", "资产总计", "流动负债合计", "负债合计", "营业收入", "营业成本",
          "净利润", "归属于母公司股东的净利润", "经营活动产生的现金流量净额"]
CELLS = {
    "fuyao_2023": [
        "33,536,371,292|29,677,907,370", "56,630,407,495|50,767,498,791",
        "15,102,799,113|17,360,373,707", "25,219,085,168|21,779,200,663",
        "33,160,996,641|28,098,754,166", "21,424,326,140|18,535,445,134",
        "5,629,108,273|4,752,795,645", "5,629,256,054|4,755,595,541",
        "7,624,580,890|5,893,041,655"],
    "fuyao_2024": [
        "36,403,083,788|33,536,371,292", "63,254,182,536|56,630,407,495",
        "17,988,933,782|15,102,799,113", "27,568,613,358|25,219,085,168",
        "39,251,657,267|33,160,996,641", "25,030,877,441|21,424,326,140",
        "7,504,038,370|5,629,108,273", "7,497,976,123|5,629,256,054",
        "8,562,187,329|7,624,580,890"],
}
PAGES = {"fuyao_2023": [84,85,85,85,88,88,89,89,91],
         "fuyao_2024": [89,90,90,90,93,93,94,94,97]}
VISUAL_PAGES = {"fuyao_2023": {88,89}, "fuyao_2024": {93,94}}


def compact(text):
    return re.sub(r"\s+", "", text)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rule_differences(freeze):
    return [{"path": path, "frozen_sha256": expected, "observed_sha256": sha((ROOT / path).read_bytes())}
            for path, expected in freeze["files"].items() if sha((ROOT / path).read_bytes()) != expected]


def source_pages(sample):
    """Keep ignored text caches source-bound; a fresh checkout can rebuild them."""
    import pdfplumber

    path = Path(sample["path"])
    assert sha(path.read_bytes()) == sample["sha256"], "source PDF hash changed"
    folder = ROOT / "tmp/annual_eval" / sample["id"]
    cache, identity = folder / "pages.json", folder / "source_sha256.txt"
    if not cache.exists() or not identity.exists() or identity.read_text() != sample["sha256"]:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        folder.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        identity.write_text(sample["sha256"], encoding="ascii")
    return json.loads(cache.read_text(encoding="utf-8"))


def reference(sample, pages, freeze_sha):
    sid, year = sample["id"], sample["year"]
    facts, calculation_inputs = [], {}
    for i, (metric, label, pair, page) in enumerate(zip(METRICS, LABELS, CELLS[sid], PAGES[sid], strict=True)):
        for prior, raw in enumerate(pair.split("|")):
            text = compact(pages[page-1])
            positions = [match.start() for match in re.finditer(re.escape(raw), text)]
            assert any(label in text[max(0,position-120):position] for position in positions), (sid, metric, page)
            start = None if i < 4 else f"{year-prior}-01-01"
            end = f"{year-prior}-12-31"
            column = f"{year-prior}年12月31日" if i < 4 else f"{year-prior}年度"
            table = "合并资产负债表" if i < 4 else "合并现金流量表" if i == 8 else "合并利润表"
            fact_id = f"{metric}_{'previous' if prior else 'current'}"
            item = {"id": fact_id, "metric_code": metric, "company": sample["company"], "row_label": label,
                    "value": str(Decimal(raw.replace(',', ''))), "raw_value": raw,
                    "currency": "CNY", "unit": "元", "raw_unit": "元", "scope": "consolidated",
                    "period_kind": "instant" if i < 4 else "duration", "period_start": start,
                    "period_end": end, "period_label": end if i < 4 else str(year-prior), "report_id": sid,
                    "source": {"table": table, "pdf_page_number": page, "printed_page_label": str(page),
                               "column_label": column, "raw_value_quote": raw},
                    "reference_review_status": "assistant_text_checked_selected_pages_visually_checked",
                    "text_verified": True, "visual_verified": page in VISUAL_PAGES[sid],
                    "human_reviewed": False, "application_validation_status": "not_run"}
            facts.append(item)
            # Offline expected-answer inputs only. Never published to the application.
            calculation_inputs[(metric,prior)] = FinancialFact(
                id=f"reference:{sid}:{fact_id}", metric_code=metric, value=Decimal(item["value"]),
                raw_value=raw, raw_unit="元", company=sample["company"], period_start=start,
                period_end=end, scope="consolidated", report_id=sid, source_id=f"reference:{sid}:page:{page}",
                source_evidence_ids=[f"reference:{sid}:page:{page}"], validation_status="verified",
                review_record={"method": "offline_assistant_reference_only", "human_reviewed": False})
    calculations = {}
    for fid, mapping in FORMULA_OPERANDS.items():
        operands = {key: calculation_inputs[(metric,int(key.startswith('previous_')))] for key,metric in mapping.items()}
        calculations[fid] = calculate_metric(fid,operands).model_dump(mode="json")
    return {"schema_version": "1.0", "purpose": "heldout_reference_only", "report_id": sid,
            "company": sample["company"], "source_sha256": sample["sha256"], "source_path": sample["path"],
            "source_url": sample["url"], "human_reviewed": False, "rules_freeze_sha256": freeze_sha,
            "source_boundary": "assistant核对原PDF文本及指定图像；未人工复核，不等于模型验收通过。",
            "evaluation_boundary": "reference_calculations基于assistant已核验事实输入，不能作为纯自动抽取成绩。应用须先抽取，再独立比较；不得用reference补造缺失事实。",
            "scope_boundary": "仅八个白名单公式所需的九个科目及其比较列；并非完整会计科目gold。",
            "period_rules": ["同比使用同一份年报内全年比较列。", "资产负债使用原表明确时点；不和流量期间混用。",
                             "归母净利润与合并净利润分别保存。"],
            "facts": facts, "reference_calculations": calculations,
            "visual_reviewed_pdf_pages": sorted(VISUAL_PAGES[sid])}


def prepare():
    freeze_path = ROOT / "evals/rules_freeze_v1.json"
    freeze_bytes = freeze_path.read_bytes()  # Absence is an error, never label pre-freeze.
    freeze, freeze_sha = json.loads(freeze_bytes), sha(freeze_bytes)
    before_rules = rule_differences(freeze)
    developer_file = ROOT / "evals/annual_development_v1.jsonl"
    developer_bytes = developer_file.read_bytes()
    developer_sha = sha(developer_bytes)
    manifest_path = ROOT / "evals/annual_sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = {s["id"]: s for s in manifest["samples"] if s["split"] == "test"}
    assert set(samples) == set(CELLS)
    texts = {sid: source_pages(sample) for sid,sample in samples.items()}
    references = {sid: reference(sample,texts[sid],freeze_sha) for sid,sample in samples.items()}
    cases = []

    def add(sid, category, question, points, quotes=(), formulas=(), status="answered", forbidden=()):
        sample = samples[sid]
        evidence = []
        for page,quote in quotes:
            assert compact(quote) in compact(texts[sid][page-1]), (sid,page,quote)
            evidence.append({"report_id":sid,"pdf_page_number":page,"printed_page_label":str(page),"quote":quote})
        expected = []
        for formula in formulas:
            calc = references[sid]["reference_calculations"][formula]
            expected.append(calc)
            for fact_id in calc["operand_fact_ids"]:
                label_id = fact_id.split(":")[-1]
                fact = next(f for f in references[sid]["facts"] if f["id"] == label_id)
                evidence.append({"report_id":sid,"pdf_page_number":fact["source"]["pdf_page_number"],
                                 "quote":fact["raw_value"],"metric_code":fact["metric_code"],"reference_fact_id":label_id})
        evidence = list({json.dumps(e,sort_keys=True,ensure_ascii=False):e for e in evidence}.values())
        for e in evidence:
            assert compact(e["quote"]) in compact(texts[sid][e["pdf_page_number"]-1])
        case = {"id":f"ANNUAL-TEST-{len(cases)+1:02d}","question":question,"report_ids":[sid],"split":"test",
                "category":category,"route":"data" if category=="boundary" else category,
                "filters":{"company":sample["company"],"doc_type":"annual_report"},
                "gold_evidence":evidence,"gold_complete":False,
                "gold_scope":"必要证据子集；未穷举重复披露，不能作为完整Recall的gold集合。",
                "expected_status":status,"expected_points":points,"expected_calculations":expected,
                "expected_behavior":"保持指定报告、合并口径、期间和单位；解释只来自所选原文。",
                "label_review":{"reviewer":"assistant","human_reviewed":False,"review_date":"2026-10-03"},
                "source_verified":{"text_verified":True,"method":"quote及原值逐页去空白连续匹配；选定页视觉核对",
                    "visual_verified_pages":sorted({e['pdf_page_number'] for e in evidence}&VISUAL_PAGES[sid])},
                "rules_freeze_sha256":freeze_sha,"run_status":"not_run","forbidden_claims":list(forbidden),
                "numeric_display":{"decimal_places":2,"rounding":"ROUND_HALF_UP","absolute_tolerance":"0.005"},
                "financial_input_boundary":"参考答案来自已核验事实输入；应用自动抽取表现需单独统计，不能用标签补齐事实。"}
        if category=="hybrid":
            case["canonical_subtasks"]=[{"query_text":f"{sample['company']}{sample['year']}年报 {question.split('并',1)[-1]}","route":"semantic"}]
        else:
            case["canonical_query"]=question
        cases.append(case)

    add("fuyao_2023","semantic","福耀玻璃2023年报把什么视为最核心的竞争力？",
        ["品牌。"],[(22,"品牌是福耀最核心的竞争力。")])
    add("fuyao_2023","semantic","福耀玻璃2023年报提及哪些管理基础支持数字化和智能化？",
        ["规范、透明、国际化的财务体系，以及基于ERP的流程优化系统。"],
        [(22,"福耀规范、透明、国际化的财务体系和基于ERP的流程优化系统")])
    add("fuyao_2023","semantic","福耀玻璃2023年报提出如何管理汇率波动风险？请列出两类措施。",
        ["监控外币交易及资产负债，优化出口结算币种或使用汇率金融工具。", "在欧美当地生产与销售。"],
        [(14,"加强监控集团外币交易以及外币资产和负债的规模"),
         (14,"优化出口结算币种以及运用汇率金融工具"),
         (14,"公司通过在欧美地区建厂，在当地生产与销售")])
    add("fuyao_2023","semantic","福耀玻璃2023年报在成本波动风险中，分别列出汽车玻璃和浮法玻璃的哪些主要成本？",
        ["汽车玻璃包括浮法玻璃原料、PVB、人工、电力及制造费用。", "浮法玻璃包括纯碱、天然气、人工、电力及制造费用。"],
        [(14,"汽车玻璃成本构成主要为浮法玻璃原料、PVB原料、人工、电力及制造费用"),
         (14,"浮法玻璃成本主要为纯碱原料、天然气燃料、人工、电力及制造费用。")])
    add("fuyao_2023","semantic","福耀玻璃2023年报如何从管理和技术两方面控制数据安全风险？",
        ["管理上规范数据分类分级、控制分发范围。", "技术上对存储、处理、流转和销毁实施加密等全生命周期控制。"],
        [(15,"规范数据分类分级，严格控制数据分发范围"),
         (15,"通过加密等控制措施对数据存储、处理、流转、销毁等全生命周期进行管控")])
    add("fuyao_2023","semantic","福耀玻璃2023年报拟订的现金股利按什么币种计值，A股与H股分别用什么币种支付？",
        ["人民币计值和宣布；A股以人民币支付，H股以港币支付。", "这是年报所述拟订方案，不是实际付款已完成的证明。"],
        [(15,"本公司派发的现金股利以人民币计值和宣布，以人民币向A股股东支付，以港币向H股股东支付。")])
    add("fuyao_2024","semantic","福耀玻璃2024年报列举了哪些高附加值汽车玻璃？至少给出三项。",
        ["例如智能全景天幕、可调光、抬头显示、超隔绝、轻量化超薄、镀膜可加热或钢化夹层玻璃。"],
        [(21,"智能全景天幕玻璃、可调光玻璃、抬头显示玻璃、超隔绝玻璃、轻量化超薄玻璃、镀膜可加热玻璃、钢化夹层玻璃")])
    add("fuyao_2024","semantic","福耀玻璃2024年报如何说明提升全球保供能力的措施？",
        ["提前规划全球供应链，持续优化生产过程、缩短投料到进仓时间，构建柔性敏捷生产能力。"],
        [(21,"根据行业趋势提前规划布局，构建完整的全球供应链体系"),
         (21,"缩短产品从投料到进仓的时间，构建柔性敏捷生产能力")])
    add("fuyao_2024","semantic","福耀玻璃2024年报如何描述汽车智能化下玻璃、饰件的角色变化？",
        ["从遮风挡雨，转为智能座舱组成部分以及与外界感知、交互的平台。"],
        [(21,"汽车玻璃、饰件等从原来的遮风挡雨，变成智能座舱的组成部分，成为与外界感知与交互的平台")])
    add("fuyao_2024","semantic","福耀玻璃2024年报产销量表中，汽车玻璃和浮法玻璃的数量单位分别是什么？汽车玻璃销量是多少？",
        ["汽车玻璃用百万平方米，浮法玻璃用万吨；汽车玻璃销量为155.87百万平方米。"],
        [(24,"汽车玻璃 百万平方米 162.00 155.87 19.24"), (24,"浮法玻璃 万吨 172.89 170.47 27.93")],
        forbidden=["不得把生产量162.00误报为销量，也不得把百万平方米改成万平方米。"])
    add("fuyao_2024","semantic","福耀玻璃2024年报解释销售费用减少时，提到包装物成本发生了什么处理变化？",
        ["因包装物精细化、专用化需求，相关包装材料作为履约成本计入营业成本。"],
        [(23,"包装物精细化、专用化需求趋势，加强对包装材料的质量管理及包装要求，本年度将其作为履约成本计入营业成本。")],
        forbidden=["不得断言减少额全部来自真实费用节省或擅自重述以前年度数字。"])
    add("fuyao_2024","semantic","福耀玻璃2024年报称其产业生态包含哪些环节？至少列出四项。",
        ["砂矿资源、优质浮法技术、工艺设备研发制造、多功能集成玻璃、铝饰件、全球R&D中心和供应链网络。"],
        [(22,"砂矿资源、优质浮法技术、工艺设备研发制造、多功能集成玻璃、延伸到铝饰件产业、全球布局的R&D中心和供应链网络")])
    for sid,question,formula,point in [
        ("fuyao_2023","按福耀玻璃2023年报，2023年合并营业收入同比是多少？","revenue_yoy","同比采用2023与2022全年可比列。"),
        ("fuyao_2023","按福耀玻璃2023年报，2023年归母净利润同比是多少？","parent_net_profit_yoy","归母净利润不能用合并净利润替代。"),
        ("fuyao_2023","按福耀玻璃2023年报，2023年合并净利率是多少？使用净利润除以营业收入。","net_margin","使用合并净利润，不使用归母净利润。"),
        ("fuyao_2023","按福耀玻璃2023年报，2023年末合并资产负债率是多少？","debt_asset_ratio","资产负债均使用2023年12月31日時点。"),
        ("fuyao_2024","按福耀玻璃2024年报，2024年合并毛利率是多少？","gross_margin","使用合并营业收入减营业成本后除以营业收入。"),
        ("fuyao_2024","按福耀玻璃2024年报，2024年末合并流动比率是多少？","current_ratio","流动资产除以流动负债，以倍数列示。"),
        ("fuyao_2024","按福耀玻璃2024年报，2024年合并经营现金流收入比是多少？","operating_cash_flow_to_revenue","采用全年经营现金流量净额除以同年营业收入。"),
        ("fuyao_2024","按福耀玻璃2024年报，2024年整体毛利率同比变动多少个百分点？","gross_margin_yoy_pp","比较同份报告的2024与2023全年；变化单位为百分点。"),
        ("fuyao_2024","按福耀玻璃2024年报，2024年归母净利润同比是多少？","parent_net_profit_yoy","归母净利润独立于合并净利润。")]:
        add(sid,"data",question,[point],formulas=[formula])
    add("fuyao_2023","hybrid","计算福耀玻璃2023年合并营业收入同比，并说明收入增长的年报解释。",
        ["使用同表全年比较列。", "公司披露加大营销力度和高附加值产品占比提升。"],
        [(23,"营业收入增加主要是因为公司加大营销力度及高附加值产品占比提升所致。")],["revenue_yoy"])
    add("fuyao_2023","hybrid","计算福耀玻璃2023年经营现金流收入比，并说明经营现金流增加的已披露原因。",
        ["现金流与营业收入均采用合并全年口径。", "收入增长，同时加快存货周转、减少储备。"],
        [(23,"经营活动产生的现金流量净额增加，主要是因为收入增长，同时加快存货周转减少储备所致。")],["operating_cash_flow_to_revenue"])
    add("fuyao_2023","hybrid","计算福耀玻璃2023年合并毛利率，并说明营业成本增加的年报解释。",
        ["整体毛利率依据合并营业收入与营业成本计算。", "公司披露成本增加主要因为营业收入增加。"],
        [(23,"营业成本增加主要是因为营业收入增加所致。")],["gross_margin"])
    add("fuyao_2024","hybrid","计算福耀玻璃2024年整体毛利率同比百分点变化，并说明包装物相关成本处理为何需关注。",
        ["使用同份2024年报内的全年比较列。", "本期相关包装物作为履约成本计入营业成本，比较费用与毛利时应注意列报变化；不可自行改写上年数字。"],
        [(23,"销售费用减少主要是因应本报告期包装物精细化、专用化需求趋势"),
         (23,"本年度将其作为履约成本计入营业成本。")],["gross_margin_yoy_pp"])
    add("fuyao_2024","hybrid","计算福耀玻璃2024年合并净利率，并说明汇兑变化对利润总额同比的披露影响。",
        ["净利率采用合并净利润。", "年报披露由上年汇兑收益转为本期损失，使本期利润总额较上年同期减少39,298.12万元；不能将这项差额当成净利润差额。"],
        [(21,"本报告期汇兑损失人民币2,428.49万元，上年同期汇兑收益人民币36,869.63万元"),
         (21,"使本报告期利润总额比上年同期减少人民币39,298.12万元")],["net_margin"])
    add("fuyao_2024","hybrid","计算福耀玻璃2024年经营现金流收入比，并说明经营现金流增加的年报解释。",
        ["两个操作数均使用2024全年合并口径。", "已披露原因是收入增长，不能扩大为公司所有回款指标均改善。"],
        [(23,"经营活动产生的现金流量净额增加，主要是因为收入增长所致。")],["operating_cash_flow_to_revenue"])
    add("fuyao_2023","boundary","只根据所选福耀玻璃2023年报，计算2025年全年营业收入同比。",
        ["所选报告不含2025全年实际营业收入，证据不足；不可预测或补造实际数。"],status="insufficient_evidence")
    add("fuyao_2024","boundary","计算福耀玻璃2024年的投入资本回报率ROIC。",
        ["ROIC不在当前八项受控公式白名单；应明确不支持，不能临时生成口径或用净利率替代。"],status="unsupported")
    add("fuyao_2024","boundary","我选择的是福耀玻璃年报，但请你据此计算贵州茅台2024年营业收入同比。",
        ["问题公司与所选报告冲突，应请用户确认公司或重新选择文档；不能跨公司取数或扩大报告范围。"],status="needs_clarification")
    assert len(cases)==30 and Counter(c['category'] for c in cases)=={'semantic':12,'data':9,'hybrid':6,'boundary':3}
    for sid,ref in references.items():
        (ROOT / "evals" / f"{sid}_reference.json").write_text(json.dumps(ref,ensure_ascii=False,indent=2),encoding="utf-8")
    encoded=("\n".join(json.dumps(c,ensure_ascii=False) for c in cases)+"\n").encode("utf-8")
    (ROOT / "evals/annual_test_v1.jsonl").write_bytes(encoded)
    assert sha(developer_file.read_bytes())==developer_sha
    assert len(developer_bytes.splitlines())==20
    (ROOT / "evals/annual_50_v1.jsonl").write_bytes(developer_bytes.rstrip(b'\r\n')+b'\n'+encoded)
    stats=[]
    for sid,sample in samples.items():
        report=Report(id=sid,sha256=sample['sha256'],path=sample['path'],filename=Path(sample['path']).name,
                      company=sample['company'],stock_code=sample['stock_code'],doc_type='annual_report',fiscal_year=sample['year'])
        page_numbers=list(range(min(PAGES[sid]),max(PAGES[sid])+1))
        parsed=parse_report(report,page_numbers=page_numbers,split_chunks=False)
        extracted=extract_pdf_facts(parsed)
        verified=verify_against_reference(extracted,references[sid])
        stats.append({'report_id':sid,'reference_facts':len(references[sid]['facts']),
                      'actual_extracted_facts':len(extracted),'reference_matched_facts':sum(f.validation_status=='verified' for f in verified),
                      'reference_not_injected_into_parser':True})
    audit={'created_at':datetime.now(UTC).isoformat(),'freeze_sha256':freeze_sha,'parser_version':PARSER_VERSION,
           'development_sha256_unchanged':developer_sha,'test_sha256':sha(encoded),'test_cases':30,
           'test_distribution':dict(Counter(c['category'] for c in cases)),
           'quote_page_checks':sum(len(c['gold_evidence']) for c in cases),'quote_page_checks_passed':True,
           'source_validation':stats,'rule_differences_before_labeling':before_rules,
           'rule_differences_after_labeling':rule_differences(freeze),'human_reviewed':False,
           'production_rules_modified_by_this_script':False,'model_evaluation_status':'not_run',
           'boundary':'reference匹配是基于assistant标签的核验；使用该事实进行计算须标为已核验输入轨道，不能称纯自动抽取或模型得分。'}
    (ROOT / "evals/annual_holdout_validation_v1.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest['labeling']['test_prepared']=30
    manifest['labeling']['combined_prepared']=50
    manifest['labeling']['rules_freeze_sha256']=freeze_sha
    for sample in manifest['samples']:
        if sample['id'] in samples:
            sample['label_status']='assistant_heldout_reference_prepared_after_freeze'
            sample['reference_path']=f"evals/{sample['id']}_reference.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Report aggregate status only; do not stream held-out question contents to rule owners.
    print(json.dumps({k:audit[k] for k in ['test_cases','test_distribution','quote_page_checks','quote_page_checks_passed','source_validation']},ensure_ascii=False))


if __name__ == '__main__':
    prepare()

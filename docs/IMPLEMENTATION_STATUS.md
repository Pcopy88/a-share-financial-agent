# 开发执行记录

依据：A股年报智能分析Agent开发文档 v1.7，第 9 节的 16 步及 F01—F12 全范围。
目标：完成应用代码、README 启动说明及逐项验收文档；不把模拟成绩冒作真实云端效果。本文记录实现进度，最终统一测试、题集冻结及对照数据由 root 填入最后快照；“已实现”不等于真实云端质量已验收。

## 执行约定

- 用户完成开发后自行填写两家 API Key；现在不要求密钥，不调用收费接口。
- 先测试有意义的边界，再实现并运行测试；本地 PDF、SQLite、Chroma 用真实组件验证。
- 2026H1 江丰电子样本、南方乳业招股书和越疆问询仅作开发资料，不能假称留出测试。
- 工程在原共享目录开发，Git 已初始化；最终提交与交付快照由 root 统一处理，不另外搬动用户样本。
- 共用类型先确定；并行任务按独立模块分工，完成后整体验证。原计划功能边界不因分工缩小。
- 年报 50 题及模型对照继续保留目标；未取得的资料、未执行的 live 检查如实记待验，不填造成绩。

## 开发清单

| 步骤 | 内容 | 当前状态 | 证据 |
|---|---|---|---|
| 1 | 资料与开发小题集 | 已完成 | 4 份用户资料、问询开发题及江丰5题；另有6份全年报，正式20/30题已执行；冻结及工程修订均登记 |
| 2 | 工程环境、配置、模拟及真实探针 | 本地验证完成，live待配置 | Python 3.11.16、uv.lock、smoke_models.py、实际子进程安装导入 |
| 3 | 类型、客户端、存储 | 已实现并有模块测试 | 两家HTTP契约、SQLite事务和故障恢复；无真实云调用 |
| 4 | PDF 解析与父子切片 | 已实现，parser v4最终全文统计待汇总 | 4份全文共732页的早期快照；固定500/80；同页不同口径、无框线及不确定表格回归 |
| 5 | 向量检索、过滤、上下文 | 本地集成验证完成 | 真实Chroma/SQLite，mock向量；版本固定、失败恢复、过滤、原文页映射 |
| 6 | 意图、改写、子问题规划 | 已实现并有模块测试 | 范围/否定条件保留、澄清与1—3子问题规划；云端理解效果未验 |
| 7 | 生成、验证、聚合 | 模块验证完成 | 实际发送集合/版本/页码/公式核验；语义支持仍需人工评测 |
| 8 | 九节点工作流、API、界面 | 已实现并完成本地联调 | 真实LangGraph/检查点、单队列、重启中断恢复、API与四页UI；实际界面收入同比及来源页跳转已验 |
| 9 | PDF 财务提取与核验 | 已实现并有真实原表验证 | 江丰18项合并事实参考核对；五份明确开发参考按SHA使用；10份PDF共1903页、12700块，最新parser v4 |
| 10 | 指标计算器 | 已实现并有模块测试 | 8项固定Decimal公式，单位/期间/口径/分母/冲突校验；不执行模型生成代码 |
| 11 | 数字及混合联调 | 已实现，最终统一回归待确认 | 事实绑定真实Evidence、逐子项状态和运行快照；江丰5题由validate_local.py重跑确认 |
| 12 | 启动、备份、最小交付 | 已实现并完成本地演练 | CLI/start.ps1、RuntimeLock、SHA校验备份恢复；临时资料库恢复后PDF与Chroma查询可用 |
| 13 | BM25、RRF、本地重排 | 已实现，CPU真权重探针通过 | 双路/RRF/过滤测试；本地bge-reranker-v2-m3 CPU FP32真实推理，CUDA未验 |
| 14 | 公告、AKShare、报告 | 已实现并有接口回归 | 公告候选/受限下载；API候选与活跃PDF版本核对；Markdown导出取运行时实际事实，不丢操作数 |
| 15 | 题集、评测程序 | 程序已实现，最终题集与实验快照待汇总 | R0—R3、M1/M2、Q0/Q1、C0/C1；逐题/冻结输入/指标测试；真实云端及跨供应商对照未验 |
| 16 | 对照、故障检查、最终验收与 README | 代码与本地验收完成，live待Key | 400测试通过、Ruff通过；50题流程和开发2题各组smoke完成 |

## 接口衔接检查

- 解析输出 Page/ParentBlock/Chunk 供 SQLite 和 Chroma 使用；统一 ID 与页码，保留表格范围。
- 两家客户端只共享通用日志和有限重试边界，不共用请求格式。
- 计算器只接收核验财务事实；LLM 不能执行任意代码或将引用文本当作系统指令。
- mock/live 隔离配置、索引和缓存；缺 Key 的 live 模式显式失败。
- UI 只调用后端；后端单队列和单 worker，持久化任务状态；中断后可恢复或重试。
- 准备期间拒绝身份修改；已提交运行固定索引、元数据和事实快照。财务导入只核对当前活跃版本，重复API快照不会覆盖既有观察记录。
- 解析与CLI共用ARA_DATA_DIR/tokenizer，旧parser版本缓存不复用；分析运行与准备任务均保存节点耗时、成功/失败/跳过状态。

## 验证记录

2026-10-03：分模块与本地集成测试已执行。并行工作期间产生的测试计数只属于当时快照，最终数量不提前填写。

- 四份全文早期快照：江丰193页、南方360页、越疆131页、第二轮48页，共732页；详见tmp/preparation/ingestion_regression_summary.json。当前parser已更新为pdfplumber-structure-v4，最终父块/子块统计由最新重跑确定，不沿用旧计数。
- 南方第116、118、142、144物理页：TABLE_STRUCTURE_UNCERTAIN，原文保留，不默认开放这些表的可信计算。
- 江丰原表参考核对18项合并事实，母公司事实未因提取成功而自动verified。近期跨行归母利润引用与口语指标识别修复后，root已确认最新五题本地回归5/5；证据为outputs/mock/acceptance/jiangfeng_workflow.json。这是确定性计算及流程核验，不是云端问答准确率。
- Windows中文路径的普通editable .pth会因cp936失败，已使用Hatch dev-mode-exact与editables修复，实际子进程导入通过。PYTHONUTF8不当作该问题的单独解决方案。
- pytest使用项目tmp下唯一basetemp，避开本机临时目录权限问题。
- scripts/smoke_models.py默认显式mock，使用真实HTTP响应校验器、Chroma和向量维度验证；只有--live才使用两家真实账户。
- 本地重排锁定BAAI/bge-reranker-v2-m3 revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e；CPU FP32真实权重探针见outputs/local_checks/reranker_smoke.json。两短文本对的单次延迟不是整条问答P95，GPU/CUDA未验。
- README中的doctor、默认mock探针、CLI help、测试脚本及临时目录backup/restore均已实际执行过；本轮文档审计只读CLI/UI/API源码，未启动共用Runtime。
- scripts/validate_experiments.py仅允许mock，显式使用六份年报及其参考核验，执行50题M2流程和两道开发题的单次各实验检查；输出outputs/mock/acceptance/experiment_workflow.json。当前执行结果由root完成后填写，参考辅助轨道不能声称独立留出质量或完整模型对照。

## 最终交付快照（由 root 填写）

| 项目 | 结果 |
|---|---|
| 最终代码版本、测试命令与完整通过/跳过/失败数量 | 见 docs/验收文档.md 及 outputs/mock/acceptance/delivery_snapshot.json |
| lint、安装/启动、API/UI及江丰五题统一验收 | 见 docs/验收文档.md 及 outputs/mock/acceptance/delivery_snapshot.json |
| 最新parser全文统计、6份年度资料与20/30题冻结信息 | 见 docs/验收文档.md 及 outputs/mock/acceptance/delivery_snapshot.json |
| 各实验逐题结果、延迟/节点/usage统计及未完成项 | 见 docs/验收文档.md 及 outputs/mock/acceptance/delivery_snapshot.json |
| 用户填写Key后的live联通、重新编码与真实问答验收 | 待用户配置后执行 |

当前仅配置DeepSeek Flash；不宣称已完成跨供应商比较、50题真实live成绩或用户人工审核。原图15%/23%提升尚未复现。逐项验收边界见[验收文档](验收文档.md)。

最终统一验证：400 passed，Ruff 全通过。实际云端质量、GPU及当前网络阻塞的公告PDF下载仍待验，不以本地模拟执行代替。Windows HNSW 中文路径与财务原值点击页码已修复并经独立进程/真实界面回归。

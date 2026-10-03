# A股年报智能分析 Agent 开发文档

版本：v1.7 本地实现与分阶段验收同步<br>
编写日期：2026年10月3日<br>
适用对象：独立开发者，以及协助实现本项目的开发人员<br>
当前状态：核心代码与本地样本流程已实现，正在扩展联调；真实云端验收待用户配置 Key。具体完成项与证据以 `IMPLEMENTATION_STATUS.md` 和最终验收文档为准，不把下面的设计目标当作已经取得的模型效果。

本文根据提供的项目截图，明确复刻范围、技术选型、系统流程、数据约束、开发顺序和验收标准。先完成一个能够“读取年报、回答问题、准确计算、展示证据”的应用：本地解析与存储资料，通过百炼 text-embedding-v3 编码文档和问题，在本地向量库检索，由 DeepSeek-V4.1-Flash 理解问题和生成答案；财务计算由本地程序执行。再补齐混合检索优化和模型对比实验。

**最重要的实现原则：模型负责理解与表达，程序负责取数、计算、校验和来源管理。** 功能是否完成，要用真实年报、可复现的测试与评测结果判断。

## 1 图片内容与复刻边界

### 1.1 图片中的文字

**A 股年报智能分析 Agent　独立开发　2025 年 3 月—2025 年 6 月**

> 项目背景：针对上市公司年报篇幅长、财务术语密集，以及传统 RAG 在精确术语检索、财务数据计算和结果可追溯性上的不足，设计并实现面向 A 股年报的智能分析 Agent。
>
> Agent 工作流编排：基于 LangGraph 搭建 9 节点工作流，串联年报获取、文档解析、知识库构建、任务路由、信息检索、指标计算与报告生成，实现年报分析流程自动化。
>
> 混合检索与智能路由：构建 BM25 + bge-m3 双路召回、RRF 融合与 Cross-Encoder 重排序链路，并根据语义、数据、混合三类问题动态选择检索或 Pandas 计算路径。Precision@5 提升约 15%，金融术语召回率提升 23%。
>
> 可信分析与模型评测：通过 Pydantic 统一处理 AKShare 字段、单位及缺失值，并将回答关联原文页码与置信度。基于 Ollama、MLX 部署本地模型，构建 50 题评测集对比 DeepSeek、Gemma 与 Qwen 系列模型。

以上是截图内容，不是本项目已经完成的履历或实验结果。截图未提供源码、样本、原始评测记录，也没有列出 9 个节点的名称。因此：

- 复刻目标是实现上述能力，不能声称恢复原作者的内部实现。
- 下文的节点划分、界面、数据库、参数及验收门槛是本项目的设计建议。
- “提升约 15%”“提升 23%”需要重新测量；不预先承诺能够达到。
- 图片中的开发日期只属于原描述，不用于标记本次开发时间。

### 1.2 本版默认范围

默认应用在 Windows 本地运行，单用户使用；所有生成类 LLM 环节采用云端 API，完整问答需要联网。文档输入覆盖年报、半年报、招股说明书、问询函和问询回复等相关 PDF；通用文本解析与检索不以沪深市场为限制，南方乳业招股书和江丰电子 2026 年半年报均作为开发样本。AKShare 自动财报适配仍先覆盖沪深市场非金融企业、人民币合并报表。原年报评测保留 3 家公司、每家连续 2 年、共 6 份可复制文字的 PDF，其中 2 家用于开发、1 家留作测试；半年报、招股书和问询资料另补对应验收样本，不冒作全年年报测试。

一次分析必须明确公司和文档范围。首版支持单公司选定资料内的问答，以及同公司相邻年度或半年报同期对比；不同文档、期间、问询轮次和版本分别保留。北交所自动数据源适配、银行保险证券公司的专门指标、任意多公司横向比较、扫描件 OCR、全市场批量抓取列入后续扩展；这不限制用户上传北交所相关 PDF 做原文检索。

用户已确认文档结构优先的递归切分、正文 500 tokens / overlap 80 tokens、下表三项检索设计，以及全部生成类环节使用云端 DeepSeek-V4.1-Flash、Embedding 使用百炼 text-embedding-v3。目标显卡为 RTX 5070 Ti 16GB，本地重排在第 13 步接入；样本规模与界面仍为实施默认方案。切分计数与云端计费 token 分开定义，见第 3.6 节。

| 已采纳的设计 | 具体要做什么 | 首版边界 |
|---|---|---|
| 小块检索与原文上下文补充 | 用小块召回和重排，再补充它所属的原文上下文 | 摘要向量仅作为后续对照实验；摘要不能作为事实依据 |
| 元数据辅助召回 | 根据公司、文档类型、问询轮次、版本等确定查找范围 | 明确条件严格过滤；猜测的章节不作为排除条件 |
| 意图识别与问题改写 | 将口语问题转成清楚的检索问题，同时保存用户原问 | 不擅自增加年份、改变否定条件；关键歧义先澄清 |

本轮另确认：普通问答的子问题处理并入现有 N4，与意图识别、问题改写共用一次正常 LLM 调用；支持 1 个问题或 2—3 个明确子问题，逐项执行并标记结果，不新增 Agent 或业务节点。人工纠错页面在最小闭环跑通后，根据真实错误再迭代；首版仍保留原文、提取值、校验状态和来源，未经核验的数据不用于计算。

开发分为三段：先完成文字问答小闭环，再接入可信数字计算形成最小分析闭环，最后逐项增加混合检索优化、自动获取、报告和完整评测。第 9 章是实际开发顺序；第 4 章描述运行时流程，两者不是同一种顺序。新方案的实现对应表和编码准备情况见第 9.1、9.2 节。

## 2 具体实现哪些功能

| 编号 | 功能 | 用户看到的结果 | 完成层级 |
|---|---|---|---|
| F01 | 上传年报 招股书与问询 PDF | 显示公司、文档类型、报告期、问询轮次、版本和处理状态 | 最小闭环 |
| F02 | 按公司和年度获取年报 | 列出公告候选，选中全文后下载；失败时保留上传入口 | 完整复刻 |
| F03 | 解析正文与表格 | 保留章节、物理页码、表格标题和单位；显示失败页 | 最小闭环 |
| F04 | 建立持久化知识库 | 同一报告重启后可继续查询；重复上传不重复入库 | 最小闭环 |
| F05 | 资料文字问答 | 识别口语问题，在选定年报、招股书或问询资料内查找原文 | 最小闭环 |
| F06 | 财务取数与指标计算 | 展示原始数值、期间、单位、口径、公式与计算结果 | 最小闭环 |
| F07 | 混合分析与子问题处理 | 分别计算和检索，逐项回答 2—3 个明确问题，列出资料缺口 | 最小分析闭环 |
| F08 | 同公司年度对比 | 并列展示两年的指标、来源、重述说明和变化 | 完整复刻 |
| F09 | BM25 与向量混合检索 | 按元数据限定范围，用小块召回和重排，再补充父级原文上下文 | 完整复刻 |
| F10 | 引用与证据状态 | 点击引用看到对应 PDF 页、原文片段和数据来源 | 最小闭环 |
| F11 | 生成简版分析报告 | 按固定模板生成并导出 Markdown 报告 | 完整复刻 |
| F12 | 模型切换与评测 | 用相同题集比较模型质量、速度、API 用量和本地资源占用 | 完整复刻 |

最小闭环可以先用向量检索；F09 完成后才称为混合检索复刻版。本项目当前验收包含 F01—F12，以及第 10 节的评测记录；截图中的 Ollama/MLX 本地生成部署改为后续可选实验，不属于当前必做范围，不能声称已经复现。

### 2.1 三类问题示例

| 类型 | 示例 | 系统主要动作 |
|---|---|---|
| 语义类 | “公司在年报中披露了哪些原材料风险？” | 检索原文，重排，组织答案与引用 |
| 数据类 | “2024 年营业收入同比增长多少？” | 获取两期同口径收入，校验，程序计算 |
| 混合类 | “2024 年毛利率变化多少？年报解释了哪些相关原因？” | 计算两期毛利率，同时检索管理层讨论 |

“为什么增长”不能只靠两个数值推导原因。原文没有明确解释时，只能报告变化，并说明未找到明确原因。

区分原文查证与财务计算：问“回复披露的产品单价是多少”属于带数字的原文查证，走 semantic，并核对表头、期间、单位及预测/实际属性；不因答案含数字就调用计算器。需要规范化财务事实或公式运算的年报指标才走 data。超出已支持指标的计算要求明确返回 unsupported，不能用文字检索绕过计算白名单。

### 2.2 页面设计

完整版本采用以下 4 个页面。文字问答小闭环先完成前三个页面的最简功能；第四页随报告和评测后端接通后再增加，不提前实现无后端的功能：

1. **资料库**：上传、查询公告、确认公司、文档类型、问询轮次和版本，查看入库状态和失败原因。
2. **分析问答**：选择公司及资料范围，输入问题，查看原问、改写后的检索问题、实际检索范围、逐个子问题的答案或未完成原因、公式和引用。
3. **证据查看**：展示引用原文、PDF 对应页、表格行列位置及来源信息。
4. **评测与报告**：选择模型、查看实验结果、导出分析报告和逐题评测结果。

生成报告使用固定章节：分析范围、经营概况、主要指标、管理层解释、风险事项、资料缺口、引用清单。缺少证据的章节保留“资料不足”，不自动补全内容。

报告任务由 N4 把固定模板转换为查询计划，将经营概况、原因说明、风险事项和所需指标分配给子问题。各项依次经过相应的 N5—N9 路线，最后按 section_id 将已校验子答案填入报告模板；复用条件边循环，不增加业务节点数。

普通问答复用同一套子问题契约与现有节点，具体执行规则见第 6.1 节。证据查看在最小闭环阶段提供只读核对；“修改数值并重新计算”的交互入口属于第 5.6 节的后续迭代。

## 3 使用什么技术

### 3.1 路线选择

| 方案 | 优点 | 代价 | 本项目选择 |
|---|---|---|---|
| Python + Streamlit + FastAPI | 方便展示财务表格，前后端职责清楚，适合单人实现 | 界面定制能力有限 | 推荐作为首版 |
| React + FastAPI | PDF 阅读和交互更灵活，适合后续产品化 | 前端开发量更大 | 第二阶段再考虑 |
| 在现成文件问答平台上配置 | 容易得到基础问答演示 | 9 节点、取数校验和评测逻辑仍需定制 | 不作为主要复刻路线 |

### 3.2 推荐技术栈与职责

| 层次 | 选择 | 具体职责 |
|---|---|---|
| 语言与环境 | Python 3.11、uv | 隔离依赖，生成锁文件；兼容性在环境验证阶段确认 |
| Web 界面 | Streamlit | 资料上传、问答、指标表、证据页和实验结果 |
| 服务接口 | FastAPI + Uvicorn | 接收上传、创建任务、查询状态、提供报告和证据 |
| 工作流 | LangGraph | 编排 9 个业务节点，条件分支、状态记录和重试 |
| PDF 解析 | pdfplumber | 按页提取正文、表格、文字位置 |
| 证据页渲染 | pypdfium2 | 把引用页渲染成图片，作为可定位的证据预览 |
| 公告及财报数据 | AKShare + httpx | 查询公告元数据、获取结构化报表、下载已确认的 PDF |
| 数据校验 | Pydantic v2 | 定义字段类型和自定义校验；规范输入输出 |
| 数据整理与计算 | Pandas + Decimal | 筛选、对齐年份和口径；按固定公式计算 |
| 关键词检索 | jieba + rank-bm25 | 中文分词、财务术语词典、BM25 排序 |
| 文本向量编码 | httpx 调用百炼 DashScope `text-embedding-v3`，1024 维 dense | 用 document 编码文档、query 编码问题，再在本地索引查找语义相关段落 |
| 切分计数 | tokenizers，仅使用固定版本的 bge-m3 tokenizer 文件 | 在 CPU 上保持原定 500/80 切分口径，不加载 bge-m3 权重或生成向量 |
| 向量存储 | Chroma 本地持久化模式 | 存储向量和可过滤的报告元数据 |
| 融合 | 自行实现 RRF | 根据两路排序名次合并候选 |
| 重排序 | FlagReranker，BAAI/bge-reranker-v2-m3 | 对“问题—候选片段”评分，重新排序 |
| 云端生成模型 | DeepSeek-V4.1-Flash，API 型号 `deepseek-flash`，经 httpx 调用 | N4 理解、改写与拆题；N8 生成答案与报告，共用一个模型配置 |
| 后续本地对照 | Ollama；MLX / MLX-LM 仅在支持的环境中考虑 | 最小闭环完成后按需要验证，不作为首版依赖或自动回退 |
| 业务存储 | SQLite | 报告、财务事实、任务、答案、引用、实验元数据 |
| 验证与质量 | pytest、Ruff、逐题 JSONL 日志 | 单元测试、集成测试、质量评测和错误定位 |

这些是设计选型，不代表已经完成安装、兼容性验证或性能测试。开发环境跑通后再固定依赖版本；不得在文档中凭空填写“已验证版本”。

### 3.3 关键技术边界

- **Pydantic 不会自动理解财务口径。** 开发者仍需编写字段映射、单位转换和跨字段规则；Pydantic 负责执行这些规则。[Pydantic 校验器文档](https://docs.pydantic.dev/latest/concepts/validators/)
- **pdfplumber 优先用于带文字层的 PDF。** 扫描件不能仅凭提取结果为空就当成“年报没有相关内容”；首版显示需要 OCR。[pdfplumber 官方说明](https://github.com/jsvine/pdfplumber)
- **LangGraph 的节点可以是普通函数。** 取数、切分、计算等环节没有必要调用大模型。[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- **Embedding 与生成模型分工不同。** text-embedding-v3 负责将问题和原文编码为向量，DeepSeek 负责理解问题和组织回答。首版只取 dense 输出，后续独立实现 BM25；模型的 sparse 输出不能直接称为 BM25。[百炼向量接口](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- **本地向量库由后端统一访问。** Streamlit 不直接写 SQLite 或 Chroma；同一报告的索引构建串行执行，避免读到半成品。[Chroma 官方仓库](https://github.com/chroma-core/chroma)
- **首版不依赖 Docker。** 先在原生 Windows 跑通；需要部署复现时再增加容器配置。

### 3.4 模型和硬件安排

所有原千问生成类环节统一改为 DeepSeek-V4.1-Flash。官方 API 的 `model` 参数使用 `deepseek-flash`，不是展示名称。通过 httpx 向 `https://api.deepseek.com/chat/completions` 发送请求；首版显式设置 `thinking={"type":"disabled"}` 和 `response_format={"type":"json_object"}`，N4、N8 分别设置输出预算。以下为官方服务配置；若实际 Key 来自第三方平台，应先核对其服务地址与支持型号，不直接认定兼容。[官方型号说明](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)

提示词包含 `json` 字样与目标结构样例。空 content、JSON 无法解析或输出因长度截断均不算成功；按已有有限修复规则处理，再由 Pydantic 和业务规则校验。JSON Object 只约束格式，不保证字段含义、引用或结论正确。[JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)

| 环节 | 模型或工具 | 运行位置及阶段 |
|---|---|---|
| 文档与问题的 Embedding | `text-embedding-v3`，1024 维 dense | 百炼云端，文字问答基线启用 |
| 候选片段重排 | `BAAI/bge-reranker-v2-m3` | 本地，第 13 步启用 |
| 意图识别、口语改写、1—3 个子问题规划 | DeepSeek-V4.1-Flash，`deepseek-flash` | 云端，N4 共用一次正常调用 |
| 答案、原因解释及报告内容生成 | 同一个 `deepseek-flash` | 云端，N8 逐子问题调用 |
| 数值计算、引用与结构校验 | Decimal、Pydantic 和业务规则 | 本地，不调用生成模型代算 |

N9 首版以确定性校验为主；后续若增加模型辅助检查、摘要生成等生成类调用，也通过同一 DeepSeek 云端配置执行，单独记录调用量。Ollama、本地生成模型或其他云端模型对照及 MLX 部署均后置；不增加首版必须部署的生成模型，也不自动回退到千问。

环境读取确认 RTX 5070 Ti 显存 16303 MiB、系统内存约 31.1 GiB，但尚未运行本项目模型。文字问答小闭环不依赖本地 GPU 推理、PyTorch 或 Ollama；第 13 步启用本地重排时，再安装对应依赖并以小批量 FP16 实测资源。切分用 tokenizer 是 CPU 上的文本处理工具，不是本地 Embedding 推理。

云端生成配置记录 provider、可取得的地域信息、base_url、请求 model_id、响应 model、可取得的 system_fingerprint、思考开关、输出格式、上下文与输出预算、超时和调用日期。`deepseek-flash` 当前对应 V4.1-Flash；未找到可固定日期的公开快照 ID，不编造版本号，也不把兼容旧型号当作固定快照。正式对照注明这一复现限制。Embedding 另记录百炼地域、API Host、model_id、维度、编码用途、归一化和服务版本标识（不可取得时记未知）；本地重排记录权重 revision、精度和设备。

生成模型读取本机 `DEEPSEEK_API_KEY`，Embedding 读取 `DASHSCOPE_API_KEY`，两家服务的 Key 不混用。`.env.example` 只写变量名与非敏感配置，真实密钥不入仓库、前端或日志。入库时全部待编码的子块、检索时的问题文本发送至百炼；N4 的原问及必要资料范围、N8 的选中证据与计算结果发送至 DeepSeek。原 PDF、解析结果和向量库保存在本地，不能因此声称原文文本从不离开本机。

两家调用分别记录阶段、request_id、耗时、usage 和费用估算。DeepSeek 输入区分缓存命中与未命中、输出单独计费，不重复累加输入总量及其缓存子项；v3 记录返回的 usage.total_tokens，不套用生成模型字段。重试、重建与子问题计入实际用量，按对应服务价格计费。用户已说明持有两家 Key，并明确在代码开发完成后自行写入环境文件。开发期间使用显式模拟客户端验证接口契约和流程；真实账户权限、联通、建库与问答效果在用户配置后集中验收，不以提前提供 Key 作为编码前置。

断网时仍可解析本地 PDF、浏览已有原文及结果、运行已有核验事实的确定性计算；新文档编码、新问题的向量编码、意图识别和生成需要网络。编码失败返回明确失败状态，不伪装成“没有相关证据”，也不自动切换本地模型或其他 Embedding。

### 3.5 text-embedding-v3 接入约定

已确定使用百炼 `text-embedding-v3` 的 1024 维 dense 输出，默认北京地域；实际地域与 API Host 必须和用户账户匹配，待用户填写配置后运行第 2 步准备的探针验证。通过 httpx 调用 DashScope 原生 HTTP 接口，以支持 document/query 区分；不照搬生成接口或 LlamaIndex 示例替换现有工作流。

北京业务空间专属地址模板为 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`。完整地址放入 `DASHSCOPE_EMBEDDING_URL`；从用户控制台实际 API Host 配置，不能发送仍带占位符的 URL。其他地域使用其对应地址，不在代码里固定北京地址。[官方接口说明](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)

| 项目 | 固定规则 |
|---|---|
| 请求体 | model=`text-embedding-v3`，文本位于 input.texts；parameters 内设 dimension=1024、output_type=dense |
| 编码用途 | 文档子块设 parameters.text_type=document；检索问题设 query；首版不额外加 instruct |
| 批量与长度 | 每请求最多 10 条，每条最多 8192 个服务端 token；初始串行发批次 |
| 响应映射 | 按 output.embeddings 中的 text_index 映射该批输入，再映射到 chunk_id；不假设返回数组天然有序 |
| 向量检查 | 数量与索引一一对应、索引不重复或越界、维度为 1024、所有值有限且向量非零；通过后统一 L2 归一化 |
| 本地检索 | Chroma 固定 cosine 距离；显式写入文档 embeddings、查询传 query_embeddings，禁用隐式默认编码器 |
| 用量记录 | 保存 request_id 与 usage.total_tokens；缺失用量记未知，不自行用切分计数冒充计费数据 |

服务端字段与限制依据同一官方接口；归一化、串行批次和向量验证是本项目的工程约定。原生 HTTP 客户端同时检查 HTTP 状态与服务错误 code/message，不依赖响应 JSON 中必有 SDK 风格的 status_code。失败处理见第 4.3、6.4 节。v3 为当前已选用的通用文本向量模型，实际招股书与年报效果仍用开发题验证；表格解析、元数据、BM25 与计算校验分别实现，不能靠换 Embedding 替代。

### 3.6 切分计数与模型变更规则

正文仍采用 500 tokens / overlap 80 tokens。为保持已经确定的计数口径，使用固定版本 `BAAI/bge-m3` 的 tokenizer 文件，仅通过 tokenizers 在 CPU 上计数，不加载模型权重、不调用它生成向量；第 2 步保存 tokenizer 文件哈希与版本，计数关闭额外 special tokens、自动 padding 和 truncation。这是一把固定的切分尺，不是 v3 的官方 tokenizer。[BGE 模型仓库](https://huggingface.co/BAAI/bge-m3)

分别保存 chunk_token_count（固定切分尺）、云端 Embedding usage（服务端计费）和 LLM 上下文预算；三者不能互相代替。未核实可精确匹配 v3 的本地 tokenizer，不用字符数换算宣称精确服务端 token 数。正文与表格使用较小片段；接口拒绝过长输入时，标记具体块，由切分层按结构再拆分并更新来源映射后重试入库，不在编码器里静默截断或用无向量块发布索引。过长查询明确要求缩短，不删改问题条件。

文档与问题使用相同服务商、模型、维度及归一化配置，并分别保持 document/query 用途。后续更换模型、维度、tokenizer 或影响文本的切分配置时，建立独立索引版本并重新编码受影响文档；同为 1024 维也不能混用 BGE 与 v3 向量。本地 bge-m3 推理仅作为后续可选对照，不能自动回退。价格和质量对照记录实际调用用量、同一批切分与问题，不将截图中的其他模型分数当成本项目成绩。

## 4 系统整体流程

### 4.1 资料准备与在线分析

资料准备流程：获取 PDF → 确认文档身份和元数据 → 本地按页解析 → 保存父级原文并切分子块 → 云端 v3 按 document 编码 → 写入本地当前阶段启用的索引 → 发布可查询版本。解析可离线完成，新的向量编码需要联网。适用的财务事实另走提取、规范化和核验分支；财务分支失败不阻断已经验证的文字检索，状态分别保存。

问题处理流程：保留用户原问 → 在 N4 同时识别意图、改写并列出子问题 → 确认各子问题的资料范围 → 逐项召回、补原文或取数计算 → 逐项生成并校验 → 汇总已回答项和未完成原因 → 展示结果。

这两条流程使用同一组业务模块。已经入库的报告直接进入问题处理流程，不重复下载、解析和生成向量。

### 4.2 9 个业务节点

以下是为复刻设计的节点划分。START、END、条件边和普通辅助函数不计入 9 个业务节点。

| 节点 | 名称 | 输入与处理 | 主要输出 |
|---|---|---|---|
| N1 | acquire_report 获取或定位年报 | 处理上传或确认过的公告，检查文件哈希和已有版本 | report_ids、入库状态 |
| N2 | parse_report 解析年报 | 提取正文、表格、页码及适用的财务候选；API 适配接通后才关联快照 | 页面、表格、候选财务事实 |
| N3 | build_knowledge 构建知识库 | 保存父子块和元数据，调用 v3 文档编码，校验后发布当前索引；独立保存财务核验状态 | index_version、父子映射及各分支状态 |
| N4 | route_question 理解与路由 | 保存原问；一次正常云端 LLM 调用产生意图、改写和子问题计划，再由程序校验范围和歧义 | 含 subtasks 的 QueryPlan 或澄清请求 |
| N5 | retrieve_evidence 双路召回 | 调用 v3 问题编码，在相同元数据范围内运行本地 BM25 与 dense，并做 RRF | 融合后的候选子块 |
| N6 | rerank_evidence 重排序 | 对子块打分，选择主证据，再补充父级原文、去重并控制长度 | 主证据及上下文证据列表 |
| N7 | calculate_metrics 取数与计算 | 按查询计划取事实，核验口径，调用白名单公式 | 数值结果、公式、操作数来源 |
| N8 | compose_answer 生成答案或报告 | 将当前子问题的证据与计算结果交给云端 LLM 生成草稿，带 subtask_id | 当前子问题的 AnswerDraft |
| N9 | verify_answer 校验与输出 | 校验当前子问题，保存状态；全部处理后按原问题顺序汇总，检查有无漏项 | SubTaskResult、AnswerResult 和日志 |

路由规则：

```text
新报告：N1 → N2 → N3
仅入库任务：N3 → END
已有报告提问：N1 → N4
新报告并提问：N3 → N4

当前子问题为语义类：N5 → N6 → N8 → N9
当前子问题为数据类：N7 → N8 → N9
当前子问题为混合类：N5 → N6 → N7 → N8 → N9
N4 生成计划后，由条件边进入第一个可执行子问题的路径
N9 保存当前结果；还有子问题时进入下一项路径，否则汇总 → END
全局范围不明确：N4 → N9，输出 needs_clarification → END
全部子项均待澄清或不支持：N4 → N9，为每项记录状态并汇总 → END
单项业务失败：当前节点 → N9，记录该项原因；其他可独立回答项继续
```

子问题循环使用条件边和同一组节点，不反复调用 N4 重新拆题。待澄清或不支持的项由程序直接记录对应结果，不进入检索、计算和生成；若同时存在可执行项，保留这些状态并继续其余项。切换子问题时清空当前候选、证据及操作数工作区，历史结果按 subtask_id 保留。文件损坏、存储不可用等使整次运行无法继续的故障优先返回 failed，保留已完成子项；不伪装成某一项“资料不足”。

阶段性实现：文字问答小闭环中 N5 只启用 dense，N6 按 dense 排名选主证据并补上下文，明确标记 reranker 未启用；第 13 步再接 BM25、RRF 与 Cross-Encoder，执行完整检索链路。N7 尚未接通时，数字计算子问题显示 unsupported，不让 LLM 代算。9 个节点描述最终职责，并不要求第一阶段就接入所有后端。

数据问题的 N7 从结构化事实中读取已经关联的 PDF 证据，不必经过语义检索。若没有所需事实，返回资料不足；不能让模型临时编造数值。

### 4.3 工作流状态与恢复

建议用 `TypedDict` 定义图状态，Pydantic 校验节点边界处的业务对象。状态只保存必要数据或持久化对象 ID，避免把整个 PDF 和所有向量放进去。

关键字段：`run_id`、`task_type`、`report_ids`、`report_index_versions`、`question`、`query_plan`、`current_subtask_id`、`subtask_results`、`candidate_ids`、`primary_evidence_ids`、`context_evidence_ids`、`fact_ids`、`calculation_ids`、`answer_draft`、`answer_result`、`warnings`、`error_code`。`question` 始终保存用户原问，改写文本位于 QueryPlan；`report_index_versions` 固定各份资料的索引版本。主证据与补充上下文分别保存，候选、事实及结果还须关联 subtask_id，便于溯源、恢复和评测。

每次提问创建新的 `run_id` 和图执行状态；用户同一会话只继承明确的公司与年份，不继承上个问题的证据或计算结果。报告、任务和索引单独持久化。

为图配置持久化检查点，并以 `run_id` 对应独立的执行线程。检查点与报告存储不是一回事；业务文件与索引必须独立落盘。[LangGraph 持久化文档](https://docs.langchain.com/oss/python/langgraph/persistence)

网络超时最多重试 2 次；真实限流和可重试 5xx 使用同一上限及退避策略，遵守 Retry-After，不与客户端默认重试叠加。先结合服务商 code 判断原因：429 若表示 BudgetLimitExceeded 等预算或订购问题，也应停止；鉴权失败、余额不足、参数错误、字段缺失和口径冲突不盲目重试。一次请求超时可能已经产生服务商用量，重试也要记录。解析和入库使用幂等键，恢复执行不得重复插入数据。首版采用串行分支，避免混合路径并行写共享状态带来的合并问题。[百炼错误码](https://help.aliyun.com/zh/model-studio/error-code)

该重试预算同时适用于 Embedding 请求；超过单条长度的输入先回到切分层处理，不原样无限重发。当前批次失败即停止该版本的向量发布，保留已成功且校验过的批次供同一构建任务恢复；问题编码失败返回 EMBEDDING_UNAVAILABLE，不映射为 insufficient_evidence。

## 5 数据与证据怎样组织

### 5.1 年报获取

上传是必需入口。自动获取使用 AKShare 的 `stock_zh_a_disclosure_report_cninfo` 查询公告，保存标题、代码、公告时间与公告链接，然后解析候选公告得到 PDF 下载地址；公告详情链接不直接视为 PDF 地址。[AKShare 公告接口](https://akshare.akfamily.xyz/data/stock/stock.html#信息披露公告-巨潮资讯)

报告年度与公告年份分别保存。例如 2024 年年报通常在之后披露，不能仅用“公告年份等于 2024”筛选。候选中排除摘要，区分全文、修订版和更正说明；出现多个版本时显示给用户选择，保留历史版本。

每份 PDF 保存 SHA256、来源 URL、下载时间、公告时间、报告年度、公司代码和版本。公司与年份从文件中提取后，仍需在界面确认；识别失败时不能靠文件名直接入库。

默认限制单份文件 100 MB、1000 页，作为工程保护参数。下载只访问允许的公告来源，设置超时、大小上限和重定向校验；上传文件名不能成为磁盘任意路径。

### 5.2 解析与切块

1. 逐页提取文本、表格和坐标，保留原始解析结果。
2. 单独识别页眉、页脚及目录；清洗结果不能覆盖原文。
3. 识别章节标题和表格标题。跨页表格保存所有来源页，只有表头与列数相符时才合并。
4. 区分表格行的本期、上期、上期重述前后、合并及母公司口径。
5. 正文采用已确认的“按文档结构切分，再将过长内容递归拆小”方案：先识别章节、小标题与段落，过长时继续按段落、句子细分。正文 chunk size 初始目标 500、overlap 初始目标 80，均按第 3.6 节固定的 tokenizer 计数；仅同一结构单元内连续正文重叠，不跨章节或不同问询问题。只沿用切分尺，实际 Embedding 全部由 v3 生成；该配置尚未经检索评测验证。
6. 表格按“标题 + 单位 + 列头 + 行”转为可检索文本；数值计算仍读取规范化财务事实，不从生成后的表格摘要重新取数。
7. 为每块保存稳定 `chunk_id` 和来源页列表。重叠块评测时需按证据单元去重，不能靠重复文本提高指标。

输入扩展到招股说明书和问询资料时，招股书沿用章节层级，问询资料优先按“问题及对应回复”建立结构单元，长回复再递归切分；每块保留问题编号和关联关系。小表尽量完整保留，大表按完整行拆分，每块重复必要的表头、年份和单位，不对表格机械应用正文的 80 token 重叠。

物理页码统一使用从 1 开始的 `pdf_page_number`；印刷页码另外记录 `printed_page_label`。例如 PDF 第 18 页印着“15”，界面必须同时区分，不能只显示一个含义不明的“第 15 页”。

### 5.3 核心数据对象

| 对象 | 必须保存的主要字段 | 用途 |
|---|---|---|
| Report | id、stock_code、company、doc_type、fiscal_year、report_periods、disclosure_date、inquiry_round、sha256、source_url、revision、status、capabilities | 确定文档身份、内容期间、版本和已验证能力 |
| Page | report_id、pdf_page_number、printed_page_label、raw_text、parse_status | 定位原文 |
| ParentBlock | id、report_id、index_version、section_path、question_number、report_periods、text、page_refs | 保存一个小节、表格或问题回复的原文及页码映射 |
| Chunk | id、report_id、parent_id、section_path、question_number、report_periods、text、page_refs、chunk_token_count、tokenizer_version、parser_version | 按固定切分尺计数的检索子块及父块关联，不存伪造的 v3 计费 token |
| EmbeddingBatchResult | ordered_vectors、input_ids、model_profile_id、request_id、usage_total_tokens | 经校验且与输入 ID 对齐的向量批次；usage 缺失时为空 |
| IndexManifest | index_version、report_id、chunk_ids、embedding_profile_hash、tokenizer_version、distance_metric、build_status | 固定服务配置、块清单与索引发布状态 |
| FinancialFact | id、metric_code、value、raw_value、unit、raw_unit、currency、period_start、period_end、scope、restatement_status、report_id、source_id、validation_status、conflict_group_id | 可信取数 |
| SourceRecord | source_type、source_url、api_name、query_params、fetched_at、snapshot_hash、json_pointer、pdf_evidence_ids | 保留数据来源链 |
| Evidence | id、report_id、parent_id、chunk_ids、role、page_refs、quote、table_id、row_label、column_label、bbox | 保存主证据或补充上下文，并关联具体原文位置 |
| Calculation | id、formula_id、formula_version、operand_fact_ids、result、unit、warnings | 重现计算 |
| QueryPlan | task_type、original_question、rewritten_question、report_ids、hard_filters、route、subtasks、needs_clarification、clarification_reason | 保存总范围和一次生成的子问题计划 |
| SubTaskPlan | subtask_id、source_text、query_text、rewritten_query、route、report_ids、hard_filters、soft_hints、fiscal_years、report_periods、metric_ids、comparison、scope、needs_clarification、clarification_reason、section_id | 保存每项的原问依据、有效范围和执行参数；半年与全年必须明确起止日 |
| SubTaskResult | subtask_id、status、claims、primary_evidence_ids、context_evidence_ids、fact_ids、calculation_ids、warnings、error_code | 保存各项已核验答案或未完成原因 |
| AnswerResult | status、subtask_results、claims、calculations、citations、evidence_level、warnings、run_id | 按原问顺序展示各项结果 |
| EvalCase | question、route、report_ids、gold_evidence_ids、expected_values、expected_behavior、split | 固定评测样本 |

沿用 `Report` / `report_id` 作为通用文档记录名，以减少接口变化。`doc_type` 区分 `annual_report / semiannual_report / prospectus / inquiry_letter / inquiry_response / other`；财务报告保存 fiscal_year，其他资料可为空，多个内容期间保存在 report_periods。半年报明确保存 H1 及起止日，不能只凭年份与全年混用。disclosure_date 表示文件披露日期，统一替代原设计中的 announcement_date，未知时留空。inquiry_round、question_number、股票代码等不能推断的值同样留空，不编造。

`Evidence.role` 为 `primary` 或 `context`。父块和子块必须属于同一文件版本，保留原文到页码的映射；多个子块指向同一父块时，共用一次父级上下文，不重复发送。

Report.capabilities 分别记录 text_search、verified_financial_facts、akshare_supported 的状态与原因。招股书缺少股票代码或不适用当前财报接口时，可以完成文字入库，不强行匹配沪深股票；API 未适配不等于文件未披露。可计算范围由已经 verified 的事实及已启用指标决定，不能仅凭全文已可检索就开放所有数值计算。

`value` 使用十进制定点表达；JSON 及 SQLite 保存规范化十进制字符串，运算时转为 Decimal。Pandas 负责筛选和对齐，避免将金额先转成二进制浮点再进行精度敏感计算。

`scope` 至少区分 `consolidated` 与 `parent`；`source_type` 区分 PDF、AKShare 快照及人工核验修正。人工修正必须保留原值、修改理由和原文证据，评测结果需披露人工参与程度。

财务事实的 `validation_status` 取值为 `candidate / verified / missing / conflicted / superseded`。N7 仅使用 `verified` 且无未解决冲突的事实。人工修正生成新事实，并用 `supersedes_fact_id` 关联原记录；原记录标记为 superseded，不直接覆盖。报告正文可以已可检索，但相关财务事实尚不可计算，界面分别显示两种状态。

上述事实状态、原值和来源存储从第一阶段就定义；它们不要求先完成人工纠错页面。verified 必须附核验方式、核验时间和依据记录，不能仅因成功解析或人工点选就自动置为已核验。

### 5.4 PDF 与 AKShare 的关系

年报问答默认使用用户选定版本的 PDF 作为权威范围。AKShare 的三张财报用于结构化获取和交叉核对，不能静默覆盖 PDF：

- `stock_balance_sheet_by_report_em`：资产负债表候选接口。
- `stock_profit_sheet_by_report_em`：利润表候选接口。
- `stock_cash_flow_sheet_by_report_em`：现金流量表候选接口。

上述接口的股票参数通常包含交易所前缀，例如 `SH600519`，内部六位股票代码需由适配器转换。字段映射须由样本响应验证，不把示例字段或单位猜测写成固定事实。[AKShare 三张报表源码](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_three_report_em.py)

开发时用实际返回样本建立字段映射，并保存原始响应；上游新增或缺失列时明确报错。报告期、公告时间、获取时间、接口版本分别记录，当前接口快照不能冒充历史时点快照。[AKShare 财务报表接口文档](https://akshare.akfamily.xyz/data/stock/stock.html)

**只有 API 数值与选定 PDF 的对应行列、单位、期间和口径核验一致，才可挂接 PDF 页码。** API 独有数据的页码必须为空，并显示 API 来源；在“按本年报回答”的默认模式下，这类数据不能单独支撑财务结论。

核对优先依据源数据精度：例如 PDF 的金额精确到万元小数点后两位，匹配时使用其显示精度推导容差，并保存规则。不能用宽泛的百分比容差掩盖口径错误。

### 5.5 缺失与冲突处理

| 情况 | 系统行为 |
|---|---|
| 空值、NaN、`--`、无法解析的数值 | 统一为缺失，保留原因；不填 0 |
| 元、万元、亿元混用 | 转为统一金额单位并保留原单位；单位未知时阻止计算 |
| 同指标的 API 与 PDF 不一致 | 保存两份事实和差异；禁止自动拼接或平均 |
| 上年数据被重述 | 同表同比优先使用该报告披露的可比重述数；显式标记 |
| 一个值来自母公司，另一个来自合并报表 | 阻止计算，要求使用一致口径 |
| 分母为 0 | 返回不可计算原因，不输出无穷大或 0% |
| 同比基期小于 0 | 默认不给常规增长率，展示金额变化和负基数说明 |
| PDF 为扫描件或关键表格无法识别 | 标记需 OCR 或人工核验，相关数值题不通过验收 |

### 5.6 人工核验与纠错的迭代边界

最小闭环阶段先提供只读原文和数值对照；开发者可针对开发样本核对公司、期间、单位、合并口径及 PDF 行列位置，生成带核验记录的事实文件，再通过受控导入进入计算器。保留原提取值、修正理由、来源和人工参与标记，不能用无记录的手改数据库冒充自动核验。API 与 PDF 的自动核对在第 14 步接通适配器后按第 5.4 节执行。

人工纠错页面在第 12 步最小分析闭环验收后再排期：先统计真实错列、单位或期间错误，只对现有指标所需字段提供“查看原文—确认或修正—重新计算”。修改产生新事实版本，受影响的旧计算标记需重算；历史答案保留旧依据，新分析使用新 run_id。该页面不阻塞第 1—12 步，也不作为本轮提前实现的任务。

## 6 检索和任务路由怎样实现

### 6.1 意图识别与口语问题改写

目标是把用户的口语问题变成清楚的检索问题，同时保留原意。例如“这家公司主要靠啥赚钱？”改写为“公司的主营业务、主要产品和收入构成是什么？”。这一步优化问法，不生成答案，也不补造事实。

N4 按以下顺序处理：

1. 原样保存 `original_question`，读取界面已选公司和资料范围。
2. 规则识别明确出现的公司、日期、问询轮次、问题编号、指标、否定词和比较方向；一次正常 LLM 调用同时输出意图、改写问题和 subtasks，复用同一提示词与 QueryPlan，不增加独立的“拆题 Agent”。
3. 核对改写前后的约束：不得擅自增加年份、替换公司、丢失“未／不”等否定条件，或把“收入是否下降”改成“收入为什么下降”。
4. “利润”对应净利润还是归母净利润、“今年”对应哪个期间等歧义会影响结果时，对受影响项返回 `needs_clarification`，不先替用户作决定。全局公司或资料范围不明则停止整次执行；只影响一项时，其他明确项可以继续。
5. 用 Pydantic 校验 QueryPlan。格式错误或不合格改写最多修复一次；原问含义明确时可回退原问检索，关键歧义仍未消除则请求澄清。

N4 输出的 `QueryPlan` 示例：

```json
{
  "task_type": "question",
  "original_question": "南方乳业第二轮问询回复里，收入是咋确认的？",
  "rewritten_question": "南方乳业第二轮问询回复中披露的收入确认政策是什么？",
  "route": "semantic",
  "report_ids": ["example_selected_response_document"],
  "hard_filters": {
    "company": "南方乳业",
    "doc_type": "inquiry_response",
    "inquiry_round": 2
  },
  "subtasks": [
    {
      "subtask_id": "q1",
      "source_text": "南方乳业第二轮问询回复里，收入是咋确认的？",
      "query_text": "南方乳业第二轮问询回复里，收入是咋确认的？",
      "rewritten_query": "南方乳业第二轮问询回复中披露的收入确认政策是什么？",
      "route": "semantic",
      "report_ids": ["example_selected_response_document"],
      "hard_filters": {"company": "南方乳业", "doc_type": "inquiry_response", "inquiry_round": 2},
      "soft_hints": {"section_keywords": ["收入确认"]},
      "fiscal_years": [],
      "metric_ids": [],
      "comparison": null,
      "scope": null,
      "needs_clarification": false,
      "clarification_reason": null,
      "section_id": null
    }
  ],
  "needs_clarification": false,
  "clarification_reason": null
}
```

这是请求结构示例，不表示已取得或验证南方乳业第二轮问询回复。所有 report_ids 和过滤条件都必须来自已入库资料以及用户明确表达或界面选择；两者冲突且无法确定用户意图时先澄清。过滤条件另存来源记录，区分用户输入、界面选择和模型推测。fiscal_years 仅作年份标签，计算必须按 report_periods 的起止日区分全年、半年及明确时点；招股书和问询资料的期间匹配遵循第 6.5 节，不要求必须有单一年度。

普通问答统一使用 subtasks：有效可执行计划中单问为 1 项，复合问句为 2—3 项。每项 source_text 对应原问中的明确要求，query_text 只补入已经确认的指代，rewritten_query 用于规范检索。顶层 route 仅用于展示总体类型，实际执行以各项 route 为准；report_ids 必须为总范围的子集，hard_filters 只能继承或根据用户明确要求收窄。超过 3 项、无法可靠拆分或需要动态发现对象再规划的问题，请用户收窄，不能静默截断。全局澄清请求允许 subtasks 为空，但必须设置顶层 needs_clarification 并记录原因；空计划不能返回 answered。首版不引入任意依赖图、递归规划或自动扩大研究任务。

例如“收入增长多少？现金流是否改善？公司如何解释？”可对应 q1 收入计算、q2 现金流比较、q3 原文解释检索。期间及“改善”的指标口径须明确；解释检索不能预设收入已经增长。每项分别执行、保存证据和结果；一项缺资料不删除其他项的已核验答案。

QueryPlan 的 task_type 为 question 或 report。报告模板后续复用 SubTaskPlan，额外设置 section_id；sections 只保存章节顺序和 subtask_ids，不另设一套 retrieval_queries 执行协议。普通问答的 3 项限制不适用于固定报告模板，但模板项数必须由配置限定并在计划中完整列出。每个子问题各取最多 5 条主证据，记录各项上下文预算；每个 run 另记录计划调用数和累计 token 上限。预算耗尽时保留已完成项，其余明确标为未完成，不能假装全部章节均已读取。

每个子问题的 BM25 使用 query_text 及已确认的指标别名，dense 与 Cross-Encoder 使用 rewritten_query；单问题的 query_text 保留完整原问。生成答案同时提供总原问、当前子问题和该项证据。首版每个子问题在每路仅执行一次查询，不默认展开多个改写版本。

SubTaskResult.status 使用 answered、partial、insufficient_evidence、needs_clarification、unsupported、not_executed 或 failed；预算耗尽而未执行的项标记 not_executed 并记录原因，执行中的 pending/running 单独记录，不作为最终答案状态。N9 确保每个计划 ID 恰有一个结果，按原问顺序编号展示，不再调用模型自由扩写综合结论。汇总优先判断全局故障：存在则为 failed 并保留已完成项，未执行项记录原因；全局澄清则返回 needs_clarification。其余情况下，非空计划全部项完整才为 answered；有任何已支持答案但仍有缺口则为 partial；全无答案时按 needs_clarification、unsupported、failed、not_executed、insufficient_evidence 的顺序选总状态，逐项原因仍全部保留。这里检查的是用户明确子问题是否漏答，不承诺自动穷尽“全部风险”等开放集合。

### 6.2 两路召回

以下为第 13 步接通后的完整检索配置；第一阶段仅运行其中的 dense 路线，仍执行相同的元数据过滤和父级原文补充。

1. 按第 6.5 节将用户明确范围和元数据条件转为同一份允许的文档／子块清单。
2. BM25 使用 jieba 分词并加入“归属于母公司股东”“经营活动现金流量净额”等术语。保留否定词、年份、金额和关键限定词。
3. text-embedding-v3 以 query 用途对通过校验的改写问题生成向量，在本地子块索引中进行 dense 检索；文档以 document 用途编码，遵守同一模型配置、维度及归一化约定。
4. 两路各取最多 30 个候选。BM25 对允许的报告块合并成一个查询语料范围后计算排序，不直接拼接不同单报告索引的原始得分；向量检索使用等价的元数据过滤。
5. 无匹配结果时返回空证据，不回退到其他公司。

### 6.3 RRF 融合与重排

设计起始参数：两路各 30 个子块 → RRF 合并并去重 → 取前 30 个子块重排 → 保留最多 5 条子块主证据 → 补充父级原文上下文 → 去重后交给 LLM。

RRF 使用名次，不直接相加 BM25 和向量相似度：

```text
RRF(d) = Σ 1 / (60 + rank_i(d))
```

名次从 1 开始；候选未出现在某一路时，该路贡献为 0。`60` 是本项目初始常数，权重先相同；在开发集确定后冻结。

该算法按名次融合的定义可参照 [RRF 官方技术说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)；本项目不需要因此引入 Elasticsearch。

Cross-Encoder 对问题和候选文本成对评分，重新排序。建议使用 FlagEmbedding 的 `FlagReranker` 调用 `bge-reranker-v2-m3`。[官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)

重排只能重排已有候选，不能找回前一阶段没有召回的内容。表格必须保留列头和单位；超过模型输入长度时，按表头和行拆分，不能静默截掉关键年份。

主证据选定后，按第 6.6 节补充父级原文上下文。Precision@5 只统计扩展前的最多 5 条子块主证据；扩展文本不能额外计为召回成功。

### 6.4 索引版本

模拟客户端仅用于显式开发/测试模式。模拟向量、缓存、索引及运行记录使用独立 mock 配置和存储命名空间；不能标成真实 v3 输出、计费或评测成绩。切到真实模式须重新编码和建库，禁止复用模拟向量。正式模式缺少 Key 时明确报告配置未完成，不能自动切换到模拟回答。

索引版本由 PDF 哈希、解析器版本、父子块划分、元数据结构版本、切块参数及 tokenizer 版本、分词词典、已启用索引类型，以及 embedding_profile_hash 共同决定。该配置哈希覆盖服务商、地域/API Host、model_id、可取得的服务版本、维度、document/query 约定、归一化与距离度量。启用双路时，BM25 和向量索引使用相同子块清单；父级原文与映射随索引版本一起发布。云端不提供固定版本时记录构建日期及已知限制，不能宣称完全可复现。

第 5 步实现最小批次恢复：按“Embedding 配置哈希＋用途＋实际发送文本哈希”保存成功结果和构建进度，只复用同配置且已校验的向量。新版本先写临时集合；返回结果按 text_index 对齐并通过数量、维度、有限数值及父子关联检查，全部预期子块完成后才在 SQLite 事务中切换 active_index_version。任何批次失败或缺项都不能发布半成品；旧版本保持不变，恢复时只重发未成功部分。服务端已计费但客户端未收到结果的请求可能再次收费，不能承诺付费请求绝不重复。

旧版本在新版本发布成功前保持可查询。已开始的查询固定使用启动时的版本，不能一半读取旧块、一半读取新向量。多报告查询在启动时一次读取并固定版本映射；参与查询的报告须使用一致的 embedding 模型、维度、距离度量及分词配置，不一致时先重建兼容索引。检索接口接收所有选定报告的清单集合。

### 6.5 元数据辅助召回

元数据用于确定“在哪些资料里找”，并在命中后定位原文。招股书和问询资料采用以下字段：

| 类别 | 字段 | 作用 |
|---|---|---|
| 文档范围 | company、doc_type、inquiry_round、revision、report_id | 区分公司、资料类型、问询轮次和文件版本 |
| 内容位置 | section_path、question_number、parent_id、page_refs | 找到章节、问题、父级原文及对应页 |
| 时间 | report_periods、disclosure_date | 分别记录内容涉及期间和文件披露日期 |

例如“南方乳业第二轮问询中，关于收入确认是怎么回复的”，先确定公司、第二轮、问询回复的资料范围，再做关键词和向量检索；“收入确认”可作为章节提示，不要求章节标题必须恰好叫这个名字。

用户明确指定或在界面选定的范围形成 `hard_filters`，两路召回及后续上下文补充必须共同遵守。用户指定的版本必须精确匹配；未指定且存在会影响答案的版本冲突时先澄清，不把不同版本拼成一份回答。

模型推测的章节等信息放在 `soft_hints`，用于构造检索表达和展示提示，首版不将其直接变成排除条件。未知元数据不能伪装成匹配：如用户指定第二轮，而某文件轮次未确认，应提示确认文件身份；不能偷偷放宽过滤。

一份招股书可以涉及多个财务年度。期间条件优先核对明确标注的块级期间，结合文件内容覆盖范围选择候选；没有块级期间标签时保留候选并在原文核验，不因标签缺失直接排除。不能用披露年份代替内容报告期，也不能把全年、半年和单季度视为同一期间。复杂期间匹配在业务层解析，再向两路检索传入一致的允许清单。

### 6.6 小块召回与父级原文上下文补充

首版确定采用“小块向量＋原文上下文补充”。500 tokens / 80 tokens 仍是正文子块的切分起点，不是父块长度或最终 LLM 输入长度。

1. **入库保存两层内容。** 父块保存一个小节、表格或某个问题对应回复的原文；子块是父块内递归拆出的短文本，通过 parent_id 关联。父块不跨文件、版本或不同问询问题，表格保留标题、年份、单位及表头。
2. **检索与重排使用子块。** 按阶段先建立子块 dense 向量，第 13 步增加相同子块的 BM25 索引；父块原文持久化存储，不额外为整份父块建立向量。
3. **命中后补原文。** 启用重排后，N6 先完成子块重排，再取相应父块；基线阶段按 dense 排名选主证据后补原文。同一父块的多个命中合并处理。问询回复需要保留该文件内对应问题的题干或编号，避免回复脱离问题。
4. **控制发送长度。** 父块放得下时提供完整父块；太长时保留命中段及必要相邻段、标题和表头，标记为局部上下文，不截取整个章节前半段冒充完整回复。扩展内容去重，优先保证主命中证据。
5. **保留精确引用。** 主证据和扩展上下文分别标记 role，记录实际发送的片段及页码；不能因为补入父块就把父块所有页面都列成支持答案的引用。

`context_token_budget` 保存在 DeepSeek 配置中，优先使用与服务版本核对过的计数方式；没有精确 tokenizer 时明确记为估算并保留余量，以实际 API usage 校准，不能用 500/80 切分尺或 Embedding usage 当作 LLM token 数。从预算中扣除提示词、原问、输出预留和安全余量。超限时减少整条补充上下文，主证据仍放不下则缩小任务并说明覆盖限制，不静默截掉关键数字、表头或问题条件。上下文补充只读取已选文件版本，仍遵守用户明确的检索范围。

### 6.7 摘要补充向量的后续实验

摘要向量不属于首版必做功能。后续可以对同一段原文增加摘要向量作为额外检索入口，观察是否改善口语和概括性问题的召回。

每个摘要必须关联原文块及版本；命中摘要后先取回原文候选，再按原文去重、重排和核验。摘要不能替代 PDF 原文、财务事实或计算结果作为证据。同一原文命中多个向量表示时也不能重复占据主证据名额。

该实验独立记录摘要生成模型、提示词版本、索引成本、质量变化和耗时。默认配置关闭；没有实验结果时不声称它一定优于首版方案。

## 7 财务计算和可信输出

### 7.1 首版指标白名单

直接取数支持营业收入、营业成本、净利润、归母净利润、总资产、总负债、流动资产、流动负债、经营活动现金流量净额。年度收入、利润和现金流量使用完整年度，半年报使用明确的 1—6 月累计期间；同比只比较上年同期，不把半年乘二或混用全年。资产负债使用期末时点，半年报期末与上年末的变化只能称较上年末变化。江丰电子半年报先用于 PDF 财务事实开发，AKShare 半年报自动适配不作为本轮新增前置。

| 指标 | 公式 | 额外约束 |
|---|---|---|
| 营业收入同比 | `(本期收入 / 可比上期收入 - 1) × 100%` | 上期必须大于 0 |
| 归母净利润同比 | `(本期归母净利润 / 可比上期归母净利润 - 1) × 100%` | 基期大于 0；负基数另行说明 |
| 毛利率 | `(营业收入 - 营业成本) / 营业收入 × 100%` | 收入大于 0；非金融企业口径 |
| 净利率 | `净利润 / 营业收入 × 100%` | 使用净利润，不冒用归母净利润 |
| 资产负债率 | `期末总负债 / 期末总资产 × 100%` | 同一报表时点，总资产大于 0 |
| 流动比率 | `期末流动资产 / 期末流动负债` | 分母大于 0；输出“倍” |
| 经营现金流收入比 | `经营活动现金流量净额 / 营业收入 × 100%` | 同一起止期间、同口径，收入大于 0 |
| 毛利率同比变化 | `本期毛利率 - 上期毛利率` | 输出“百分点”，不是百分比增长 |

这些是本项目采用的分析定义，不宣称等同于所有数据供应商的指标口径。复杂 ROE、杜邦分析及金融行业专用指标在后续阶段增加，不能用简单期末权益比率冒充加权平均 ROE。

归母净利润不等于母公司单体利润表净利润；毛利率的成本字段是营业成本，不是营业总成本。负净利润与负经营现金流可以是有效数值，不能一律当缺失或脏数据。

### 7.2 计算执行方式

模型仅选择 `metric_id` 和参数；程序从注册表找到预先编写的计算函数。禁止把模型生成的 Python、SQL 或表达式直接交给 `eval`、`exec` 或不受约束的执行器。

计算顺序：筛选事实 → 对齐公司与期间 → 校验币种/单位/口径 → 处理缺失与基期 → Decimal 运算 → 按展示规则舍入 → 保存 Calculation。

金额展示可以选择元、万元或亿元，内部计算保持统一单位。比例展示保留 2 位小数，内部结果不提前舍入。每个结果同时保存公式版本与操作数 ID。

### 7.3 一个完整输出示例

以下数值仅用于说明展示格式，不是真实公司数据：

```text
结论：2024 年营业收入同比增长 20.00%。

本期营业收入：120.00 亿元。
可比上期营业收入：100.00 亿元。
计算公式：(120.00 / 100.00 - 1) × 100% = 20.00%。
口径：人民币、合并报表、完整年度。
来源：所选年报合并利润表；本期与上期列分别对应已核验单元格。
引用：[E1] 和 [E2]，均可打开对应 PDF 物理页。
证据状态：完整；数值、期间、口径和来源均已核对。
```

### 7.4 置信度怎样呈现

首版将“置信度”实现为可解释的**证据完整程度**，使用 `high / medium / low`，界面显示“证据完整 / 证据部分完整 / 证据不足”。这不是经过统计校准的正确概率。

- 完整：所有关键结论有来源，所有计算操作数通过校验，没有来源冲突。
- 部分完整：一部分问题可回答，另一部分缺少证据；逐项说明已支持和未支持的结论。
- 不足：关键取数缺失、口径冲突或主要结论无证据；返回资料不足或澄清请求。

不能把模型自报的 95%、向量相似度或重排分数直接当作答案正确率。即使来源完整，文字解释是否忠实仍要通过逐条核对和人工评测判断。

### 7.5 最终校验

N8 生成 `claims[]`，每条包含文本及 `evidence_ids`、`fact_ids`、`calculation_ids`。N9 检查 ID 是否存在、是否属于所选报告、页码是否有效，以及引用原文是否存在于对应解析页。直接取数的金额对照已核验 FinancialFact，派生指标对照 Calculation；不能要求所有直接披露的原始值都必须有计算记录。

其中 FinancialFact 核对适用于已接通的规范化财务取数。产品单价、机器人负载等原文披露查证，核对实际引用中的数字、表头、单位及限定词；不将它们标为已规范化财务事实，也不直接参与财务公式。预测、超过某金额、含意向协议等条件必须随原文保留。

草稿及每条 claim 带 subtask_id。N9 检查其证据、事实和计算属于当前子问题的有效输入，不允许直接引用另一子问题尚未核验的结果；同一原文可服务多个子问题，但必须分别被选入对应输入并通过范围校验。汇总前校对子问题清单：每项必须有答案或具体未完成原因，不能把未执行项写成“未找到资料”。

引用的 evidence_ids 还必须属于本次实际发送的 primary / context 证据集合，引用文字须落在实际发送的原文片段内，并满足本次生效的 hard_filters（如公司、轮次、版本及明确的问题编号）。同一父块中被过滤或因预算省略的内容，不能仅凭文件 ID 和页码有效就通过校验；分章节生成时分别检查各章实际提供的证据。

数值和来源采用确定性校验。语义支持可补充模型辅助检查，但不能把模型互审视为事实保证。未通过的结论删除或转成资料缺口；首版不设置无限生成重试。

年报及检索文本作为数据进入提示词，其中即使出现“忽略规则”等语句也不得改变系统行为。工具调用限制在已定义的取数、检索和公式接口内。

## 8 建议代码结构与接口

以下目录是拟创建结构，本次仅生成开发文档，不表示这些代码已经存在。

```text
annual_report_agent/
  pyproject.toml
  uv.lock
  .env.example
  README.md
  configs/
    app.yaml
    metrics.yaml
    field_mapping.yaml
    model_profiles.yaml
  src/annual_report_agent/
    api/                 # app.py、reports.py、tasks.py、analysis.py
    ui/                  # app.py 和 4 个页面
    schemas/             # report.py、finance.py、query.py、answer.py、model.py、index.py
    ingestion/           # acquisition.py、parser.py、chunker.py、indexer.py
    finance/             # akshare_adapter.py、normalizer.py、calculator.py
    retrieval/           # embedding_client.py、bm25.py、dense.py、fusion.py、reranker.py、metadata.py、context.py
    graph/               # state.py、builder.py、planner.py、9 个节点及子问题循环
    generation/          # backend.py、cloud_client.py、prompts.py、composer.py、verifier.py
    storage/             # sqlite_repo.py、files.py、index_manifest.py、embedding_cache.py
    evaluation/          # dataset.py、metrics.py、runner.py
    cli.py
  tests/
    fixtures/
    unit/
    integration/
    acceptance/
  data/                  # raw、parsed、snapshots、indexes、app.db
  evals/                 # 题集、标注说明、实验配置
  outputs/               # 分析报告、逐题日志和汇总
  scripts/               # start.ps1、环境检查脚本、smoke_models.py
  docs/
```

采用分层函数接口，避免将所有逻辑塞进一个 Agent 类：

| 接口草案 | 责任 |
|---|---|
| `parse_report(report_id: str) -> ParsedReport` | 生成按页文本与表格，不调用生成模型 |
| `normalize_facts(raw: RawStatement, context: ReportContext) -> list[FinancialFact]` | 映射字段、单位与期间，保留异常 |
| `encode_batch(texts: list[str], input_ids: list[str], purpose: Literal["document", "query"], profile: EmbeddingProfile) -> EmbeddingBatchResult` | 每批最多 10 条，调用 v3；按响应索引对齐 ID，校验及归一化向量，记录请求用量 |
| `build_index(report_id: str, config: IndexConfig) -> IndexManifest` | 构建当前配置启用的索引及父子映射，并发布同一版本 |
| `plan_query(request: AnalysisRequest) -> QueryPlan` | 一次正常模型调用生成意图、改写和子问题；程序校验范围及能力 |
| `retrieve(task: SubTaskPlan, manifests: list[IndexManifest]) -> list[Candidate]` | 固定版本和当前子问题范围，按配置运行 dense 或双召回与 RRF |
| `rerank(question: str, candidates: list[Candidate]) -> list[Evidence]` | 重排并选择主证据 |
| `expand_context(primary: list[Evidence], task: SubTaskPlan, budget_tokens: int) -> list[Evidence]` | 返回当前项的主证据及去重父级原文，标记 primary / context |
| `calculate(task: SubTaskPlan, facts: list[FinancialFact]) -> list[Calculation]` | 按当前项调用指标白名单并保存依据 |
| `compose(original_question: str, task: SubTaskPlan, evidence: list[Evidence], facts: list[FinancialFact], calculations: list[Calculation]) -> AnswerDraft` | 只用当前项的有效输入生成带 subtask_id 的回答 |
| `verify(draft: AnswerDraft, context: VerificationContext) -> SubTaskResult` | 检查当前项的引用、事实与计算，返回逐项结果 |
| `aggregate(plan: QueryPlan, results: list[SubTaskResult]) -> AnswerResult` | 程序检查漏项、汇总状态，按原问或报告章节顺序展示 |

表中业务类型由 `schemas` 模块定义，列表是设计契约，不是可直接运行的代码。EmbeddingProfile 与 EmbeddingBatchResult 放在 schemas/model.py，IndexConfig 与 IndexManifest 放在 schemas/index.py；purpose 决定 query/document 请求用途。两个 API 客户端共享超时/日志约定，但保留不同请求与响应适配，不共用错误的消息体。

所有模式均取当前 SubTaskPlan 调用检索和计算接口；顶层 QueryPlan 不直接作为执行参数。启用重排时 N6 调用 `rerank(task.rewritten_query, candidates)`，再调用 expand_context；向量基线模式使用候选排名选前 5 条并记录配置。compose 保留总原问及当前项；VerificationContext 包含该项有效过滤条件、索引版本、实际发送的证据片段、事实与计算 ID。N9 每项调用 verify，最后调用 aggregate。数字问答的事实原文证据也须计入实际发送记录和 token 预算。

建议 API：

| 方法与路径 | 用途 |
|---|---|
| `POST /api/reports/upload` | 上传文件，返回 report_id 与 task_id |
| `GET /api/reports/search` | 按代码与年度返回公告候选 |
| `POST /api/reports/import` | 导入用户确认的候选，返回 task_id |
| `GET /api/reports` | 列出报告及处理状态 |
| `GET /api/tasks/{task_id}` | 查询进度、阶段和错误 |
| `POST /api/analysis` | 提交问题或报告任务，同时返回 task_id 与 run_id |
| `GET /api/analysis/{run_id}` | 获取最终答案、公式和引用 |
| `GET /api/reports/{id}/pages/{page}` | 获取证据页图片及对应原文 |
| `GET /api/analysis/{run_id}/export` | 下载 Markdown 分析报告 |
| `POST /api/evaluations` | 提交指定配置的评测任务，返回 task_id 与 eval_run_id |
| `GET /api/evaluations/{eval_run_id}` | 读取配置、逐题结果地址和汇总指标 |

AnalysisRequest 保留用户原始 question、已选 report_ids 和用户明确的文档过滤条件；这些字段属于输入。original_question、rewritten_question、条件来源及实际生效范围保存在 QueryPlan 和运行记录中，随分析结果返回供界面核对。不得把模型生成的过滤表达式直接当作可执行查询，必须先映射到允许字段并校验。

上传、解析、API 调用和评测都可能耗时。首版使用后端管理的单执行队列与 SQLite 任务记录，前端轮询状态；重启后将遗留 running 任务标为 interrupted，由用户重试或从已完成阶段恢复。API 服务采用一个 worker，避免重复队列和并发写入；后续本地重排也避免多进程重复加载权重。[FastAPI 部署与内存说明](https://fastapi.tiangolo.com/deployment/concepts/)

任务查询响应包含 `task_id`、`status`、`progress`、`resource_type`、`resource_id` 和 `error_code`。分析任务的 resource_id 为 run_id，评测任务为 eval_run_id，前端成功后据此读取结果。

任务状态统一为 `queued / running / succeeded / failed / interrupted`；答案及子问题结果状态为 `answered / partial / insufficient_evidence / needs_clarification / unsupported / not_executed / failed`，汇总遵循第 6.1 节。前者表示任务执行情况，后者表示答案是否充分，不能混为一谈。正常返回澄清或资料缺口可属于 succeeded，进程崩溃或全局故障则为 failed/interrupted。

## 9 每一步具体该做什么

按实际依赖推进，每一步留下可核对的产物。第 1—8 步形成文字问答小闭环；第 9—12 步接入可信计算，形成最小分析闭环；第 13—16 步按需要逐项完成后续能力。一个分支失败只阻止依赖它的功能，不让财务接口拖住全文检索。原先 20—30 个有效工作日的整包估算不作为承诺，每个阶段根据实际验证结果重新估算。

用户将在代码开发完成后自行配置 API Key。因此下列步骤分为“代码与本地流程验证”和“真实云端效果验收”：前者可使用真实 PDF、固定模拟响应和本地存储完成，后者待配置后执行。每次运行标明 mock 或 live；模拟模式不证明语义召回质量、LLM 理解能力或回答正确率。此规则适用于第 5—8、11—12、15—16 步中涉及云端的检查，不阻止继续编写后续模块。

### 第 1 步 确定第一阶段样本与小题集

- [ ] 登记已提供的南方乳业招股书、两份越疆问询回复及江丰电子 2026 年半年报，确认实际路径、哈希、公司、类型、版本和物理页码。
- [ ] 先标注 5—10 道开发问题及原文页码，覆盖正文、表格、上下文和无证据情况；这些题属于开发用例，不冒充独立测试。
- [ ] 以江丰电子 2026 年半年报及其中 2025 年同期数作为第一份财务开发样本；全年年报与留出公司后续补充。问询轮次和修订版本未核实的部分单独记录。
- [ ] 年报评测逐步扩至 2 家开发公司和 1 家留出公司，避免预先查看留出题目来改规则；不要求开工前集齐六份报告或完成 50 题。

产物：evals/sample_manifest.json、可读取 PDF、小题集及缺失资料清单。<br>
验收：至少一份真实 PDF 可读取，每道当前开发题有答案依据或明确的无证据预期。

### 第 2 步 建立最小环境和接口验证工具

- [ ] 创建 Python 3.11 项目环境、pyproject.toml、.env.example 和 .gitignore；首批安装 PDF、httpx、Pydantic、Chroma 与 tokenizers 所需依赖，重排的 PyTorch/FlagEmbedding 后置。
- [ ] 用 pdfplumber 提取正文和表格，用 pypdfium2 渲染证据页；只获取固定 tokenizer 文件，验证 CPU 计数、特殊 token 设置及版本哈希。
- [ ] 在 .env.example 列出 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY、DASHSCOPE_EMBEDDING_URL 和模式配置；实现缺失提示，密钥缺失时只报告变量名。用户开发完成后自行填写实际配置。
- [ ] 准备 scripts/smoke_models.py 的真实探针：v3 编码两条文档和一个 query，检查 1024 维、text_index 与向量有效性；显式写入临时 Chroma 集合并用 query_embeddings 查询。开发期间先以固定模拟响应验证相同契约，真实探针标记待配置。
- [ ] 在同一脚本准备 `deepseek-flash` 显式关闭思考的 JSON 探针，使用脚本内的小型校验对象，不提前依赖第 3 步的 QueryPlan；先用模拟响应检验 JSON 及错误处理。live 运行时才记录真实 request_id、usage 和耗时。
- [ ] 建立 uv.lock、model_profiles.yaml 和本地检查记录，不等待真实 API 调用。正式复用客户端在第 3 步实现；本地重排第 13 步再验证，Ollama/MLX 不作为前置。

产物：独立环境、锁文件、配置模板、tokenizer 版本、模拟检查记录和待运行的真实探针。<br>
开发验收：PDF 提取、CPU 计数、模拟 document/query 契约、Chroma 显式向量写读及 JSON 校验通过即可继续。用户填写 Key 后再检查实际地域、权限、真实编码和生成；此前联通状态为待验证，不用模拟输出冒充成功。

### 第 3 步 先定义共用契约和存储

- [ ] 定义 Report/Page/ParentBlock/Chunk/Evidence、AnalysisRequest、QueryPlan、SubTaskPlan、SubTaskResult、AnswerDraft/Result 及 VerificationContext。
- [ ] 定义财务事实和计算契约、各类状态、单位和来源规则；提前固定过滤条件与子问题 ID 的含义。
- [ ] 定义 EmbeddingProfile、EmbeddingBatchResult、IndexConfig/Manifest 与用量记录；在 retrieval/embedding_client.py、generation/cloud_client.py 实现各自适配器，N4/N8 共用生成客户端。
- [ ] 实现 SQLite 存储、最小迁移、文件哈希、任务记录、来源记录和索引清单，以及 storage/embedding_cache.py 的成功批次保存；配置与实际文本哈希不同不能复用向量。
- [ ] 验证重复上传、非法字段、子问题越权范围、未知元数据、单问与三子问题的结构化输入。
- [ ] 用受控响应验证返回数组乱序时的 text_index 对齐、缺项、错维度、非有限值、鉴权失败、预算型 429 与有限重试；这些客户端用例不替代用户配置 Key 后的真实验收。

产物：schemas/*.py、storage/*.py、两家云端客户端、数据库初始化和契约测试。<br>
验收：后续检索、改写、上下文和计算模块使用同一组类型；普通问答 1—3 项、ID 唯一、范围不扩大。先用手工固定的合法计划验证模块，不依赖尚未完成的 LLM 规划。

### 第 4 步 上传解析并保存父子块

- [ ] 完成上传 API、文件限制、身份确认、单执行队列和任务状态查询。
- [ ] 按页保存正文、表格、解析状态与原文坐标；抽查至少 10 页或文件全部页面（取较少者）。
- [ ] 按章节、表格或问题回复保存父块，超长正文递归拆分；按第 3.6 节固定切分尺执行 500/80，记录 chunk_token_count 和 tokenizer_version；标题与表头计入实际待编码文本。
- [ ] 保存父子关系、文档类型、内容期间、轮次、版本与精确页码；实现证据页图片接口。

产物：ingestion/parser.py、chunker.py、结构化页面及父子块、证据页接口。<br>
验收：正文重叠不跨结构单元，表格重复必要表头，未知轮次留空；本步完成解析落盘，向量入库在第 5 步完成；财务适配不适用不阻断文本处理。

### 第 5 步 向量检索 元数据过滤与原文补充

- [ ] 通过 encode_batch 以 document 用途逐批编码子块，按最多 10 条串行调用并保存成功批次；问题以 query 用途编码。统一 1024 维、L2 归一化、cosine 距离；禁用 Chroma 隐式编码器。
- [ ] 原文父块和页码映射单独持久化，新向量写临时集合；所有预期块通过校验后原子更新活动索引，按第 6.4 节恢复中断批次。长度错误回切分层更新块清单，不丢块发布。
- [ ] 实现 metadata.py，用固定 SubTaskPlan 验证公司、类型、轮次、版本及内容期间的有效范围。
- [ ] 向量基线按候选排名选最多 5 条主证据，调用 expand_context 补父级原文、去重并控制生成模型 token 预算。
- [ ] 验证超长父块、同父块重复命中、版本固定、重启查询、编码中途失败后恢复与旧索引保持有效；查询 API 失败应报错，不算空召回。摘要向量默认关闭。

产物：retrieval/dense.py、metadata.py、context.py、云端批次及索引清单、前 5 条证据记录。<br>
验收：小块能找到同版本父块，发送原文片段有精确引用，过滤不过界；明确记录这是未启用 BM25 和重排的向量基线。

### 第 6 步 在 N4 实现理解 改写与子问题计划

- [ ] 在 graph/planner.py 中经共用云端后端调用 `deepseek-flash`，一次正常调用输出意图、改写和 1—3 个明确子问题。
- [ ] 核对每项 source_text、否定、年份、公司和范围，校验模型结构；无效输出最多修复一次。
- [ ] 单问走同一契约；复合问句按原问顺序列项。对局部歧义、全局歧义、超出三项和尚未启用的计算能力分别处理。
- [ ] 保留原问和过滤条件来源；追问只继承明确范围，不沿用上轮证据。

产物：plan_query、模型提示词、QueryPlan 样例及针对性用例。<br>
验收：不把“是否下降”改成“为什么下降”；三问不会少一问；计划不自行扩大任务。此步只验收计划生成，不声称计算分支已经完成。

### 第 7 步 先完成逐项回答 引用校验和汇总

- [ ] 实现 compose、verify 和 aggregate，可独立于 LangGraph 调用；compose 复用 DeepSeek 云端后端，每个草稿和 claim 带 subtask_id。
- [ ] 只传入当前项的实际证据，检查引用片段、范围、页码、原文及预算；校验失败不输出该结论。
- [ ] 为每个计划 ID 保留一个结果：已回答、部分回答、无证据、待澄清、不支持、未执行或失败。
- [ ] 验证两项都成功、一项成功另一项无证据、局部澄清、超预算未执行、跨子问题引用被拒绝等情况。

产物：generation/composer.py、verifier.py、aggregate 逻辑和逐项回答模板。<br>
验收：每个问题有结果或原因，总状态正确；未经执行不能写成“没有披露”，汇总不自由补造新结论。数字校验此时只测接口契约，第 10—11 步用真实计算接通。

### 第 8 步 接入工作流和文字问答小界面

- [ ] 将已验证的 N1—N6、N8/N9 业务接入 LangGraph 条件边；N7 未接通时明确返回 unsupported。
- [ ] 实现按 subtask_id 串行执行和恢复，切换时清空临时证据；加入检查点、幂等和基本错误记录。
- [ ] 制作资料上传、提问和证据查看的最简界面，复用后端任务状态，不直接操作数据库。
- [ ] 用显式模拟客户端运行小题集对应的流程用例，验证上传、两三个文字子问、重启和失败路径；先保存最小回归记录。配置 Key 后重新运行真实小题集，才记录真实答案质量和两家用量，不等第 15 步完整评测界面。

产物：graph/state.py、builder.py、节点模块、FastAPI/Streamlit 文字小闭环。<br>
开发验收：真实 PDF 解析与模拟问答流程能展示逐项结果和原文，第二次提问不重做解析，页面明确显示模拟模式。用户填写 Key 并完成真实建库及小题集验证后，才声明文字问答实际可用；不等计算、自动下载或多模型实验。

### 第 9 步 从真实财务报告形成可核验的事实

- [ ] 使用已提供的江丰电子 2026 年半年报，定位现有公式需要的两期数据；以 2026H1 对 2025H1 比较收入和利润，以 2026-06-30 对 2025-12-31 保存资产负债时点，先做最少字段。
- [ ] 实现 PDF 事实提取与 normalizer，保存原值、单位、期间、合并口径、行列位置及证据。
- [ ] 按第 5.6 节生成可追溯的开发核验记录和受控导入；仅核验通过的事实成为 verified。
- [ ] 处理缺失、冲突和不适用情况，文字入库与财务能力分别显示。AKShare 交叉核对移到第 14 步，先不作为 PDF 计算的必需前置。

产物：finance/normalizer.py、带核验记录的财务事实及原始抽取记录。<br>
验收：至少一项计算所需的两期操作数可在真实 PDF 找到，单位及期间明确；人工核验比例如实记录。半年报样本可验证当前 PDF 取数和计算，但不能冒称已验证全年年报或 AKShare 适配。参考样本核对完成也不等于应用自动提取通过。

### 第 10 步 实现计算器并接通 N7

- [ ] 先完成收入同比和毛利率，再逐项实现第 7 节其余指标；未实现的 metric_id 明确返回 unsupported。
- [ ] 先写确定输入与预期输出，再实现 Decimal 固定公式；Pandas 只负责筛选和对齐。
- [ ] 覆盖缺失、零分母、负基期、单位、合并口径和重述；操作数只使用 verified 事实。
- [ ] 保存公式、操作数及来源，接通 N7，并补齐第 7 步预留的数字和证据校验。

产物：finance/calculator.py、指标注册表、N7 与计算测试。<br>
验收：120/100 的同比为 20%，30% 到 32% 的变化为 2 个百分点；真实样本结果有原文来源，未核验输入被阻止。

### 第 11 步 联调数字和混合子问题

- [ ] 使用真实已核验事实，执行文字、数字和混合路线；同一问句最多 3 个明确子问题。
- [ ] 逐项验证各自的过滤条件、证据、操作数和引用，不让另一项结果覆盖当前项。
- [ ] 验证“收入变化＋公司解释”等案例；公司未披露解释时保留计算结果并明确文字缺口。
- [ ] 接通问答页面的公式、逐项状态和事实来源；真实问询原件未取得时，轮次专项仍标记未验证。

产物：三类问题的完整节点轨迹、跨子问题隔离与缺口案例。<br>
验收：两问都成功和一问成功一问失败均可解释；相同问题在重启后仍使用固定来源与版本。

### 第 12 步 交付最小分析闭环

- [ ] 完成上传、检索、可信计算、逐项答案和只读证据查看，不把人工纠错页面作为当前验收前置。
- [ ] 整理已支持的输入、指标、已验证样本与未验证项，运行这一阶段相关的单元、集成和验收用例。
- [ ] 提供 start.ps1、配置样例、安装说明和最小备份恢复流程，按锁文件验证项目环境。
- [ ] 保存阶段运行记录及失败案例，据此安排下一项改进；人工纠错页面可在此后按第 5.6 节单独迭代。

产物：可启动的最小应用、阶段验收记录、真实错误清单。<br>
开发验收：界面、工作流与确定性计算通过本地用例，必要事实未经核验时明确停算；模拟界面不冒充真实分析。用户填写 Key 后运行真实探针、PDF 建库、开发小题集与端到端检查，通过后即可实际使用和迭代，不要求第 13—16 步同时完成。

### 第 13 步 增加 BM25 RRF 和重排

- [ ] 在现有相同子块及过滤范围上增加 jieba/rank-bm25，不重写解析与父子块存储。
- [ ] 此时再安装并验证 PyTorch/FlagEmbedding 与 bge-reranker-v2-m3 的本地调用、长度和资源；向量仍来自云端 v3。再接入 RRF 和重排：两路各 30 → 融合前 30 → 重排前 5。
- [ ] 保持先选主证据、再补原文的顺序；复用 context.py，只替换候选选取方式。
- [ ] 用开发题比较向量基线与新配置，记录质量、耗时及失败题，更新锁文件和配置版本。

产物：bm25.py、fusion.py、reranker.py、混合检索配置及初步对照记录。<br>
验收：两路有效范围一致，重排与原文扩展顺序正确，改进有逐题证据；允许出现无提升或变慢的结果。

### 第 14 步 按需接入 AKShare 自动获取和报告

- [ ] 保存真实 AKShare 响应，建立字段映射与 PDF 对照，先验证一家公司的三张报表；扩展验收抽查至少 30 条事实。
- [ ] 实现公告候选、全文/摘要识别、修订版选择和下载；外部来源失败不影响已入库 PDF 的问答与计算。
- [ ] 补齐同公司两份年报并列展示及全部首版指标，不将披露年份当作内容期间。
- [ ] 报告模板复用 SubTaskPlan 和已校验结果，接通 Markdown 导出后再开放报告界面。

产物：akshare_adapter.py、acquisition.py、报告模板与导出、相关界面。<br>
验收：API 值不得冒挂 PDF 页码，冲突有记录；报告按章节列明缺口。这些功能可分别提交、分别验收，不组成一个必须一次交付的大任务。

### 第 15 步 完善题集和评测程序

- [ ] 逐步补齐 2 家开发公司的 20 道开发题，完成调参与错误定位后，冻结解析、词典、提示词、模型配置和切分。
- [ ] 再对留出公司的资料完成 30 道测试题及标注，保存样本和题集哈希；不得根据测试答案修改解析与检索规则。
- [ ] 为子问题完整性、问题改写、父级上下文和问询范围保留专项案例，分别计数；一题拆成三项仍只是一道整题。
- [ ] 将第 8 步的小题集回归扩为完整评测 runner、逐题日志和汇总重算，后端可运行后再接评测页面。

产物：evals/dataset_v1.jsonl、标注规则、evaluation/runner.py 和可查看结果。<br>
验收：开发与测试边界清楚，人工核验参与情况单列；标准答案不入检索库，汇总能够追溯到逐题记录。

### 第 16 步 运行扩展对照与最终交付检查

- [ ] 按第 10 节运行四组检索实验、改写与上下文对照；固定其他条件，分别报告答案质量和开销。
- [ ] 先保留 `deepseek-flash` 云端基线，再按需要增加云端或本地模型对照；记录可取得的版本、地域信息、用量及成本。本地 Ollama/MLX 和三系列比较不属于当前必做验收；只有实际运行后才报告对应成绩。
- [ ] 保存未运行、失败与负结果，补全配置、备份、依赖及模型准备说明。
- [ ] 在干净项目环境验证重启、来源追踪和失败恢复；断网时检查解析、已有资料/结果浏览及确定性计算，新的编码、向量查询和生成明确报错，不宣称完整离线问答。

产物：实际完成的逐题实验、运行说明、最终交付清单。<br>
验收：每项成绩有来源、分母与配置；未完成实验不阻止已验证最小应用使用，但不能冒称完整复现所有实验。

### 9.1 本轮方案与开发步骤对应

| 已确定的方案 | 先定义什么 | 实现与验证位置 |
|---|---|---|
| DeepSeek 生成与百炼 v3 编码 | 第 2 步模拟验证与待运行探针，第 3 步两家适配器、配置及用量契约 | 第 5—8 步完成编码、规划及生成代码，用户配置后真实验收；第 13 步本地重排 |
| 云端失败与索引保护 | 第 3 步批次状态和配置哈希 | 第 5 步恢复批次与原子发布，第 8、16 步错误与断网检查 |
| 结构优先、500/80 子块与父级原文 | 第 2 步固定 CPU tokenizer，第 3 步父子和证据契约 | 第 4 步切分，第 5 步补原文，第 13 步接入重排后的同一扩展链路 |
| 元数据辅助召回 | 第 3 步身份、期间和过滤字段 | 第 4 步入库，第 5 步单路过滤，第 13 步双路一致性 |
| 意图识别、改写和 1—3 个子问题 | 第 3 步 QueryPlan/SubTaskPlan/Result | 第 6 步一次规划，第 7—8 步文字逐项执行，第 11 步混合验证 |
| 引用、漏项与状态校验 | 第 3 步草稿和结果结构 | 第 7 步先实现校验函数，第 8 步再接图，第 10—11 步补真实数字 |
| 人工纠错页面后置 | 原值、版本、核验依据从第 3 步保存 | 第 9 步可受控导入核验事实；第 12 步之后再决定界面范围 |
| BM25/RRF/Cross-Encoder 与摘要实验 | 统一子块与配置版本 | 第 13 步增加混合检索；摘要向量仍为后续可选，不混入当前必做 |

### 9.2 编码准备情况与尚缺的验证

截至 2026 年 10 月 3 日，本次核对的是设计和本机可见环境，不是项目运行验收。

| 项目 | 已知状态 | 对开工的影响 |
|---|---|---|
| 项目代码 | 已有文档、evals 准备材料和临时 PDF 预览；尚无应用代码、pyproject.toml 或 uv.lock | 可以按第 2—3 步建立工程；不能声称应用已经可运行 |
| 真实文字样本 | 南方乳业招股书及两份越疆回复已登记路径、哈希与物理页数；已编写 10 道问询开发题并核对参考原文 | 可用于解析与检索开发；题集不是已执行的评测结果，南方乳业专项题仍可后续补充 |
| 工具与 GPU | 当前 PATH 可找到 Python、uv、Git 和 nvidia-smi；GPU 为 RTX 5070 Ti，16303 MiB；系统内存约 31.1 GiB | 仅说明工具和硬件可见，不等于项目依赖、CUDA 或模型已经验证 |
| 两家云端模型 | 已确定 DeepSeek-V4.1-Flash（`deepseek-flash`）和百炼 `text-embedding-v3`（1024 维），用户说明已有两家 Key | 先编写客户端和模拟用例，用户配置后验证实际 endpoint、权限及效果 |
| 本机接入配置 | 用户明确在代码开发完成后自行写入环境文件；本轮未读取密钥或调用付费接口 | 开发交付配置模板及待运行探针，不要求现在提供 Key；真实验收保持待配置 |
| 依赖与切分计数 | 尚无项目锁文件或实测；首版只需要 tokenizer 文件，不加载 BGE Embedding 权重 | 第 2 步固定最小依赖和切分尺；本地重排依赖第 13 步再加 |
| 财务计算样本 | 江丰电子 2026 年半年报共 193 页，已核对主要指标及三张合并报表的关键页，包含 2025 年同期数据 | 可启动第 9 步 PDF 财务开发；全年年报、留出公司与真实 API 响应仍按后续阶段补充 |
| 问询专项样本 | 已登记越疆 131 页回复和 48 页第二轮回复；第一份正文未确认轮次，不自动填首轮 | 可做真实问询检索开发；首轮身份对应和修订版本专项尚未验证。前置海外公告文字提取有乱码，日期已按页面视觉核对 |

当前文字与首份财务开发材料已到位，可以进入编码。依赖锁定和模拟检查在开发期间完成；用户写入 Key 后再进行真实联通、建库及小题集验收。真实运行暴露的问题记入失败清单后小步修正，文档审查不等于保证不存在运行时问题。

本次准备产物见 [开发题集与公式预览](D:/Desktop/A股年报分析系统/evals/开发题集与公式预览.md) 和 [使用说明](D:/Desktop/A股年报分析系统/evals/README.md)。evals 下保存样本清单、问询开发题、8 个公式参考及 jiangfeng_2026h1_reference.json。后者登记半年报原值、期间、页码和参考计算，不是应用已运行结果。用户口述的 2022 年与文件正文不同，按正文登记 2026H1；“更正后”仅为文件名标签，未取得原版及更正公告，不推断更正项目。

### 9.3 本轮开发顺序检查结论

16 步的总体顺序成立。第 2 步准备环境、模拟检查及独立探针，第 3 步定义正式契约和客户端，第 4 步解析，第 5 步实现向量入库；用户配置 Key 后再运行真实路径，避免后置密钥阻止后续代码开发。

实际开发仍按“工程环境与接口契约 → 存储 → PDF 父子切分 → 建库与检索 → 规划与带引用回答 → 财务核验与计算 → 扩展与评测”推进。核心源码及四页界面现已接通，江丰半年报5道计算题通过真实原文+模拟生成的本地全流程。用户配置后仍须重新编码真实资料、运行探针和云端问答验收；本地模拟流程不证明两家云端质量。

实施补充：Windows中文路径采用Hatch exact editable模式；后端及CLI对同一数据目录取得进程锁，避免多进程误恢复任务。资料身份和事实在分析开始时固定快照，建库期间拒绝改身份。解析缓存按解析器版本失效；开放边框财务表通过文字基线补列，不能确认的表保留待核验。备份仅在服务停止时执行，恢复校验SHA并重定位PDF路径。多子问题纯检索评测用固定子问题列表、逐排名交替合并去重后取TopK，避免第一项独占指标；实验文件记录这一汇总口径。冻结生成输入必须与当前问题和报告范围一致，C0只保存实际发送的短块。

## 10 50 题评测怎样做

### 10.1 题目分布与隔离

| 类型 | 总数 | 开发集 | 锁定测试集 |
|---|---:|---:|---:|
| 语义检索与文本回答 | 20 | 8 | 12 |
| 数据取数与计算 | 15 | 6 | 9 |
| 混合分析 | 10 | 4 | 6 |
| 缺失 冲突 歧义与边界 | 5 | 2 | 3 |
| 合计 | 50 | 20 | 30 |

这是首版工程评测集，样本规模不足以代表全部 A 股公司。额外的计算单元测试不计入 50 题。术语标签横跨语义与混合类，预先标注，不只挑系统答得好的题。

每题记录公司、报告版本、问题类别、标准答案要点、允许的证据集合、数值及展示精度、是否应澄清或拒答。检索相关性优先按段落或表格证据单元标注，然后映射到 chunk；重复或重叠 chunk 不重复计功。

第 1 步的招股书及问询小题集单独登记；年报开发题逐步补齐为这 20 道，不在最终测试中重复计数。第 15 步先冻结规则与配置，再处理留出公司的测试资料；报告可入库，题目、标准答案和评分标签不得入库。只有 30 题属于本设计的独立测试，不能写成“50 题全部独立测试”。

### 10.2 对照实验

| 编号 | 配置 | 用途 |
|---|---|---|
| R0 | 仅 BM25 | 观察精确术语检索 |
| R1 | 仅 text-embedding-v3 dense，1024 维 | 作为当前向量 RAG 检索基线 |
| R2 | BM25 + dense + RRF | 测量融合贡献 |
| R3 | R2 + Cross-Encoder | 测量重排贡献 |
| M1 | 固定 R3 输出证据，仅替换生成模型 | 比较表达和证据使用能力 |
| M2 | 同一题集与配置，允许不同模型参与路由及生成 | 比较完整系统效果 |

R0—R3 使用同一语料、同一切分、同一问题和同一标签。M1 不能与 M2 混在一个指标中。云端模型记录服务商、地域、实际 model_id/可取得版本、思考模式与测试日期；本地模型记录量化、模型规模和硬件。服务端硬件或参数量未公开时记为未知，不能把服务部署差异解释为纯模型架构差异。

R0—R3 的纯检索算法对照统一使用同一份冻结的规范检索问题，排除不同问法造成的干扰；第 6.1 节生产路径的“BM25 原问＋dense 改写”另记为生产配置。增加两个独立对照：Q0 两路及重排均使用原问、Q1 使用生产改写策略，用来评估问题改写；C0 只发送主命中子块、C1 在同一主证据集合上补充父级原文，用来评估上下文补充。每组固定其他参数，记录答案质量、发送 token 数和耗时。

复合题在纯检索与改写对照中固定同一份子问题清单，BM25 原问指当前项 query_text，改写指 rewritten_query，避免把拆题变化混成检索改进。端到端复合题另测 N4 的拆题与逐项执行，不以子问题数量增加整题样本数。

摘要向量对照单列为后续实验，不计入首版必需验收。父级上下文即使补全了答案，也不改变扩展前的 P@5 / Recall@30；回答质量与检索指标分别报告。新增招股书和问询专项验收案例单独列数量，不混称为原有 50 题的测试成绩。

可以另设“直接给模型提供年报”的参考实验，但必须记录实际可送入的文档长度、截断方式和解析能力；超出上下文时不能假装模型读过完整报告。

### 10.3 指标定义

| 指标 | 本项目评分方式 |
|---|---|
| Precision@5 | 前 5 个去重证据中相关证据数 / 5；不足 5 个按缺失位置不相关处理 |
| Recall@30 | 前 30 个候选覆盖的标准相关证据数 / 该题全部标准相关证据数 |
| 术语问题 Recall@30 | 在预先标注的术语题子集计算上述证据召回率，不等同于词语出现率 |
| 路由准确率 | 三类正常问题路由正确数 / 正常问题数；澄清题单列 |
| 复合题整题正确率 | 全部子项的答案或未完成状态均符合人工标注的整题数 / 复合题数；缺一项即不算整题正确 |
| 数值正确率 | 数值、单位、年份、口径均正确的数值目标数 / 全部数值目标数；再报告整题全对率 |
| 引用正确率 | 真实存在且支持对应结论的引用数 / 全部引用数；无引用时为不适用，不记满分 |
| 结论支持率 | 被引用证据支持的可核验事实结论数 / 全部可核验事实结论数 |
| 应拒答识别率 | 应拒答或澄清案例中处理正确的数量 / 此类案例数 |
| 错误拒答率 | 可回答案例中被错误拒答的数量 / 可回答案例数 |
| 延迟 | 记录端到端及各节点 P50、P95，区分冷启动和预热 |
| 资源与稳定性 | 峰值内存/显存、超时、结构化输出失败、重复运行不一致 |
| 云端用量与成本 | 百炼文档入库、问题编码分别记录 total_tokens；DeepSeek 记录输入/输出及缓存用量；汇总调用次数、重试与费用，区分本地资源与云端服务 |

P@5 的分母固定为 5，如果某题只有 1 个有效证据，其理论最高分可能只有 0.2。必须解释这一限制，可辅助报告 Hit@5；不要为了凑满 5 条而把相邻无关片段标成相关。纯计算且不走检索的题不进入检索指标分母。

无答案题不进入 P@5 和 Recall 的分母，单列拒答表现。检索指标先逐题计算，再对适用题做宏平均，报告适用题数。标准证据应由人工尽可能完整标注，并复核各对照组候选；若只标出少数参考片段，指标应改称“标准证据覆盖率@30”，不能声称完整相关集合的 Recall。

先统计开发集再冻结参数；测试集逐题保存结果。生成实验建议每题运行 3 次并记录温度、种子（后端支持时）及硬件；即使温度为 0，也不把完全确定性作为保证。

### 10.4 怎样报告截图中的提升数字

相对提升 = `(新指标 - 基线指标) / 基线指标 × 100%`。

例如 P@5 从 0.60 到 0.69，表示相对提升 15%，绝对增加 0.09，即 9 个百分点；不能写成提升 15 个百分点。这只是说明口径的例子，不是复刻实测结果。

最终报告应写：“在固定的 N 道测试检索题上，R1 为 X，R3 为 Y，相对变化 Z%，题集版本为 V。”基线为 0 时不计算相对提升。50 题中用于术语评测的子集更小，因此必须给出题数和逐题变化，不能把结果泛化到所有金融问答。

## 11 验收标准与失败处理

以下门槛是建议的工程验收要求，不是已实现的成绩。吞吐和耗时在目标硬件验证后再制定，不直接承诺“15 秒内回答”。

分阶段验收：第 8 步验证文字流程，第 12 步验证当前已启用指标的最小分析流程；用户配置 Key 前仅声明代码与本地验证结果，配置后补做真实云端验收才声明问答可用。F01—F12、完整指标及 50 题对照属于扩展交付范围；尚未接入的能力单列，人工纠错页面不属于前两个阶段的必需项。

| 检查项 | 建议通过条件 |
|---|---|
| 功能闭环 | 当前阶段声明的功能有实际运行证据；只有全部 F01—F12 通过才声明完整复刻 |
| 工作流 | 第 8 步验证文字及不可执行项路径；第 11—12 步验证三类路线与子问题循环；不为拆题增加业务节点 |
| 解析正确性 | 抽查页码 100% 对应；用于验收计算的关键字段全部核对 |
| 计算器 | 当前启用指标的缺失、零分母、负基数、口径冲突测试通过；未接通指标明确拒绝 |
| 端到端数值 | 锁定测试集数值目标正确率建议至少 95%，同时公布分子与分母 |
| 引用结构 | 引用 ID、范围、页码均有效；引用原文属于本次实际发送的证据且满足生效过滤条件；禁止无来源的关键数值 |
| 语义支持 | 人工复核结论支持率建议至少 90%；暴露不支持的具体案例 |
| 边界行为 | 冲突、缺失和不明确范围的验收案例均按预期拒答或澄清 |
| 检索实验 | 四组对照可复现，实际结果完整报告；不要求必然提升 15% 或 23% |
| 数据持久化 | 重启能继续查询；重复上传幂等；失败版本不会替换可用索引 |
| 来源隔离 | 跨公司、跨年度及提示注入样例不会突破所选数据范围 |
| 父子上下文 | 子块可定位同版本父块；同父块多次命中不会重复发送；扩展后引用仍定位实际原文页 |
| 输入长度 | 超长父块按命中位置补上下文；保留主证据及表头，记录省略范围，实际输入不超配置预算 |
| 云端编码与索引 | 文档/query 用途正确，1024 维与输入 ID 对齐；批次失败不发布新版本，旧索引保留，问题编码失败不冒充无证据 |
| 断网边界 | 可解析和浏览本地资料、运行已有核验事实计算；新编码、向量查询和生成显示相应云端服务错误 |
| 问询元数据 | “第二轮回复”不混入第一轮或问询函；指定版本不混入其他版本；缺失轮次时提示身份未确认 |
| 期间与日期 | 披露日期与内容期间分开；招股书包含多个年度时能检索其中明确的目标期间 |
| 意图与改写 | 口语问法可规范化；原问不被覆盖；公司、年份、否定和比较方向不漂移，关键歧义先澄清 |
| 子问题执行与漏项 | 单问、两问和三问均逐项有结果；局部缺资料不删除已回答项；全不可执行可正常结束，空计划不误报成功 |
| 子问题隔离与故障 | 年份、轮次、证据和操作数不串项；全局故障优先为 failed，已完成项保留，未执行项不冒称无证据 |
| 纠错迭代边界 | 最小闭环保留原值、核验依据和来源；用户修改页面后置，不以无记录手工修正支撑自动化成绩 |
| 摘要边界 | 首版摘要向量关闭；后续摘要命中须回到原文去重和核验，摘要本身不能作为事实引用 |

首版可使用的错误代码包括：`REPORT_AMBIGUOUS`、`PDF_NEEDS_OCR`、`PARSE_FAILED`、`SOURCE_UNAVAILABLE`、`FIELD_SCHEMA_CHANGED`、`UNIT_UNKNOWN`、`DATA_CONFLICT`、`METRIC_UNSUPPORTED`、`FEATURE_NOT_ENABLED`、`SUBTASK_LIMIT`、`RUN_BUDGET_EXCEEDED`、`INSUFFICIENT_EVIDENCE`、`MODEL_TIMEOUT`、`EMBEDDING_UNAVAILABLE`、`EMBEDDING_INVALID_RESPONSE`、`EMBEDDING_INPUT_TOO_LONG`。

日志记录 `run_id`、`subtask_id`、节点、配置版本、原问、改写及提示词版本、过滤条件及来源、主命中子块、实际扩展片段、输入 token 数、各项状态、耗时和错误码。失败页面给出可执行的下一步，例如重新选择报告版本、上传原 PDF、补充年度或等待数据源恢复，不向用户直接展示堆栈。

## 12 最终交付物

开发交付：可启动应用、已完成模块源码与锁定依赖、当前样本清单、解析和财务字段规则、子问题工作流、带 mock/live 标记的验证记录、DeepSeek 与百炼 v3 客户端、待运行联通探针、固定 tokenizer 版本、环境变量模板及启动备份说明；不交付真实密钥。用户自行填写后补做真实联通、建库及问答验收，形成实际可用的最小分析闭环。后续扩展再交付本地重排、50 题及标注、检索和模型对照、自动获取与报告样例；未完成部分单列。

开发状态需分别标记：**设计完成、代码完成、样本验证通过、测试集验证通过、尚未验证**。这份文档属于设计完成，不代表其余状态已达成。

按第 9 节设置三个代码里程碑：第 8 步文字问答、第 12 步含可信计算的最小分析流程、第 13—16 步扩展。每个里程碑分别记录本地验证与待配置的真实验收；用户完成环境配置后，集中补跑已实现范围的真实检查。下一轮只选一个具体问题迭代，人工纠错页面在第 12 步之后按实际错误单独安排。

## 13 官方参考资料

以下资料用于核对技术能力和接口边界。项目参数、模块拆分和验收门槛由本文设计，不应理解成官方推荐值。资料核对日期为 2026 年 10 月 3 日，开发时仍需验证实际版本和返回数据。

- [LangGraph 图与条件路由](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Pydantic 自定义校验器](https://docs.pydantic.dev/latest/concepts/validators/)
- [AKShare 股票与财务接口](https://akshare.akfamily.xyz/data/stock/stock.html)
- [AKShare 三张财务报表源码](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_three_report_em.py)
- [pdfplumber 文本和表格提取](https://github.com/jsvine/pdfplumber)
- [Chroma 官方仓库](https://github.com/chroma-core/chroma)
- [FastAPI 部署概念](https://fastapi.tiangolo.com/deployment/concepts/)
- [bge-m3 模型卡](https://huggingface.co/BAAI/bge-m3)
- [bge-m3 官方论文](https://arxiv.org/html/2402.03216v5)
- [DeepSeek 型号与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
- [DeepSeek 思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)
- [百炼 Embedding 接入示例](https://help.aliyun.com/zh/model-studio/dashscopeembedding-in-llamaindex)
- [百炼 text-embedding-v3 模型与价格](https://help.aliyun.com/zh/model-studio/text-embedding-v3)
- [百炼 Embedding 同步接口](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- [百炼地域与接入域名](https://help.aliyun.com/zh/model-studio/beijing-access-information)
- [百炼错误码](https://help.aliyun.com/zh/model-studio/error-code)
- [bge-reranker-v2-m3 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [RRF 算法说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [Ollama Windows](https://docs.ollama.com/windows)
- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)


## 14 本轮代码交付与验收口径（2026-10-03）

F01—F12 的代码位置、测试记录与实际边界统一见 [验收文档](验收文档.md)，运行方式见 [README](../README.md)。第 9 节的 16 步已按依赖关系执行，原定后段的本地重排、自动获取适配、报告、正式 50 题和对照 runner 均已纳入本轮代码交付。代码完成与真实云端问答质量通过分别记录，用户尚未提供 Key，不据 mock 宣称完整效果复现。

此次实现补充以下工程约束：

1. **保证重启后仍能查询。** Windows 中文目录的原生 HNSW 持久化问题用同盘英文目录别名处理；实际数据仍留原目录。独立进程、超过 1000 向量的重启测试作为回归，不能用同进程新建客户端替代。[上游同类问题记录](https://github.com/chroma-core/chroma/issues/7090)
2. **损坏索引可以恢复。** 准备时核对索引 ID 并执行真实向量查询；异常时从本地成功编码缓存构建恢复版本，全部通过后才切换活动版本。保留旧版本与原文，关闭服务时释放 Chroma 句柄。
3. **对照确实使用同一份输入。** 冻结文件保存索引与事实快照；Q0/Q1 若遇到资料或事实更新，拒绝执行并要求重新冻结。M1/C0/C1 与正式工作流使用相同的结果筛选，不能把被拒绝的计算算成正确回答。
4. **评分不改变排名。** TopK 先按真实候选位置截断；重复标准证据可不重复奖励，但不能靠标准答案删除前面的候选，让 K 外结果提前。表格命中核验原 PDF 页及行列顺序。
5. **对未实测内容保留空值。** 模拟问答不汇总成真实质量；未独立复核的语义支持率为空；未配置日期化单价时费用为空。当前机器仅验 CPU 重排，GPU 与云端性能留待实际配置验收。

正式题集由 20 道开发题和 30 道福耀留出题组成。财务参考核验轨道与纯自动抽取成绩分开；本地 50 题执行和开发 2 题各组对照 smoke 也分开。规则冻结后发生的存储、异常、用量记录等工程修复另行登记，不宣称所有文件 SHA 严格不变。没有依据留出答案调节切片、提示词、公式或检索参数。

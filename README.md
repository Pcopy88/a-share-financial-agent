# A 股财报资料分析 Agent

上传年报、半年报、招股说明书或问询回复，在选定资料中检索原文、计算财务指标并查看来源页。当前生成模型为 DeepSeek `deepseek-flash`，向量模型为百炼 `text-embedding-v3`（1024 维）。本地保存 PDF、解析结果、SQLite 和 Chroma。

**应用代码与本地验收已完成。** 默认 `mock` 模式不用 API Key，可以检查界面、解析、计算与工作流；它不能证明云端模型的真实问答质量。真实账户由用户完成开发后填写。实际验收记录见 [验收文档](docs/验收文档.md)，当前实现与待验项见 [开发执行记录](docs/IMPLEMENTATION_STATUS.md)。

## 1. 安装与启动（Windows PowerShell）

在本项目目录执行。需要 uv 和联网下载依赖；Git 为可选工具，本项目使用 Python 3.11。先复制留空密钥的 `.env.example` 即可运行模拟模式，两家 Key 在开发完成后再填写。

```powershell
uv sync --locked --all-extras
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
.venv/Scripts/python.exe -m annual_report_agent doctor
.venv/Scripts/python.exe -m annual_report_agent models prepare-tokenizer
./start.ps1
```

界面：[http://127.0.0.1:8501](http://127.0.0.1:8501)。接口说明：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。启动命令保持运行，按 `Ctrl+C` 结束后端与界面。若 PowerShell 不允许执行脚本，可直接运行：

```powershell
.venv/Scripts/python.exe -m annual_report_agent start
```

也可分开启动，在两个终端分别执行 `.venv/Scripts/python.exe -m annual_report_agent api` 和 `.venv/Scripts/python.exe -m annual_report_agent ui`。后端只使用一个 worker。同一个 `data/mock` 或 `data/live` 目录只允许一个后端或批处理实例；看到 `INSTANCE_ALREADY_RUNNING` 时先停止原实例。不要通过删除锁文件绕过它。

`.env.example` 的默认端口与上述地址一致。若修改 `ARA_API_PORT`，同时把 `ARA_API_URL` 改成 UI 可访问的完整后端地址，例如 `http://127.0.0.1:8001`；修改 `ARA_UI_PORT` 后用新端口打开界面。修改配置后重启应用，CLI 命令仍从项目目录执行。

基础功能可仅安装 `uv sync --locked`；`--all-extras` 额外安装 AKShare、PyTorch 和 FlagEmbedding。安装完后使用上面的 `.venv` 命令，避免再次执行未包含 extras 的 `uv run` 改变当前可选依赖。

## 2. 第一次使用

1. **资料管理**：上传 PDF，核对公司、文件类型、年度、内容期间、问询轮次与文件版本，再点“解析并建立知识库”。
2. **分析问答**：选择资料，输入问题。可以一句话问 2—3 个明确小问题；条件不清楚时系统应澄清。
3. **证据与指标**：检查 PDF 物理页、提取原值、单位、期间与核验状态。只有 verified 的事实能进入计算器。
4. **评测与报告**：生成固定模板的 Markdown 报告，或运行已登记的开发/测试题集。结果逐题保存在本地。

提交后使用“刷新任务状态”查看完成结果；资料准备期间先等待完成，再修改公司或报告期。新上传文件提取的财务值默认可能为 `pending`，不能直接计算。当前明确登记的开发参考文件仅在源 SHA 匹配后参与核验；“查询 AKShare 候选值并核对”只支持已确认股票代码、年度且已建库的沪深全年年报，半年报和问询资料不走此入口。

扫描图片页暂不自动 OCR；页面会标记 `needs_ocr`。表格结构不确定时保留原文并标记，不能因此声称财务数值已自动提取正确。文字搜索和可信财务计算是独立能力。

已提供的江丰文件正文是 **2026 年上半年**，不是 2022 年。数字计算用本报告的 2025 年同期比较列；未经审计。“更正后”来自文件名，不能据此推断更正内容。

## 3. 导入本次真实样本

先停止已启动的应用，再执行；样本清单使用本机路径，换电脑需要更新 path，SHA 必须保持匹配。

```powershell
.venv/Scripts/python.exe -m annual_report_agent samples --manifest evals/sample_manifest.json --prepare
.venv/Scripts/python.exe -m annual_report_agent samples --manifest evals/annual_sample_manifest.json --split development --prepare
```

第一份清单包括南方招股书、越疆两份回复、江丰半年报。第二份清单包含 3 家公司各两年公开全年报；开发资料为茅台、格力，留出资料为福耀。下载脚本与官方 URL、SHA 保存在 `scripts/prepare_annual_eval.py` 和年报清单中。题目、答案和财务核验标签不会进入向量库。

`--prepare` 在 `live` 模式会发送切片至百炼并计费；默认 `mock` 使用独立模拟向量。大 PDF 需要等待，命令定期显示正在处理，失败可重试并复用已完成编码批次。

可先试这几题：

- 选择江丰半年报：“2026 年上半年营业收入同比增长多少？”
- 选择江丰半年报：“2026 年上半年的毛利率是多少？”
- 选择越疆相应回复：“公司的产品矩阵覆盖哪些负载范围和轴数？”

## 4. 开发完成后，填写真实云端配置

在本机 `.env` 填写以下内容，密钥不要发到聊天、不要提交 Git：

```dotenv
ARA_MODE=live
DEEPSEEK_API_KEY=填入你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DASHSCOPE_API_KEY=填入你的百炼密钥
DASHSCOPE_EMBEDDING_URL=填入百炼控制台实际完整Embedding地址
ARA_RETRIEVAL_MODE=hybrid
```

两家的 Key 不互换。百炼地址必须与账户地域和业务空间一致，不能保留 `{WorkspaceId}` 等占位符。若 Key 来自第三方平台，先核对服务商的地址与型号；当前客户端按官方接口实现。

停止应用后运行联通探针，再以 `live` 重新导入建库：

```powershell
.venv/Scripts/python.exe scripts/smoke_models.py --live
.venv/Scripts/python.exe -m annual_report_agent samples --manifest evals/sample_manifest.json --prepare
.venv/Scripts/python.exe -m annual_report_agent start
```

`data/mock` 和 `data/live` 完全分开，真实检索不能复用模拟向量。缺配置时 live 明确失败，不自动切回模拟回答。N4 的问题与范围、N8 的选中原文与计算结果会发送给 DeepSeek；切片与检索问题会发送给百炼。

## 5. 本地重排

```powershell
.venv/Scripts/python.exe -m annual_report_agent models prepare-reranker
```

随后在 `.env` 设置 `ARA_RERANK_ENABLED=true`、`ARA_RETRIEVAL_MODE=hybrid` 并重启。模型是固定 revision 的 `BAAI/bge-reranker-v2-m3`，小批量 2、最大输入 1024 tokens；超长问题明确报错，不截掉年份或表头。本地模型只负责候选相关性排序，不负责生成答案。

本机已验证 CPU FP32 加载与推理。当前锁定依赖默认安装的是 CPU PyTorch，RTX 5070 Ti 的 CUDA 推理尚未验收；CPU 模式可以运行完整链路。需要 GPU 时，应依据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选取与本机驱动兼容的 CUDA wheel，再运行同一探针，确认输出设备实际为 `cuda:0`。不要把已检测到显卡当作 GPU 已成功推理。

`ARA_RERANK_DEVICE=auto` 自动选择可用设备；也可设 `cpu`。首次下载权重会占用磁盘并耗时，后续默认从本地缓存加载。

## 6. 测试与评测

执行下面的 `evaluate` 前，先按第 3 节在**当前模式**导入对应资料并建库，再停止 API/界面；评测 CLI 也会独占该模式的资料目录。通过界面提交评测时保留应用运行，由后端同一任务队列执行。修改 `.env` 为 `live` 后，评测会按实验配置调用真实云端并计费。

```powershell
./scripts/test.ps1 -q
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/python.exe scripts/smoke_models.py
.venv/Scripts/python.exe -m annual_report_agent evaluate --dataset evals/development_inquiry_v1.jsonl --experiment R0
.venv/Scripts/python.exe -m annual_report_agent evaluate --dataset evals/annual_development_v1.jsonl --experiment M2
```

测试脚本使用项目 `tmp` 中的独立目录，绕开本机系统临时目录权限问题。不使用脚本时可加 `--basetemp=tmp/pytest-本次唯一名字`。

已导入江丰样本时，可在停止应用后运行 `.venv/Scripts/python.exe scripts/validate_local.py` 核对五道计算题；结果写到 `outputs/当前模式/acceptance`。该脚本默认拒绝 live 模式，真实运行需另加 `--live`。没有用户本机 PDF 的机器会跳过部分真实资料测试，不能把跳过计作通过。

开发验收还提供 `.venv/Scripts/python.exe scripts/validate_experiments.py`，**先按第 3 节完成四份开发全年报建库、按第 5 节准备重排缓存，然后停止应用，仅在 mock 模式执行**。脚本读取六份年报清单并准备两份留出资料，显式启用对应参考核验，走正式 50 题（20 开发 / 30 留出）的 M2 流程，并用两道开发题各运行一次 R0—R3、M1/M2、Q0/Q1、C0/C1 功能检查；结果保存到 `outputs/mock/acceptance/experiment_workflow.json`。这是借助参考核验财务事实的流程检查，默认应用仍只登记五份开发参考；留出参考文件不作为检索内容或生成提示。两题单次检查不是完整实验对照，50题的模拟结果也不是真实质量成绩。

实验含义：R0 关键词；R1 向量；R2 双路融合；R3 再加重排；M2 整个问答流程；M1 复用冻结证据只比较生成；Q0/Q1 固定子问题比较原问与改写；C0/C1 固定命中比较是否补充上下文。M1/Q/C 实验需传入先前 M2 保存的 `--frozen outputs/当前模式/evaluations/评测编号/frozen.json`。

R3 需要先准备权重并启用重排。生成类实验可以用 `--repetitions 3`，模型档案由本地 `model_profiles.yaml` 管理；当前默认只有 DeepSeek Flash，不自动调用其他供应商。

逐题日志与配置保存在 `outputs/mock/evaluations` 或 `outputs/live/evaluations`。模拟结果不进入正式质量均值；缺少人工语义评分时“结论支持率”等保持未评，不自动给满分。现有 gold 是必要片段子集，使用证据覆盖率，不把它包装成完整 Recall。8 项指标公式用 Decimal 执行，单位、时间或口径冲突时拒绝计算。

## 7. 备份和恢复

先停止应用与批处理，再执行：

```powershell
.venv/Scripts/python.exe -m annual_report_agent backup --to backups/mock-20261003.zip
.venv/Scripts/python.exe -m annual_report_agent restore --archive backups/mock-20261003.zip --to restored_data/mock
```

备份包含当前模式的原 PDF、SQLite、Chroma、检查点与任务；不包含 `.env`、tokenizer/重排模型缓存及输出报告。每次备份使用新文件名；恢复先校验每个文件 SHA，只写入新的空目录，并重定位报告路径。在 `.env` 设置 `ARA_DATA_DIR=restored_data`、`ARA_MODE=mock` 后，启动时会使用恢复的 `restored_data/mock`；live 同理，模式必须与归档一致。

准备新资料或重新建库前，再运行第 1 节的 `models prepare-tokenizer`，它会把计数文件放入新的 `ARA_DATA_DIR/tokenizer`。若启用本地重排，还要按第 5 节重新准备重排缓存。已有资料库不会替你恢复模型文件；真实密钥仍由本机环境文件单独保存。

## 8. 常见问题与边界

| 现象 | 处理 |
|---|---|
| 页面无法连接后端 | 查看 `outputs/当前模式/server_logs/api.log`，确认端口 8000/8501 未占用 |
| `CONFIGURATION_REQUIRED` | 补齐提示的环境变量名并重启，不需要把 Key 发给开发者 |
| `INDEX_PROFILE_MISMATCH` / `INDEX_STALE` | 在当前模式、最新身份配置下重新建库 |
| `METADATA_UNCONFIRMED` | 在资料管理中确认身份信息 |
| `REPORT_BUSY` | 等待该资料的准备任务结束，再修改公司、期间或文档类型 |
| `TOKENIZER_UNAVAILABLE` | 从项目目录运行 models prepare-tokenizer，并确认使用同一个 ARA_DATA_DIR |
| `needs_ocr` / 表格结构不确定 | 换可复制文字版本，或保留标记等待人工核验，不将失败页视为没有披露 |
| 数字查询资料不足 | 查核验事实状态；未核验、分母为零、期间不一致或口径冲突均不能硬算 |
| 公告下载失败 | 改用本地 PDF 上传；动态详情页、代理 Fake-IP、超时和非允许源有明确错误 |
| 重排权重未就绪 | 先运行 prepare-reranker；默认不在每次问答中自动联网下载 |

项目面向本机单用户，不作为公开互联网服务部署。无登录、多租户和后台权限体系；默认仅监听 `127.0.0.1`。原文引用的程序校验能验证来源存在、数值与公式一致，但不能自动证明任意文字解释完全正确，仍需独立评测。

核心目录：`ingestion` 解析建库，`retrieval` 双路与重排，`finance` 数值规则，`graph` 九节点，`generation` 云端与核验，`api` 后端，`ui` 界面，`evaluation` 评测，`evals` 题集与参考。正式设计见 `docs/A股年报智能分析Agent开发文档.md`。


### Windows 中文路径说明

当前锁定的 Chroma 1.5.9 在本机中文目录中曾出现超过 1000 个向量后，HNSW 文件未落盘、重启无法查询的问题。程序会自动在同盘最近的英文祖先目录下创建 `.annual-report-agent/chroma-paths/` 目录别名；实际数据仍保存在 `ARA_DATA_DIR` 内，备份范围不变。别名已用 1100 个向量、两个独立进程验证。若机器不允许创建目录别名，请将 `ARA_DATA_DIR` 配置为可写的纯英文绝对路径。

重新点击“解析建库”会检查现有索引；损坏时从已缓存向量构建新版本，成功后才切换，不删除原版本。停止应用时会关闭向量库句柄。恢复备份后目录别名会按新路径重建。

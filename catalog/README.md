# 数据浏览器：VLN Atlas / BetaData

本目录已加入面向 **R2R、RxR** 的 VLN Atlas 模式，支持多语言导航指令、参考路径俯视图、
服务端分页筛选、470×280 / 8 FPS LeRobot GT 视频、全量视频批处理和 JSONL 清单导出。

**VLN 的启动方式、数据路径配置和实现说明见 [VLN_README.md](VLN_README.md)。**
入口为 `build_vln_catalog.py` 与 `serve.py --vln-db`，无需 Habitat 或 GPU。
下文保留原 BetaData 机器人数据目录的说明，其原服务器路径不适用于 VLN 模式。

## BetaData 数据集语义浏览器

BetaData 是一个面向机器人、仿真、UMI、Ego 和 VQA 数据的目录管理网站。它从服务器上的真实文件/文件夹名称生成语义目录，关联 LeRobot 转换状态与视频，并提供筛选、人工复核、清单导出和任务词条散点视图。

本目录包含网站运行所需的前端、目录扫描、转换状态识别、视频索引与播放服务、每小时同步服务以及回归测试。程序只读取原始数据；不会移动、重命名、解压或删除数据集。

## 目录结构

```text
dataset_semantic_explorer/
├── README.md                    # 使用与部署入口
├── README.rst                   # 详细数据规则和实现说明
├── catalog.py                   # 原始目录扫描、语义分类与场景归一化
├── conversion_status.py         # LeRobot 转换证据和状态识别
├── embedding.py                 # 文本向量与二维 PCA（只手动更新）
├── build_catalog.py             # 完整生成 catalog.json
├── instruction_inventory.py     # 安全读取异构 metadata 中的 instruction
├── build_instruction_inventory.py # 生成完整 instruction JSONL
├── instruction_classification.py # 固定词表的 LLM 文本分类
├── classify_instructions.py     # 可续跑的分类命令行入口
├── video_index.py               # 视频文件发现、映射与安全校验
├── build_video_index.py         # 生成 video-index.json
├── video_serve.py               # 视频列表、Range 播放与按需转码 API
├── serve.py                     # 网站静态服务及同源视频代理
├── sync_catalog.py              # 安全的一次性/周期目录同步器
├── catalog_sync_service.sh      # 当前服务器的小时调度服务
├── web/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── video-config.json
└── tests/
    ├── test_catalog_and_services.py
    ├── test_sync_catalog.py
    ├── test_video_backend.py
    └── test_ui.js
```

## 当前数据层级

```text
数据域 → 场景 → 数据集 → 任务类型 → 原子动作 → 具体任务
```

场景和任务类型是语义分类；`native_scene`、`native_task` 和 `source_path` 始终保留原始值。已知英文原生场景只在语义字段中转换为中文，例如：

- `bedroom` → `卧室`
- `living_room` → `客厅`
- `kitchen` → `厨房与餐饮`
- `study` → `书房`

## 服务器路径

| 用途 | 路径 |
|---|---|
| 原始数据根目录 | `/mnt/cfs/data/datasets` |
| PFS 转换目录 | `/mnt/pfs/data/datasets/lerobotv3/high_priority_80d` |
| CFS 转换目录 | `/mnt/cfs/data/datasets/lerobotv3/high_priority_80d` |
| 网站目录索引 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json` |
| 视频索引 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-index.json` |
| 视频密钥 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-api.secret` |
| 小时同步状态 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog_sync_status.json` |
| 小时同步日志 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog_sync.log` |
| Instruction 清单 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.jsonl` |
| Instruction 汇总 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.summary.json` |
| 文本分类清单 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.jsonl` |
| 文本分类汇总 | `/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.summary.json` |

## 生成目录

从仓库根目录执行：

```bash
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python \
  scripts/dataset_semantic_explorer/build_catalog.py \
  --root /mnt/cfs/data/datasets \
  --output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json \
  --instruction-inventory /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.jsonl \
  --conversion-pfs-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
  --conversion-cfs-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d \
  --embedding-cache-dir /mnt/pfs/wangqiongyan/.cache/huggingface \
  --embedding-local-files-only \
  --embedding-device cuda
```

该命令会同时更新语义散点。日常目录同步不要运行它；使用下方的小时同步服务即可。

HiFi-UMI-2K 的 `chunk-* / part-*` 仅作为存储分块，不进入语义层级。扫描器读取各分块
`meta/episodes/**/*.parquet` 中的 `tasks`，按 instruction 聚合成具体任务，并把出现次数记录为
`episode_count`。转换状态通过 `HiFi-UMI-2K-80D-full` 各分块的 `info.json`、
`canonical_80d.json` 和 `tasks.parquet` 核验，并将输出分块名称与 episode 数逐项对照原始 metadata；
全部一致时显示为“已转换”，否则保守显示为“部分转换”。

EgoVerse、Something-Something-V2、Hy-Embodied-0.5-VLA-Data，以及只在正式 LeRobot 输出中提供
任务词条的 Galaxea-Open-World、LET-claw、RT-1、RoboCOIN，会从完整 Instruction 清单按数据集和
metadata 来源精确筛选后聚合进目录。转换副本（例如 `RoboMIND-2.0-full`）不会作为新数据集重复展示。
Something-Something-V2 按动作模板聚合，其余按 instruction 聚合。小时同步会消费已有清单，但不会每小时
重扫数十万个 Zarr/Parquet 文件；原始 metadata 新增后，应先重新生成 Instruction 清单。

## 生成 Instruction 清单

扫描目录任务、Zarr 根属性、Something-Something-V2 标签、LeRobot Parquet 和转换后的
`episodes.jsonl`。输出是一条记录一行的 UTF-8 JSONL；有任务到 episode 映射的数据按 episode
保留，没有映射的任务词表按 task definition 保留。扫描器只读 metadata，不读取媒体、不解压归档，
也不会反序列化不可信的 pickle。

EgoVerse 优先使用下载器生成的 `.egoverse-download/active_episodes.tsv` 枚举已下载 episode，避免在
CFS 上遍历数十万个一级目录；缺少该清单时才回退到一级目录扫描。

```bash
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python \
  scripts/dataset_semantic_explorer/build_instruction_inventory.py \
  --root /mnt/cfs/data/datasets \
  --catalog /mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json \
  --conversion-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
  --conversion-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d \
  --metadata-workers 16 \
  --output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.jsonl \
  --summary-output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.summary.json
```

每行包含稳定 `record_id`、`instruction_hash`、数据域、数据集、episode/task ID、原始 instruction、
归一化文本、字段名和真实 metadata 路径。后续分类只需按 `instruction_hash` 去重调用模型，再把结果映射
回所有 occurrence。

汇总文件会分别记录成功读取的 metadata 文件数、缺少本地 metadata 的 manifest 项、缺少 instruction
的记录和解析错误样本，便于区分“没有标注”和“数据尚未下载完整”。

转换目录只发现 `数据集/meta/episodes.jsonl` 和 `数据集/子集/meta/episodes.jsonl` 两种布局，
不会进入任务、episode、data 或 videos 深层目录；隐藏目录以及名称包含 `smoke`、`preflight`、
`integration-test` 的测试产物不会进入正式清单。

## Instruction 文本分类

分类器只处理唯一 `instruction_hash`，结果作为 sidecar JSONL 保存，不覆盖原始 instruction 或 metadata。
场景固定为 10 个粗粒度类别：厨房与餐饮、家居与日常生活、桌面与办公、清洁与维护、工业与工具、
织物与穿戴、户外与公共环境、通用操作场景、仿真通用场景、视觉问答与空间理解。模型不得创造新场景。

每条结果还包含一个用于树目录分组的 `primary_atomic_action`，以及最多 6 个有序 `atomic_actions`。
原子动作使用固定 27 类，包括抓取、放置、移动、持握、开/关、推/拉、按压、旋转、插入、拆卸、
倾倒、涂抹、清洁、折叠、扣合、切割、烹饪、排列、形变、抛掷、书写和观察等。接入网站后的层级为
`数据域 → 场景 → 数据集 → 任务类型 → 主原子动作 → 具体任务`；完整动作序列只在详情中展示，
避免一个多步骤任务在树上重复出现。

`native_group` 继续保留在记录、详情、转换映射和视频索引中，仅从目录层级与层级开关中移除。

Something-Something-V2 的大量实例文本先按原生动作模板归并，只分类约 205 个模板，再映射回每条唯一文本。
VQA 和仿真数据域使用固定场景规则。检查点逐批追加，进程中断后使用相同模型和词表可继续运行。

```bash
read -rs SKYWALK_API_KEY
export SKYWALK_API_KEY
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python \
  scripts/dataset_semantic_explorer/classify_instructions.py \
  --inventory /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.jsonl \
  --output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.jsonl \
  --summary-output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.summary.json \
  --checkpoint /mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.checkpoint.jsonl \
  --model gpt-5.4-mini \
  --batch-size 32 \
  --workers 4
unset SKYWALK_API_KEY
```

输出包含场景、任务类型、主原子动作、有序原子动作、置信度、简短依据、分类来源、模型、出现次数及
来源数据集。置信度低于 `0.72`、归入“其他 / 复合任务”或主原子动作为“其他”的记录会标记为
`needs_review`，方便后续人工抽检。

## 每小时自动同步

服务器没有可用的 cron/systemd，因此使用 `setsid + nohup` 启动单实例常驻调度器。它每 3,600 秒执行一次浅层扫描：

- 不生成或修改 embedding；
- 只在目录或转换状态发生实质变化时发布；
- 使用稳定记录 ID，浏览器中的人工复核不会因普通更新丢失；
- 原子替换 `catalog.json`，失败时网站继续使用上一版；
- 挂载缺失、记录自动删除、完成状态回退或未知状态增加时拒绝发布；
- 新页面或刷新后的页面会读取最新目录。

```bash
# 启动（重复执行不会启动第二个进程）
bash scripts/dataset_semantic_explorer/catalog_sync_service.sh start

# 查看进程、最近扫描结果和下一次运行时间
bash scripts/dataset_semantic_explorer/catalog_sync_service.sh status

# 立即执行一次安全扫描
bash scripts/dataset_semantic_explorer/catalog_sync_service.sh run-once

# 停止或重启
bash scripts/dataset_semantic_explorer/catalog_sync_service.sh stop
bash scripts/dataset_semantic_explorer/catalog_sync_service.sh restart
```

常驻进程可以跨 SSH 退出继续运行，但服务器/容器重启后需要重新执行 `start`。

## 生成视频索引

```bash
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python \
  scripts/dataset_semantic_explorer/build_video_index.py \
  --catalog /mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json \
  --output /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-index.json \
  --conversion-pfs-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
  --conversion-cfs-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d
```

视频索引独立于目录小时同步。新增转换视频后需要重新生成视频索引，避免每小时扫描大量媒体文件。
10Kh-RealOmin 使用正式 `valid_sources.jsonl` 中的全局编号和精确 MCAP 来源路径，只选取目录中已有任务
对应的 LeRobot 单回合分片；不会遍历全部 66 万个分片，也不会仅按同名任务猜测映射。

索引器也识别 Galaxea-Open-World、LET-claw、RT-1、RoboCOIN、Something-Something-V2 等正式
转换目录。HiFi-UMI-2K 使用 LeRobot v3 的共享视频布局：网站根据每个回合在 Parquet 元数据中的
起止时间定位到同一个大 MP4 的对应片段，不会把整段视频从头开始当作单个回合播放。
普通数据集默认每个任务保留 200 个回合用于视频预览；HiFi 与 Hy 的索引项只是共享 MP4 的轻量时间段定位，
因此单独保留最多 10,000 个回合，可完整覆盖当前任务，避免把安全采样上限误显示成映射缺失。
视频服务接受的索引文件上限为 512 MiB，可容纳当前约 420 MiB 的完整 HiFi/Hy 索引，同时继续拒绝异常超大文件。
Hy-Embodied-0.5-VLA-Data 使用 `Hy-Embodied-0.5-VLA-Data-80D-starvla-v3` 正式产物，
包含头部、左腕和右腕三视角 MP4，并核验转换及官方/StarVLA 读取器验收报告。

## 启动服务

视频 API 仅监听服务器本机端口 `8790`：

```bash
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python -u \
  scripts/dataset_semantic_explorer/video_serve.py \
  --index /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-index.json \
  --secret-file /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-api.secret \
  --media-root /mnt/pfs/data/datasets/lerobotv3/high_priority_80d \
  --media-root /mnt/cfs/data/datasets/lerobotv3/high_priority_80d \
  --media-root /mnt/pfs/data/datasets \
  --media-root /mnt/cfs/data/datasets \
  --transcode-cache /mnt/pfs/wangqiongyan/dataset_semantic_explorer/video-transcode-cache \
  --token-ttl 86400 \
  --host 127.0.0.1 \
  --port 8790
```

网站服务监听 `8788`，并把 `/video-api/*` 同源代理到 `8790`：

```bash
/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python -u \
  scripts/dataset_semantic_explorer/serve.py \
  --catalog /mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json \
  --video-api-upstream http://127.0.0.1:8790 \
  --host 127.0.0.1 \
  --port 8788
```

`catalog.json` 和 `catalog.js` 会在浏览器声明支持时使用 gzip 传输。当前完整任务目录在磁盘上约
43 MB，公网首次加载的压缩传输量约 1.4 MB；目录内容仍保持完整，不会截断具体任务。

不要把原始视频服务直接绑定到公网地址。公网访问应通过可信 HTTPS 代理进入 `8788`。

## 状态含义

下载状态和转换状态是两个独立维度。目录存在只表示源路径可见，不等同于完整下载。

转换状态包括：

- `converted`：结构化报告证明正式产物完成；
- `partial`：存在有效产物，但不能和所有原始任务一一对应；
- `not_converted`：正式输出根存在，但没有该任务的有效产物；
- `excluded`：正式报告明确排除该来源；
- `unknown`：挂载或结构化证据不足。

不同数据集采用与其转换流程匹配的完成证据：Galaxea-Open-World 核验源任务包与 227 个输出根逐项对应；
Hy-Embodied-0.5-VLA-Data 核验完整转换报告、22 个表及 episode 总数；LET-claw 核验完成清单、审计、
逐 episode 状态和明确排除项；RT-1、Something-Something-V2 核验逐 episode 状态，后者还会对照官方
标签清单；RoboMIND 核验正式清洗报告并把原始分卷聚合到清洗后任务。只有全部证据一致才显示“已转换”。
RoboCOIN 当前源目录与有效 LeRobot 输出根仍不完整，因此保持“部分转换”，避免把已有产物误判为全量完成。

人工筛选与人工复核保存在当前浏览器本地，并随 Manifest 导出；它们不会修改服务器数据。

## 测试

```bash
PYTHONPYCACHEPREFIX=/tmp/betadata_pycache \
  python -m unittest \
  scripts.dataset_semantic_explorer.tests.test_catalog_and_services \
  scripts.dataset_semantic_explorer.tests.test_sync_catalog \
  scripts.dataset_semantic_explorer.tests.test_video_backend \
  scripts.dataset_semantic_explorer.tests.test_instruction_inventory \
  scripts.dataset_semantic_explorer.tests.test_instruction_classification

node --test scripts/dataset_semantic_explorer/tests/test_ui.js
```

Python 测试覆盖目录扫描、转换证据、视频 Range 代理和安全同步；Node 测试覆盖前端层级、筛选、治理状态、视频回合/视角和转发子路径。

## 已知限制

- Egocentric-100K 的归档 clip JSON 目前未发现 instruction；Touch-in-the-Wild 场景包仅发现场景和 episode 边界；
- VQA canonical 索引状态还没有完全映射到网站转换状态；
- 语义散点不会随小时扫描自动更新；
- 不同电脑之间暂不共享浏览器本地的人工复核状态，请使用导出的 Manifest 交接。

更多分类、视频安全和数据治理设计见 [README.rst](README.rst)。

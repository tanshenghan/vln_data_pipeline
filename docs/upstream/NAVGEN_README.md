# 生成式导航数据管线（豆包 / Seedance 2.0）

这是一个可离线运行、也可切换到火山方舟真实 API 的导航训练数据生成管线。它实现了：

1. 结构化场景、任务和路线合同生成；
2. Seedance Prompt Compiler（镜头、场景锚点、时间线、终点条件和负面约束）；
3. 可选的 Doubao Prompt Refiner（只改写表达，不改变路线事实）；
4. Seedance 2.0 异步视频任务适配器；
5. 视频预筛、Embedding 一致性检查和 VLM Filter；
6. 基于外部位姿、视频运动估计或路线合同的轨迹生成（来源显式记录）；
7. 速度、加速度、抖动、碰撞和终点约束过滤；
8. 样本打包、质量分级和审计报告；
9. 可重复的离线小批量冒烟测试和接口单元测试。
10. 每阶段 token、延迟、请求/实际视频时长和成本状态统计。

默认不依赖第三方 Python 包（仅在生成离线 MP4 时需要系统中的 `ffmpeg`，图片处理使用 Pillow）。真实豆包调用通过环境变量启用：

```bash
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_API_KEY="<your-key>"
```

不要把真实 API Key 写入代码、配置文件、日志或提交记录。

## 快速开始

```bash
cd /home/beta/hongyu.yao
python3 -m navgen_pipeline.cli smoke --count 3 --out runs/offline_smoke
python3 -m unittest discover -s tests -v
```

只想检查“场景/任务 → Seedance 提示词”阶段时，可先导出计划和逐样本
`seedance_prompt.txt`、`negative_prompt.txt`、`prompt_report.json`：

```bash
python3 -m navgen_pipeline.cli plan --count 3 --duration 15 --out runs/prompt_preview
```

离线冒烟测试会生成 3 个小视频、轨迹、过滤报告和 `run_summary.json`。如果已经配置 `ARK_API_KEY`，可以运行固定生产管线：

```bash
python3 -m navgen_pipeline.cli run-fixed --count 1 --out runs/live_smoke
```

固定 live 管线会先由 `Doubao-Seed-2.1-pro` 规划结构化场景，再由本地
`PromptCompiler` 编译 Seedance 专用提示词。需要额外的模型润色时开启：

```bash
python3 -m navgen_pipeline.cli run --mode live --remote-prompt --count 1 --out runs/live_prompt_smoke
```

`--remote-prompt` 是可选的第二次 Responses 请求；即使模型超时或返回格式不合规，
也会回退到可审计的确定性编译器，不会把一句短指令直接交给视频模型。

每次运行还会写出 `usage_summary.json`，每个样本写出
`samples/<sample_id>/usage.json`。其中：

- `input_tokens/output_tokens/total_tokens` 是 Responses 文本请求（规划、提示词润色、VLM）的 token；
- `video_provider_*_tokens` 原样记录 Seedance 任务返回的 provider counter，和文本 token 分开，不能直接当作 LLM token；
- `video_requested_duration_s`、`video_actual_duration_s` 分别来自请求参数和 `ffprobe`；`latency_s` 是端到端 API 等待/处理耗时；
- 成本只有在服务返回费用或配置账号单价后才计算。可在本地 `.env` 设置
  `LLM_INPUT_CNY_PER_1K`、`LLM_OUTPUT_CNY_PER_1K`、
  `VLM_INPUT_CNY_PER_1K`、`VLM_OUTPUT_CNY_PER_1K`、
  `SEEDANCE_CNY_PER_SECOND`/`SEEDANCE_CNY_PER_VIDEO`（或明确的
  `SEEDANCE_CNY_PER_1K_TOKENS`）。未配置时报告 `cost_status=unpriced`；离线 mock
  标记为 `not_applicable`。`.env.example` 提供了带日期的公开刊例价估算档，实际账单仍以控制台为准。

工作站示例 `.env` 已按 2026-09-01 火山方舟公开刊例价配置估算：Seed-2.1-pro
输入/输出 6/30 元/百万 token，Seed-2.0-pro VLM 输入/输出 3.2/16 元/百万
token，Seedance 2.0 纯文本生视频 46 元/百万 provider token。估算仅用于预算，
实际费用以控制台账单、免费额度和账号折扣为准。

真实模式会调用（模型 ID 可通过环境变量覆盖）：

```text
BASE_URL       https://ark.cn-beijing.volces.com/api/v3
VIDEO_MODEL    ep-20260630104958-tmgzb
PLANNER_MODEL  ep-20260624154257-tl6bm
VLM_MODEL      doubao-seed-2-0-pro-260215
EMBED_MODEL    ep-20260610171621-gs9vd
```

视频接口采用异步任务提交和轮询；具体 `content` 结构以账号所在区域的官方文档为准。Seedance 2.0 的 `duration` 在 live 模式要求整数 4--15 秒，代码会在发请求前校验。

## 固定生产管线

默认 `NAVGEN_PIPELINE_PROFILE=fixed`。live/auto（检测到 API key）只允许下面这一条有序流程：

```text
Doubao-Seed-2.1-pro 场景/任务规划
  → Prompt Compiler（固定结构化 Seedance prompt）
  → Seedance 2.0 视频
  → Pre-QC / 抽帧
  → VLM Filter（场景一致性 + 导航修正）
  → VGGT RGB 轨迹
  → smoothed_scaled（窗口 5 + 路线端点先验）
  → Trajectory Filter
  → DatasetPackager / manifest
```

每个运行和样本都会写入 `pipeline_contract`（`navgen-fixed-v1`），用于确认阶段顺序和
轨迹变体。fixed 配置即使在离线模式也拒绝 OpenCV、raw 或其它轨迹变体（离线只允许
`auto`、`route_contract` 或使用同一 canonical 后处理的 `vggt`），避免把实验结果误当成生产样本。
离线 `smoke` 仍使用 mock 视频和 route contract 只验证编排；
推荐使用 `run-fixed` 启动固定 live 管线（`run --mode live` 默认同样采用 fixed profile）；
`compare-trajectory`、`estimate-trajectory` 以及 `NAVGEN_PIPELINE_PROFILE=experiment`
仅用于离线实验。固定 live 在规划或提示词编译失败时会 fail-closed，不会用模板悄悄替代
并继续生成视频。

### 运行指定的 RxR episode

如果任务指令和参考路径已经来自 RxR/Matterport 等数据集，不应再让规划器重写它们。
`run-episodes` 会把原始 episode 导入为 `SceneManifest + TaskSpec + RouteContract`，保留
英文原文、episode/scene/instruction ID 和参考路径，然后从 Prompt Compiler 开始执行同一条
固定管线（Seedance → Pre-QC/VLM → VGGT `smoothed_scaled` → Trajectory Filter → 打包）：

```bash
python3 -m navgen_pipeline.cli run-episodes \
  --episodes data/rxr_sn83yjsr3w2_episodes.json \
  --mode live --remote-prompt \
  --vggt-repo /opt/vggt \
  --out runs/rxr_live
```

完成后将真实调用的样本发布到工作站页面（不会发布 offline/mock 样本）：

```bash
python3 scripts/publish_site_samples.py \
  --run runs/rxr_live --site-demo site/public/demo
```

本次示例数据文件包含 episode `13288`、`13289`、`13290`（scene
`SN83YJsR3w2`）；其中 `13290` 使用源数据中的完整指令，截图里混入的截断/助手文本不会
进入任务。规划参考路径会作为 route prior 留档，但不会冒充视频执行真值。

## 工作站展示

当前演示页面只保留真实 Seedance 调用样本，未调用模型的离线样本不会出现在页面或导出
manifest 中。构建后可在工作站直接启动局域网服务：

```bash
cd /home/beta/hongyu.yao/site
npm run build
npm run start:workstation
```

访问 `http://192.168.110.66:3001`。这里的 `192.168.110.66` 是工作站局域网地址；
默认端口是 `3001`，如需其他端口可使用 `PORT=xxxx npm run start:workstation`。
该内网地址与 Sites 公网地址是两个独立入口，内网部署不会把公网 DNS 解析改成私网 IP。

## 目录结构

```text
navgen_pipeline/
  models.py          # 数据契约与序列化
  planner.py         # 模板规划器、豆包规划器
  prompting.py       # Seedance prompt 编译与可选 Doubao 润色
  ark_client.py      # Responses、Seedance、Embedding API 适配器
  video.py           # 视频生成/抽帧/预筛
  vlm_filter.py      # VLM 判定和离线启发式判定
  trajectory.py      # 位姿导入、路线合同到轨迹的生成
  video_trajectory.py # OpenCV RGB 视频运动估计（仅用于 review）
  vggt_trajectory.py # VGGT 单目 RGB 轨迹估计（原始、仅用于 review）
  trajectory_variants.py # raw / smoothing / route-prior scale 对比变体
  rxr.py            # RxR/Matterport episode 导入与源指令/路径保留
  trajectory_filter.py
  pipeline.py        # 端到端编排与打包
  cli.py             # 命令行入口
schemas/
  sample.schema.json # 样本 JSON 契约
data/
  rxr_sn83yjsr3w2_episodes.json # 本次发布的 13288/13289/13290 源记录
docs/
  pipeline.md        # 流程、分级和轨迹边界
scripts/
  publish_site_samples.py # 将真实 run 的审计产物发布到工作站 demo
tests/
  ...
```

## 为什么离线视频是方块

`smoke`/`offline` 模式使用 `MockVideoGenerator`，故意绘制简单几何场景来验证
编排、抽帧、VLM/轨迹过滤和文件打包；它不是视觉质量评测，也不会伪装成 Seedance
输出。配置 `ARK_API_KEY` 后使用 `--mode live`，视频才会由
`ep-20260630104958-tmgzb` 异步生成真实室内画面。

## 设计边界

### 真实执行轨迹

Seedance 只返回 RGB 视频，不返回机器人位姿。因此控制台会把 planned route 和 observed
trajectory 分开：live 运行默认用 VGGT 单目估计生成 `vggt_monocular_estimate`，并在报告中
标注“未定尺度 / 非地面真值”；offline mock 才使用 route contract 验证编排。设置
`VIDEO_ESTIMATOR=video_vo` 可切换到 OpenCV 视频运动估计。要得到可用于
A 级训练的真实轨迹，请提供仿真器 ground truth 或机器人传感器 JSONL：

```bash
python3 -m navgen_pipeline.cli estimate-trajectory --video video.mp4 \
  --estimator vggt --vggt-repo /opt/vggt \
  --vggt-checkpoint /models/VGGT-1B/model.pt --out runs/vggt_review

NAVGEN_PIPELINE_PROFILE=experiment \
TRAJECTORY_SOURCE=external \
EXTERNAL_TRAJECTORY_PATH=/data/poses/{sample_id}.jsonl \
python3 -m navgen_pipeline.cli run --mode live --count 1 --out runs/live_with_poses
```

VGGT/视频运动估计需要视觉扩展（新环境可执行 `pip install -e '.[vggt]'`）；如果只接入
仿真/机器人 JSONL，则不需要这组依赖。

### VGGT 安装与输入边界

VGGT 不是 NavGen 的强制依赖。工作站首次启用时可以执行：

```bash
git clone https://github.com/facebookresearch/vggt.git /opt/vggt
python3 -m pip install -e '.[vggt]'
# The official requirements pin an older torch; choose a CUDA build compatible
# with the workstation GPU (the example below is suitable for recent GPUs).
python3 -m pip install 'torch>=2.7' 'torchvision>=0.22'
```

如果没有 `VGGT_CHECKPOINT`，估计器会尝试从 `facebook/VGGT-1B` 下载官方权重；生产环境
建议把权重放在本地并显式配置路径。默认从 15 秒视频抽取不超过 48 帧；VGGT 原始输出位于
第一相机水平坐标系的相对尺度中，随后由生产后处理（`smoothed_scaled`）平滑并按路线端点
做显式先验对齐。对齐结果仍是估计轨迹，`ground_truth=false`，不能标为 GT。

### 原始轨迹与后处理对比

可以在一次 VGGT 推理后生成四个变体，避免为每种后处理重复运行大模型：

```bash
python3 -m navgen_pipeline.cli compare-trajectory \
  --video path/to/video.mp4 \
  --reference path/to/planned_trajectory.jsonl \
  --vggt-repo /opt/vggt \
  --vggt-anchor-start 0.8 0.8 \
  --vggt-anchor-end 4.98 3.73 \
  --out runs/vggt_comparison
```

输出 `raw`、`smoothed`、`scaled`、`smoothed_scaled` 四个目录，以及
`comparison.json` 和 `comparison.png`。`smoothed` 使用小窗口三角移动平均；
`scaled` 使用声明的路线起止点做相似变换。尺度变体只是 route prior 的假设，报告会
写入 `scale_anchor=route_endpoint_displacement`、`ground_truth=false` 和
`metrics_are_metric=false`，不能当作真实米制位姿。

流水线现在固化使用 `VGGT_POSTPROCESS=smoothed_scaled`：先做时序平滑，再用声明的路线
起止点做尺度/方向锚定。该锚定仍是 route prior，不是外部真值；如需复现实验基线，可在
`compare-trajectory` 或 `estimate-trajectory` 中显式选择 `raw`、`smoothed` 或 `scaled`，
并调整 `VGGT_SMOOTHING_WINDOW`。

外部文件每行至少是 `{t, x, y, yaw}`，建议同时写入 `z/vx/vy/omega/frame_id/source`。
ROS 2 场景可记录 `/tf`、`/odom`、相机、深度和 IMU 后再转换为该格式；仿真场景应在同一
渲染步导出 RGB 与 agent pose。单目视频的尺度和相机标定无法由 Seedance 自动保证，不能把
`vggt_monocular_estimate` 或 `video_vo_estimate` 当作 ground truth。

Seedance 生成的是 RGB 视觉内容，不天然提供带真实尺度的机器人位姿。因此正式训练数据建议
使用仿真/场景图生成的 metric 轨迹，或带传感器尺度锚点的重建轨迹；纯视频重建轨迹必须带
置信度并单独分级。离线 mock 视频只用于验证管线逻辑，不代表 Seedance 的视觉质量。

控制台地图从 `sample_spec.scene` 读取语义。只有 manifest 显式提供 `regions` 时才绘制多房间
分区；没有分区时按单一开放场景显示，并直接绘制 `objects/obstacles` 的坐标 footprint，避免
用固定的 LOUNGE、STUDY、OFFICE 作为装饰性标签。`room_type` 仅是规划先验；VLM 输出受控的
`observed_scene_label` 且置信度达到 0.7 后，才会写入带来源的 `semantic_label`，否则页面显示
`UNVERIFIED SPACE`，并把先验单独标注为 planner prior。

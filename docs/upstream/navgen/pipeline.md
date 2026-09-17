# NavGen 基础管线说明

## 数据流

```text
Scene/Task Planner (Doubao-Seed-2.1-pro / template)
      │  SceneManifest + TaskSpec + RouteContract
      ▼
Seedance Prompt Compiler (+ optional Doubao Prompt Refiner)
      │  camera / scene anchors / timeline / negatives
      ▼
Seedance 2.0（live）/ MockVideoGenerator（offline）
      │  video.mp4
      ▼
Pre-QC → 抽帧/接触表 → Local/Ark Embedding
      │
      ▼
VLM Filter（Scene consistency + Navigation correction）
      │  pass / review / reject、有效区间、纠正指令
      ▼
Trajectory Generator（外部真值、VGGT 单目观测或 route prior）
      │
      ▼
Trajectory Filter（速度/加速度/角速度/抖动/碰撞/终点停留）
      │
      ▼
DatasetPackager → samples/<sample_id>/ + manifest.jsonl
```

## 固定生产契约

默认配置为 `NAVGEN_PIPELINE_PROFILE=fixed`，live/auto（有 API key）固定执行：

`场景/任务规划 → Prompt Compiler → Seedance 2.0 → Pre-QC/抽帧 → VLM Filter →
VGGT → smoothed_scaled（窗口 5、路线端点先验）→ Trajectory Filter → 打包`。

运行和每个样本都会记录 `pipeline_contract.name=seedance_vggt_navigation`、
`version=navgen-fixed-v1` 以及阶段列表。固定 profile 下切换到 `video_vo`、`raw`、
`smoothed`、`scaled` 或外部轨迹会在启动时拒绝（offline 仅允许 `auto`、
`route_contract` 或同一 canonical 后处理的 `vggt`）；这些选项只能通过
`NAVGEN_PIPELINE_PROFILE=experiment` 用于离线/A-B 实验。offline/mock 运行仍保留
route contract 作为编排测试，并在 contract 中同时记录 `runtime_trajectory_variant`。
固定 live 在规划或提示词编译失败时 fail-closed，不会用模板任务继续发起 Seedance 请求。

推荐入口：`python3 -m navgen_pipeline.cli run-fixed --count 1 --out runs/live_fixed`。

### 指定 RxR episode 的生产入口

对于已经带有自然语言指令和参考路径的 RxR/Matterport episode，使用 importer 保留源数据，
不让 generative planner 改写任务；importer 之后的 Prompt Compiler、Seedance、两级过滤、
VGGT 和打包阶段与 fixed 合同相同：

```bash
python3 -m navgen_pipeline.cli run-episodes \
  --episodes data/rxr_sn83yjsr3w2_episodes.json \
  --mode live --remote-prompt --vggt-repo /opt/vggt \
  --out runs/rxr_live
python3 scripts/publish_site_samples.py \
  --run runs/rxr_live --site-demo site/public/demo
```

`sample.json` 会同时保留 `source_episode_id`、`source_scene_id`、`source_instruction_id`、
原始英文指令和 `source_reference_path`。参考路径只是 prompt/尺度锚定的 route prior；
Seedance RGB 视频上的 VGGT 轨迹仍标记 `ground_truth=false`，不能直接当作机器人 GT。

每个样本都会保留原始视频、关键帧、接触表、轨迹和两个过滤器报告。只有两个过滤器都通过且没有待处理的指令纠正时，`quality.tier=A`、`training_eligible=true`；`B` 级保留用于人工复核，拒绝样本也不删除，便于审计和调阈值。

## 用量与成本审计

`usage_summary.json`（运行级）和 `samples/<sample_id>/usage.json`（样本级）记录每个阶段的：

- Responses 的输入、输出、推理和总 token，以及每次请求延迟；
- Seedance 返回的 provider token counter（单独命名，不与文本 token 合计）；
- 请求视频时长、`ffprobe` 测得的实际时长、分辨率/比例和任务等待耗时；
- `cost_cny`、`estimated_cost_cny`、`billable_cost_cny` 与 `cost_status`。

默认不把公开价格冒充实际账单。若需要本地估算，请把账号控制台的当前单价注入环境变量：
`LLM_INPUT_CNY_PER_1K`、`LLM_OUTPUT_CNY_PER_1K`、
`VLM_INPUT_CNY_PER_1K`、`VLM_OUTPUT_CNY_PER_1K`、
`SEEDANCE_CNY_PER_SECOND` 或 `SEEDANCE_CNY_PER_VIDEO`；只有显式配置后才会显示估算金额。

本次工作站真实 Seedance 2.0 单样本审计（`runs/live_smoke_real_v3_20260901`）记录了：
文本 6,899 tokens、Seedance provider counter 324,900、请求 15.0 s / `ffprobe`
实际 15.042 s、任务等待 206.695 s，输出 1280×720、24 fps、约 6.91 MB。
该账号的任务响应没有返回费用。本项目附带的工作站 `.env` 使用公开刊例价作估算：
Seed-2.1-pro 输入/输出 6/30 元/百万 token，Seed-2.0-pro VLM 输入/输出
3.2/16 元/百万 token，Seedance 2.0 纯文本生视频 46 元/百万 provider token。
这只是带来源和日期的估算，不代表账号折扣、免费额度或最终账单；换成控制台单价后会自动重算。

按这张价卡重算本条样本：规划 `¥0.0594`、提示词润色 `¥0.0378`、VLM
`¥0.0098`、Seedance `¥14.9454`，合计约 `¥15.0524`；服务方未返回账单金额，
因此 `billable_cost_cny` 仍为空。

演示站点当前批次保留 3 条真实 Seedance 调用（RxR episode `13288`、`13289`、`13290`），
不把 offline/mock 结果混入列表。局域网部署使用工作站地址
`http://192.168.110.66:3001`；启动命令见 README，端口可按需要替换。

## Offline 与 live

```bash
python3 -m navgen_pipeline.cli smoke --count 3 --out runs/offline_smoke
python3 -m unittest discover -s tests -v

export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export ARK_API_KEY='<inject-through-secret-manager>'
python3 -m navgen_pipeline.cli run-fixed --count 1 --out runs/live_smoke
```

`offline` 不访问网络，mock 视频只用于验证编排、数据契约和过滤逻辑；`live` 使用用户提供的 Seedance endpoint、Responses VLM/规划器和可选 Embedding SDK。API key 永远从环境变量读取，日志和 `safe_dict()` 会脱敏。

## Prompt 为什么单独成阶段

视频模型不应从“走到柜子旁”自行猜测房间布局、目标身份和停止时刻。Prompt Compiler
把结构化合同编译成可复现的 Seedance prompt，固定以下锚点：

可以在不调用视频服务的情况下单独导出并审阅这一阶段：

```bash
python3 -m navgen_pipeline.cli plan --count 3 --duration 15 --out runs/prompt_preview
```

每个样本会同时写出 `seedance_prompt.txt`、`negative_prompt.txt` 和带检查项的
`prompt_report.json`，供人工抽检或作为后续 Seedance 请求的输入。

- 第一视角、相机高度、帧率、比例和时长；
- 场景尺寸、静态物体外观/相对位置和动态目标身份；
- 每个航点的时间线、终点距离、朝向和 terminal dwell；
- 速度/加速度/安全间隙以及禁止穿墙、变形、跳帧、字幕等负面约束。

在 live 模式需要更自然的语言时，可打开 `--remote-prompt` 让 Doubao 做第二次润色；
润色结果会经过必需段落检查，失败则回退到确定性编译结果。

## 轨迹边界

Seedance 输出 RGB 视频，不直接提供机器人坐标系下的真实米制位姿。因而“视频生成成功”不能等同于“执行轨迹真实”。当前版本把轨迹来源写入 `trajectory_report.json`：

| source | 含义 | 是否可作为训练真值 |
| --- | --- | --- |
| `external_ground_truth` | 仿真器或机器人传感器导出的位姿流 | 可以（仍需两级过滤） |
| `video_vo_estimate` | 从 RGB 连续帧估计的相机运动；可用起止点做尺度锚定 | 不可以，默认 B 级人工复核 |
| `vggt_monocular_estimate` | VGGT 相机头从 RGB 多帧预测的原始相对相机轨迹 | 不可以，默认 B 级人工复核 |
| `vggt_monocular_estimate_smoothed` | 原始 VGGT 经过小窗口三角移动平均；不改变尺度 | 不可以，默认 B 级人工复核 |
| `vggt_monocular_estimate_scaled` | 原始 VGGT 按路线起止点相似变换；是 route prior 尺度假设 | 不可以，默认 B 级人工复核 |
| `vggt_monocular_estimate_smoothed_scaled` | 先平滑、再按路线起止点相似变换 | 不可以，默认 B 级人工复核 |
| `route_contract` | 规划航点插值，只是 planned prior | 不可以作为真实执行位姿 |

当前 live 管线固定选用 `vggt_monocular_estimate_smoothed_scaled`；其余 VGGT 变体仅用于
离线对比，不写入演示站的样本轨迹。

`TRAJECTORY_SOURCE=auto` 在 live Seedance 运行中会优先执行 VGGT，offline mock 才保留
`route_contract` 以验证编排。生产默认 `VGGT_POSTPROCESS=smoothed_scaled`：只读取 RGB
帧，使用 camera head 输出相机外参/内参，转换到第一帧水平坐标系后做时序平滑，并按声明
的路线端点做相似变换。尺度仍是 route prior，报告明确 `ground_truth=false`、
`metrics_are_metric=false`；`raw` 仍可显式用于基线实验。可以单独运行：

```bash
python3 -m navgen_pipeline.cli estimate-trajectory \
  --video path/to/video.mp4 \
  --estimator vggt --vggt-repo /opt/vggt \
  --out runs/vggt_review
```

VGGT 的可选依赖是 `pip install -e '.[vggt]'`；只接入外部仿真/机器人位姿时不需要安装。若要继续使用旧的 OpenCV 估计器，可传 `--estimator opencv` 或设置 `VIDEO_ESTIMATOR=video_vo`。

如需比较后处理效果，可用一次 VGGT 推理生成四种变体：

```bash
python3 -m navgen_pipeline.cli compare-trajectory \
  --video path/to/video.mp4 --reference path/to/planned_trajectory.jsonl \
  --vggt-repo /opt/vggt --vggt-anchor-start 0.8 0.8 \
  --vggt-anchor-end 4.98 3.73 --out runs/vggt_comparison
```

其中 `smoothed` 是三角移动平均，`scaled`/`smoothed_scaled` 使用路线端点作为显式
尺度先验；所有变体仍为 `ground_truth=false`，且尺度变体会标记
`metrics_are_metric=false`。

正式训练集建议采用以下两种采集方式：

1. **仿真真值（首选）**：在 Isaac Sim、Habitat-Sim 等环境每个渲染步同时导出 RGB 和 agent/base 位姿，时间戳使用同一仿真时钟。
2. **真实机器人遥测**：用 ROS 2 `ros2 bag record` 同步记录 `/tf`、`/tf_static`、`/odom`、`/cmd_vel`、相机、深度和 IMU；以 `map→base_link`（或明确声明的 `odom→base_link`）作为执行位姿，再按图像时间戳对齐。

统一导出为 JSONL，每行至少包含：

```json
{"t": 0.0, "x": 0.8, "y": 0.8, "z": 0.0, "yaw": 0.0,
 "vx": 0.0, "vy": 0.0, "omega": 0.0,
 "frame_id": "map", "source": "external_ground_truth"}
```

把文件交给管线：

```bash
NAVGEN_PIPELINE_PROFILE=experiment \
TRAJECTORY_SOURCE=external \
EXTERNAL_TRAJECTORY_PATH=/data/poses/{sample_id}.jsonl \
python3 -m navgen_pipeline.cli run --mode live --count 1 --out runs/live_with_poses
```

只有带外部真值（或明确的仿真真值）的样本才会被允许进入 A 级；纯单目视频即使画面连续，也只作为带不确定性的观测轨迹展示。

## 场景语义可视化

地图监控不再内置固定的 `LOUNGE / STUDY / OFFICE` 房间框。前端读取每个样本
`sample.json` 中的 `sample_spec.scene`：`width_m/depth_m` 决定坐标范围，`objects` 和
`obstacles` 决定物体/障碍物 footprint，只有 `scene.regions` 明确提供边界和标签时才绘制
多个语义区域。`regions` 为空表示一个开放场景，界面会标注“语义未确认”，而不会根据
`room_type` 猜造房间布局。`room_type` 只是规划阶段的先验；VLM 只有在输出受控的
`observed_scene_label` 且置信度不低于 0.7 时，才会把它写入 `semantic_label`，同时保留
`semantic_source` 和 `semantic_confidence`。先验与观测不一致的样本进入 review，不会静默
修改视频任务。

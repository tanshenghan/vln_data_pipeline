# EVT-Bench 原生环境探测与 HM3D 共享输入

## 状态边界

`probe_evt_bench.py` 只做真实 GPU 环境探测：原生动态人物、RGB、panoptic、
机器人/目标位姿、模拟时间与预览视频。它运行 TrackVLA 仓库公开的
`GTBBoxAgent` 基线（使用 GT 检测框），**不是 TrackVLA/LightNav 推理，也不是
专家轨迹或完整 benchmark 评测**。不能把视频生成成功当成 BetaNav GT 验收。

EVT-Bench 是移动目标跟随任务。原标注的 `start_position` 是目标人的起点，
机器人起点在 `info.robot_position`，`goals` 是人的路线目标，不是机器人的
固定终点。不能将这些字段直接填进 R2R 回放器，也不能在截断处制造 STOP。
本探测不生成 FM/RVQ/apos/opos、深度或 LeRobot 数据；报告始终保持
`betanav_gt_complete=false`、`web_collection_accepted=false`。
网页仍未开放 EVT-Bench 生产采集，独立跟随契约需另行确定。

## 独立环境

上游：https://github.com/ShaoanWang/TrackVLA
本次测试版本：`c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd`。
使用 Python 3.9、Habitat-Sim 0.3.1 headless + Bullet、NumPy 1.26.4，
以及该仓库内的 Habitat-Lab fork；不要安装进 R2R Habitat、导出或推理环境。
大型仓库、人体资源、场景和输出都不进入本项目 Git。

```bash
# 按自己的目录替换下面的环境变量，不要改导师的环境。
conda create -p data_pipeline/.venv/evt --override-channels \
  -c conda-forge -c aihabitat python=3.9 habitat-sim=0.3.1 withbullet headless 'numpy<2' pip
data_pipeline/.venv/evt/bin/python -m pip install 'numpy==1.26.4' \
  'opencv-python<4.12' 'gym==0.23.0' -e "$EVT_REPO/habitat-lab"

PYTHONDONTWRITEBYTECODE=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
flock -w 1 runtime/gpu.lock data_pipeline/.venv/evt/bin/python \
  data_pipeline/probe_evt_bench.py \
  --upstream "$EVT_REPO" --humanoids "$EVT_HUMANOIDS" \
  --scenes-root "$VLN_SOURCE_ROOT/scene_datasets" \
  --scene-family mp3d --mode STT --split train --steps 30 \
  --output "$VLN_RESOURCE_ROOT/validation/NEW_EVT_PROBE"
```

`--humanoids` 指向含 `humanoid_data/` 的目录；`--scene-family` 支持
`mp3d` / `hm3d`，`--mode` 支持 `STT` / `DT` / `AT`，`--split` 支持
`train` / `val`。默认选择第一个已有场景的 episode，也可传 `--episode-id`。
`--steps` 为 1–300，默认 30；`--gpu` 默认 0；输出目录必须不存在。
每个新场景都会重算该 fork 所需的 navmesh 到个人 `workspace/`；共享场景
仅以单文件只读用途的符号链接接入，绝不把整个共享目录链接为可写缓存。

输出 `report.json`、原始 episode 快照、实际配置、PNG、panoptic NPY、
`frames.json` 和 `review.mp4`。位姿保留 Habitat Y-up、米、world-from-base；
记录模拟器实际时钟，不把 10 FPS 预览速度解释为物理动作时长。
探测会重新读取保存的图像/位姿，检查真实机器人与目标移动及目标可见像素。

## HM3D 共享目录与授权

下述 2026-09-27 示例范围是历史记录。2026-09-28 已加入授权全量下载、校验安装和
ObjectNav 版本隔离工具，最新操作及实时完成判定见 [HABITAT_OBJECTNAV.md](HABITAT_OBJECTNAV.md)。
启动下载不等于已经装齐，请以安装状态清单为准。

场景入口统一使用 `VLN_SOURCE_ROOT/scene_datasets/hm3d`。本机按用户指定
部署在 `/home/beta/.local/share/vln-pipeline/scene_datasets/hm3d`，与 MP3D 同级。
2026-09-27 仅下载了 Matterport 官方公开示例：

| 官方 split | 示例场景 |
|---|---|
| train | 00337-CFVBbU9Rsyb、00770-NBg5UqG3di3 |
| val | 00861-GLAQ4DNUx5U |

实际数据在 `example/`，`train/`、`val/` 为各场景的相对链接，split 对应
官方 `metadata.csv`。下载来源和 SHA-256 写在共享目录 `README.local.md`。
没有下载场景 semantic 文件。**目录存在不代表完整 HM3D 就绪**，任何采集
必须逐 episode 检查 `.basis.glb`。正式 HM3D 场景仍需账号持有人完成
[官方授权](https://github.com/matterport/habitat-matterport-3dresearch)；使用受
原始学术/非商业条款约束，不能因为示例可直接下载就推断获准商业使用。

ScaleVLN 原始离散图路线与连续 Habitat 轨迹不是同一输入格式。所选连续源
[StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data)
受访问授权限制。早先探测返回 `403 GatedRepo`；2026-09-27 用户完成授权后，
已成功下载固定版本并完成连续轨迹适配。最新可采集范围、完整链路和验收方法见
[SCALEVLN.md](SCALEVLN.md)，早先探测报告仅保留历史状态。
轨迹授权不等于完整 HM3D 场景授权，公开示例也不保证全部路线的场景覆盖。

# ScaleVLN 连续轨迹接入

## 使用范围与共享输入

这里接的是 [StreamVLN 发布的 ScaleVLN 连续轨迹子集](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data)，
不是原版 ScaleVLN 490 万条离散图路线，也不是下载预渲染视频后直接冒充本项目观测。
固定 revision `dc61ee9b4e90aa7ba63c1163b2134df5610dccb9`，实际含 **155,098** 条
英文合成指令轨迹、795 个 HM3D train 场景。当前支持 `train / guide / all / local`，
`all` 在这里仅代表不做语言过滤，源指令是英文。暂不提供全划分分片、云采集或 Seedance。

输入布局遵循 `VLN_SOURCE_ROOT`（未设置时沿用 `VLN_RESOURCE_ROOT`）：

```text
datasets/ScaleVLN_StreamVLN/
  raw/scalevln_subset_150k.json.gz
  raw/annotations.json
  raw/README.md
  train/train.json.gz
  train/train_gt.json.gz
  source_manifest.json
scene_datasets/hm3d/train/<number-scene>/<scene>.basis.glb
scene_datasets/hm3d/train/<number-scene>/<scene>.basis.navmesh
```

本机共享根目录为 `/home/beta/.local/share/vln-pipeline`。2026-09-27 轨迹授权已验证，
上述完整轨迹子集已准备好；已安装 HM3D 官方示例中的两个 train 场景
`00337-CFVBbU9Rsyb`、`00770-NBg5UqG3di3`，覆盖 **414 条路线 / 2 个场景**。
**这不是完整 HM3D 场景覆盖，也不是 414 条全部逐条通过验收。** 页面显示当前覆盖，
默认仅从可读 mesh + navmesh 的样本中选取；每条实际回放仍需通过路线与终点检查。
更多场景安装到同一布局后会自动参与可用性检查，毋须重做源转换。
轨迹访问授权与 HM3D 完整场景授权分别管理，仍须遵守各来源条款。

源文件、场景和导师的解释器只读复用。新采集和验收结果写入个人
`VLN_RESOURCE_ROOT/runs/`、`validation/`，索引写入个人 `VLN_RUNTIME_ROOT`。
代码不写死个人目录；分支合并后，导师复用相同共享输入，重建自己的索引并重启
自己的 API/Worker/Atlas 即可加载新入口，不需要接入开发者的运行数据库。

## 从原始源准备索引

账号持有人完成访问授权后，下载固定版本的 `ScaleVLN/scalevln_subset_150k.json.gz`、
`ScaleVLN/annotations.json` 和数据卡。密钥留在本机认证存储，不写进仓库或命令参数。
适配器核对两个原始文件的 SHA-256，只有完全一致才转换；输出目录必须不存在：

```bash
.venv/bin/python data_pipeline/scalevln_source.py \
  --episodes /absolute/download/ScaleVLN/scalevln_subset_150k.json.gz \
  --annotations /absolute/download/ScaleVLN/annotations.json \
  --readme /absolute/download/README.md \
  --output /absolute/shared/vln-pipeline/datasets/ScaleVLN_StreamVLN

.venv/bin/python configs/build_source_catalog.py
.venv/bin/python -m catalog.build_vln_geometry \
  --catalog runtime/vln.sqlite --output runtime/vln-geometry.sqlite
```

本机已经完成转换，不要覆盖重跑。`source_manifest.json` 是最后写入的完成标记，
保留源版本、原始/转换文件哈希、场景数量、动作解释和适配器哈希。采集前重新验证转换文件哈希。
已有相同原始输入的部署可直接共享准备好的目录，无需重复下载。
如果已部署 `vln-videos.sqlite`，它也绑定源索引快照，必须按原来的来源清单重建后
再重启 Atlas；只有平台数据库登记的产物时，可运行 `.venv/bin/python -m workers.catalog_sync`。
不要丢弃手工导入的视频来源清单。

## 代码怎样接通

1. `scalevln_source.py` 按 episode ID 关联起点、旋转、目标和动作标注，同时核对
   指令文本与 scene/video 身份。场景路径规范化到 `hm3d/train/`。
2. `vlnce_source.py` 和 `settings.dataset_root` 增加 `scalevln`；源索引注册
   `scalevln-streamvln-150k`。API 的 `catalog_service.py` 默认只选已有场景的路线，
   `capacity.py` 启动前逐条检查 mesh、navmesh、帧数和个人输出空间。
3. Worker 复用 `collect_r2r.py` 的真实 Habitat 回放。保留源动作的 0.25 m 前进、
   15° 转向，不做 RxR 式转向展开；以本项目相机参数重新渲染 RGB、深度和位姿。
4. 原数据卡明确约定首个 `-1` 是占位，STOP 被省略，因此仅对此固定哈希版本采用
   `actions[1:] + [0]`。必须实际到达目标（末帧测地距离 < 0.5 m），且路线偏差通过
   原有 0.5 m 检查，才允许生成终止监督；其他 EOF、损坏或截断输入不会自动变成 STOP。
5. 从真实逐帧位姿生成 FM-30、RVQ-10、apos、opos，再走现有 LeRobot 导出、官方加载器、
   独立几何校验、视频预览和 Atlas 产物登记。无需 VLM/Ark 配置。

`reference_path` 只有稀疏路标，GT sidecar 中明确标记 `locations_source=source_reference_path`。
它只用于路线检查/浏览，几何索引拒绝把它当作数据集逐帧评测 GT。
真正的监督来自本次渲染保存的 body/camera poses。生成时继续使用 470×280、HFOV 120°、
相机高 0.88 m、pitch 0°、独立 mask、`fps=1` 合成时间轴、`physical_dt_s=null`；
不导出前进/左转/右转分类标签。

## 网页采集与验收

打开本人的工作站 GT 页，选择 **ScaleVLN（StreamVLN 连续轨迹）**，使用 train、
Guide、全部语言、本机采集，先设置采集数量 1。可留空 episode ID，让平台只选已有
场景的路线；也可指定 `25465`。点击“检查采集范围与容量”，然后“开始采集”。
任务成功后页面应有 LeRobot dataset、独立校验报告和可播放的 GT 预览视频，
原始 capture 另在个人任务目录的 `capture_archive/` 保留审计副本。
指定未下载场景的 ID 会被明确拒绝，不会偷偷替换成另一条路线。

独立环境验收入口（读取当前 `.env`）：

```bash
.venv/bin/python data_pipeline/verify_environment.py --dataset scalevln --episode-id 25465
```

需要自动验证真实网页点击时，使用 `verify_scalevln_web.py --url <自己的服务地址>`
`--chromium <浏览器可执行文件> --output <新证据目录> --start-collection`。
此命令依赖 Playwright，**确实会新建一个采集任务并使用 GPU**，不属于只读页面巡检；
默认测试空 episode ID 的可用场景选择，也可加 `--episode-id 25465`。
截图、任务 ID、容量结果、独立校验结果与视频播放证据保存到指定目录。

结果写入 `runtime/scalevln_environment_validation.json`，不会覆盖已有 R2R 工作站验收。
该过程验证真实渲染、全部四类监督、LeRobot 导出，并把原始 capture 移到 archive 后
独立读回正式数据集，最后生成预览。成功仅证明所测路线和完整链路通过，不等于全量逐条通过。

2026-09-27 已完成两种场景的链路验收：CLI episode `25465`（37 帧），网页实际点击
自动选取的 `156815`（62 帧）。网页任务 `7d8b117a-e236-4162-8c73-dc6b87f20eba`
一次执行通过，视频播放和产物来源关联通过。验收路径、哈希、测试范围和限制见
[小型验收报告](../reports/scalevln_acceptance_20260927.json)。

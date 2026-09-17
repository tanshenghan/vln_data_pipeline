# R2R / RxR LeRobot 地面 GT 视频：来源、预览与保存

更新：2026-09-17。预览确认后，主网站已改用原生 **470×280、8 FPS** 的 LeRobot GT 视频，
显示在俯视图上方，并移除人工筛选表单。首页默认展示全部 **171,066 条指令**。
已接入 **71,071 条观测**，全量后台渲染已启动；不是只在点击时生成几个样本，也不代表全部视频已完成。
网页顶部每 10 秒刷新进度，展开后显示各划分的覆盖情况。公网入口见 [public_access.md](public_access.md)。

## 全量覆盖

| 数据集 / 划分 / 角色 | 全部指令 | 已有观测 | 缺少观测 |
|---|---:|---:|---:|
| R2R train guide | 10,819 | 10,819 | 0 |
| R2R val_seen guide | 778 | 0 | 778 |
| R2R val_unseen guide | 1,839 | 0 | 1,839 |
| R2R test guide | 3,408 | 0 | 3,408 |
| RxR train guide | 60,300 | 60,252 | 48 |
| RxR train follower | 58,752 | 0 | 58,752 |
| RxR val_seen guide | 6,746 | 0 | 6,746 |
| RxR val_seen follower | 6,571 | 0 | 6,571 |
| RxR val_unseen guide | 11,006 | 0 | 11,006 |
| RxR val_unseen follower | 10,847 | 0 | 10,847 |
| 合计 | **171,066** | **71,071** | **99,995** |

RxR 接入 1,077 个验证完成的 shard，采集任务的 shard 00344 因专家回放偏离失败，少 48 条。
未通过校验和已废弃的采集版本不参与索引；其余划分需要采集各自的 RGB、深度、位姿和标签。
167,658 条指令已有评测 GT，R2R test 的 3,408 条没有公开 GT。GT 坐标不等于 RGB 观测。

## 视频画什么

- 每个保存的 RGB 观测对应一帧视频，按当前相机内参与真实相机位姿投影。
- 蓝线及步号：当前时刻之后最多 10 个真实动作步的机体地面位置，保留世界高度；转向也计入步数。
- 绿色 apos：保存的可见导航点，通常来自 `min(t+10, N-1)`，必要时沿未来路径回退。
- 粉色 opos：保存的最终地面停靠点，固定在末帧世界位置。
- RGB 中隐藏遮挡、深度无效或视野外的轨迹部分；完整未来坐标保存在同名 JSON 中。
- apos/opos 仅在 `state=point` 且有效时绘制；重合点以同心圆显示，不挪动坐标。
- opos 保留后半段窗口和真实深度可见性限制；其他帧显示未开放、不可见或未知状态。
- 尾部只画实际存在的未来步，不重复末帧凑够 10 步。
- float32 舍入到 width/height 边界的像素不钳入图像；绘制状态为 `boundary_not_drawn`，原标签保留在 JSON。

源数据的 `fps=1`、`timestamp=frame_index` 是合成索引时间，`physical_dt_s=null`。
正式视频 **8 FPS（8 Hz）** 仅用于观看，每条原始观测都保留，不能解释为真实机器人控制频率。
Episode 515 的 41 帧播放时长为 5.125 秒。

## 原始数据存在哪里

R2R 源数据根目录：

```text
/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912/
├── data/chunk-*/file-*.parquet
├── meta/info.json
├── meta/episodes/chunk-*/file-*.parquet
├── meta/beta_nav.json
├── meta/summary.json
├── meta/quality.json
├── meta/tasks.parquet
├── meta/provenance/
└── assets/action_tokenizer/
```

这是 image-backed LeRobotDataset v3.0，R2R train 共 10,819 个指令 episode、631,244 帧。
RxR 使用同样的结构，根目录为：

```text
/mnt/cfs/data/beta_nav/lerobot_rxr_train_guide_470x280_rxrnav_20260916.work/shards/<shard>/
```

虽然 RxR 父目录名含 `.work`，每个接入的 shard 都是独立、自包含、验证完成的数据集，
必须有 `meta/validation.json` 且 `complete=true`，不是读取 scratch。网站使用期间不要移动或删除这些 shard。

训练数据没有要求独立的 RGB 文件夹或 `videos/` 目录：

| 列或文件 | 实际内容 |
|---|---|
| `observation.images.rgb` | Parquet 中内嵌的无损 PNG 字节，原始尺寸 470×280 |
| `observation.depth_m` / `observation.depth_valid` | float32 米制 camera-Z 深度和独立有效 mask |
| `observation.world_from_body` | 世界坐标中的实际机体变换，平移为地面位置 |
| `observation.world_from_camera_cv` | OpenCV 相机到世界变换；相机 +X 右、+Y 下、+Z 前 |
| `action.fm.*` | 干净的未来 30 步路径、有效 mask、未来帧号、垂直偏移；绘图使用前 10 步对应的真实位姿 |
| `action.apos.*` / `action.opos.*` | 原图像素、世界点、目标世界点、token、网格、状态和原因 |
| `source.episode_id` / `source.scene_id` | 原始 R2R episode 和场景标识 |
| `episode_index` / `frame_index` / `index` | LeRobot 内部 episode 编号、episode 内帧号、全局行号 |
| `meta/episodes` | episode 对应的数据文件和全局帧范围、指令 |
| `meta/beta_nav.json` | 相机 K、可见性配置、每帧哈希及来源；文件较大，代码流式读取 |
| `meta/summary.json` | 适合人工查看的小型概要，避免直接打印完整 beta_nav.json |

不要把 `source_episode` 指向的历史 `.work/scratch` 当作现有 RGB 来源；采集 scratch 可能已清理。
本次渲染只依赖最终数据集中的 Parquet 与元数据，不重新采集、不修改源文件。

截图中的原始 **Episode 515** 对应 LeRobot **episode_index=27**，共 41 帧。
位于：

```text
/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912/data/chunk-000/file-007.parquet
```

全局行区间为 `[981, 1022)`；场景为 `17DRP5sb8fy`，完整指令为
“Exit the bedroom, enter the bathroom, wait at the toilet.”。
网页的 episode ID 使用原始标识，不能直接拿它当 LeRobot episode_index 查询。

## 预览和最终视频位置

早期 5 FPS 样片目录（保留用于比较，主网站使用下方的 8 FPS 正式输出）：

```text
/mnt/pfs/tanshenghan/vln4project/data_web/runtime/lerobot-preview-20260917/
├── index.html
└── r2r/train/17DRP5sb8fy/episode_515/
    ├── gt10_apos_opos_470x280.mp4
    ├── gt10_apos_opos_470x280.png
    └── gt10_apos_opos_470x280.json
```

MP4 为 H.264/yuv420p/faststart，41 帧、8.2 秒、严格 **470×280**。
使用 `--scale 1 --rgb-only` 在原始 RGB 尺寸上叠加轨迹与两类点，不增加画外边栏。
指令、图例和状态说明显示在播放器外。旧的带边栏预览保留供比较，当前播放器使用 470×280 版本。
PNG 选择第 24 帧作为海报，便于同时检查两个点。
JSON 保存原始来源、内部索引、校验信息，以及每帧未来位置和 apos/opos 的状态、坐标及原因。
预览页支持播放、速度调整、拖动帧号和逐帧查看。

查看预览：[本地播放器](http://127.0.0.1:8790/)。远程服务器需要转发 8790 端口。
主 VLN 网站仍使用 8788；这两个入口不同，预览不会自动成为正式网站媒体。

**当前全量主网站实际保存目录**：

```text
/mnt/cfs/data/beta_nav/visual/vln_gt_videos_8fps_v1/<r2r或rxr>/<split>/<role>/<scan>/episode_<source_episode_id>/
```

每条 episode 保存同名 MP4、PNG 海报和 JSON。
例如 Episode 515 的正式视频路径是：

```text
/mnt/cfs/data/beta_nav/visual/vln_gt_videos_8fps_v1/r2r/train/guide/17DRP5sb8fy/episode_515/gt10_apos_opos.mp4
```

视频是可重新生成的衍生产物，不改写训练 Parquet、`meta/info.json` 或原始标签。
后续效果版本使用新版本目录，不覆盖已确认成片。
后台任务使用 8 个 CPU worker；网页也可以优先请求尚未生成的样本。跨进程文件锁避免重复生成。
临时文件保存在该 episode 的 `.render-*` 子目录，成功后发布 MP4、PNG，最后写 JSON 作为完成标记。
失败任务可在网页重试，批量任务支持断点续跑，跳过通过校验的成片。

当前 71,071 条指令有 11,991 组不同内容哈希。仅当场景、相机、RGB/深度/mask、位姿与标签数组、
samples 哈希完全一致时才复用 MP4/PNG（优先硬链接，跨文件系统复制），不依靠相同 episode ID 猜测。
每条指令仍有自己的 JSON，保存本条来源、内部编号、指令及 `shared_video_from_record_id`，明确披露复用关系。
旧 R2R 源目录下 `review/gt10_apos_opos_8fps_v1/` 成片保留作历史记录，当前网站使用上述统一目录。

## 代码和读取流程

实现位于 [render_episode_gt.py](/mnt/pfs/tanshenghan/vln4project/LightNav-0/visual/render_episode_gt.py)：

1. 索引构建时流式读取元数据并压缩保存每条 episode；批量渲染读取缓存，核对源指纹后交给 `load_lerobot_episode()`。
2. 从 episode 索引确定 Parquet 文件和全局行范围，按内部编号筛选，只加载选中的 episode；支持跨文件。
3. 核对帧号、来源、指令、clean FM 与真实未来位置的一致性，核对有效 pointing 世界点的投影像素。
4. 每帧从 Parquet 解码 RGB/深度并核验原始哈希，使用相机 K 与位姿进行地面投影和遮挡检查。
5. 编码到临时 MP4，检查完整帧数后发布成片；同时写海报和逐帧 JSON，已有文件拒绝覆盖。

不使用 RVQ 解码轨迹替代真实未来位置，也不把目标在视野外时的位置钳到图片边缘。
运行参数和可重跑命令见
[数据管线可视化说明](/mnt/pfs/tanshenghan/vln4project/LightNav-0/data_pipeline/docs/VISUALIZATION.md)。

## 与现有网站索引的关系

网站仍通过以下文件读取指令、原始评测 GT 和俯视图信息：

- `/mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln.sqlite`
- `/mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln-geometry.sqlite`
- `/mnt/pfs/tanshenghan/vln4project/data_web/runtime/topdown/`

视频映射单独保存在 `/mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln-videos-all.sqlite`。
source、dataset、split、role、Scan、原始 episode ID、去除首尾空白后的完整指令必须匹配；
元数据有 instruction_id、language 时也必须匹配。当前 71,071 条匹配，0 条未匹配。
UI 的“可生成 GT 视频”统计和视频筛选使用该索引。
`vln-videos-all.progress.sqlite` 持久保存生成进度，`vln-video-batch.log` / `.pid` 保存任务日志及进程号。
旧模型媒体仍保留在原目录和历史主索引中，网站详情不再展示它们。

`build_vln_gt_videos.py` / `vln_video_collection.py` 构建映射与压缩元数据缓存；`vln_gt_video.py` 处理队列、共享内容和进度。
这两个网站适配器调用同一个 `LightNav-0/visual/render_episode_gt.py` 进行实际渲染。
构建命令只建立来源映射，不提前渲染所有轨迹；使用包含 ijson 的已有环境：

独立克隆 Codeup 的 `data_web` 时，先按 `requirements-vln-video.txt` 安装可选视频依赖，
将 `vln_video_sources.example.json` 复制为 `runtime/vln-video-sources.json` 并调整源路径。
下方工作区命令的 `--renderer` 可替换为本仓库的 `visual/render_episode_gt.py` 绝对路径，
`--python` 替换为已安装依赖的 Python；该随库快照与工作区维护源逐字节一致，见 [visual/README.md](../visual/README.md)。

```bash
/mnt/pfs/tanshenghan/vln4project/LightNav-0/.venv-train/bin/python \
  /mnt/pfs/tanshenghan/vln4project/data_web/build_vln_gt_videos.py \
  --catalog /mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln.sqlite \
  --sources /mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln-video-sources.json \
  --output /mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln-videos-all.sqlite \
  --video-root /mnt/cfs/data/beta_nav/visual/vln_gt_videos_8fps_v1 \
  --renderer /mnt/pfs/tanshenghan/vln4project/LightNav-0/visual/render_episode_gt.py \
  --python /mnt/pfs/tanshenghan/vln4project/LightNav-0/.venv-train/bin/python
```

已有索引拒绝覆盖；重建时先使用新的 `--output` 路径，验证后让服务读取新索引。
`--catalog` 指向主标注索引，`--sources` 是下方多源 JSON，`--video-root` 指定成片根目录，
`--renderer` 和 `--python` 指定渲染入口及其环境。旧单源 `--dataset` 与 `--sources` 互斥；`--source-id` 仅用于旧模式。
源数据/渲染样式改变时应使用新的输出版本目录。

```json
[
  {"source_id":"r2r-vlnce-v1-3","dataset":"R2R","split":"train","role":"guide",
   "root":"/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912"},
  {"source_id":"rxr-vlnce-v0","dataset":"RxR","split":"train","role":"guide",
   "shards_root":"/mnt/cfs/data/beta_nav/lerobot_rxr_train_guide_470x280_rxrnav_20260916.work/shards"}
]
```

从工作区根目录启动或断点续跑全量任务（当前任务在运行时不要重复启动）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LightNav-0/.venv-train/bin/python data_web/render_vln_video_collection.py \
  --index data_web/runtime/vln-videos-all.sqlite \
  --catalog data_web/runtime/vln.sqlite --workers 8
```

`--workers` 默认 8，范围 1–32；单个 ffmpeg 使用一个编码线程。批次文件锁禁止同时启动两个批量任务。
日志出现 `batch_finished` 后仍应检查 `failed=0` 且 `ready=records`，不能只凭进程退出判断全部成功。

网站启动必须传 `--vln-videos data_web/runtime/vln-videos-all.sqlite`。
不指定时仍兼容旧默认 `vln-videos.sqlite`，会只接入旧观测范围。
索引核对主数据库 ID、生成时间和 LeRobot 元数据指纹；主数据库重建后也要重建视频映射。
无需为了新的视频重建主标注和俯视图索引。服务环境须能读取源数据、调用指定 Python，并写入视频输出目录。

接口：`GET /vln-api/gt-video?id=<record_id>` 查询状态；`POST` 同一路径请求生成，`retry=1` 重试失败任务。
`GET /vln-api/video-progress` 返回全量覆盖、ready/rendering/pending/failed 数量。
`/vln-api/gt-video-media?id=<record_id>&kind=video|poster|report` 读取成品，MP4 支持 HTTP Range。
媒体接口只接受已索引的样本 ID 与固定资源类型，不接受任意文件路径。

查看 [Episode 515 的网站详情](http://127.0.0.1:8788/?dataset=R2R&split=train&scene=17DRP5sb8fy&q=515)。

新增划分必须先提供各自对应的 LeRobot 观测，然后更新来源配置并构建新视频索引；不能复用 train 视频顶替。
原始数据库存放与重建约定见 [data_store.md](/mnt/pfs/tanshenghan/vln4project/data_web/docs/data_store.md)。

# VLN Atlas：R2R / RxR 数据浏览器

在原 BetaData 项目中增加独立的 VLN 浏览模式。沿用 Python HTTP 服务与原生
HTML/CSS/JavaScript；数据读取、索引和网页运行只需要 Python 3.10+，无需 Habitat、
PyTorch、GPU、模型 API 或额外 Python 包。

后续新增数据的目录约定、标注与 GT 示例、数据源登记和重建流程见
[数据存放与读取说明](/mnt/pfs/tanshenghan/vln4project/data_web/docs/data_store.md)。
公网入口与重启方法见 [公网访问说明](docs/public_access.md)，全量视频数据源与批处理命令见 [LeRobot 视频说明](docs/lerobot_video.md)。

## 从机器人目录改为导航样本目录

| 原浏览器 | VLN 模式 |
|---|---|
| 数据域、任务类型、原子动作 | 数据集、版本、split、scan、语言、guide/follower |
| 任务到文件夹的关系 | 一条导航指令对应一个样本，并关联原始 episode / trajectory ID |
| 任务词条二维 embedding | 真实参考路径的俯视投影与逐节点坐标 |
| LeRobot 视频 | 470×280、8 FPS GT 视频，叠加未来 10 步及 apos/opos |
| 一次读取完整 catalog.json | SQLite 索引、服务端筛选和分页，详情按需读取 |
| 筛选与导出 | 按数据集、场景、指令等筛选，导出 JSONL manifest |

目录侧栏按数据集、split、scan 导航；语言与角色在顶部筛选。Scan 是原始场景标识，
没有将它猜测成“厨房”等房间语义。详情中的 instruction 保持原文，支持 RxR 的
英语、印地语和泰卢固语。版本与语言均保留原始粒度。

第一版以数据浏览和筛选为主。已有模型运行画面单独标注，不作为数据集参考轨迹，
也不把不同实验的评测结果聚合成数据集质量指标。

## 当前工作区直接使用

当前机器的路径已记录在被 Git 忽略的 `vln_sources.local.json`，索引位于
`runtime/vln.sqlite`。读取的是：

- `/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed`
- `/mnt/cfs/data/habitat/datasets/RxR_VLNCE_v0`
- `../InternNav/internnav/visual/episode_site/public/episodes/catalog.json` 中已有的模型回放

从包含 `data_web` 的工作区根目录运行：

```bash
# 数据或媒体报告变化后，重新生成索引。
python data_web/build_vln_catalog.py \
  --config data_web/vln_sources.local.json \
  --output data_web/runtime/vln.sqlite

# 数据或标注索引变化后，接入 GT 并生成场景俯视底图。
# 此命令使用已有 Habitat 环境的 CPU PathFinder，不需要 GPU。
python data_web/build_vln_geometry.py \
  --catalog data_web/runtime/vln.sqlite \
  --output data_web/runtime/vln-geometry.sqlite \
  --scene-root /mnt/cfs/data/habitat/scene_datasets/mp3d \
  --habitat-python /mnt/pfs/tanshenghan/vln4project/InternNav/.venv-habitat/bin/python \
  --meters-per-pixel 0.04 --workers 4

# 启动 VLN 模式，自动加载同目录下的 vln-geometry.sqlite。
python data_web/serve.py \
  --vln-db data_web/runtime/vln.sqlite \
  --vln-videos data_web/runtime/vln-videos-all.sqlite \
  --host 127.0.0.1 --port 8788
```

本机访问 `http://127.0.0.1:8788/`。当前公网入口为
[VLN Atlas](https://crucial-prot-coupled-hdtv.trycloudflare.com/)，无需访问者转发端口。
该临时 HTTPS 地址依赖服务和隧道持续运行，隧道重启会换地址。网页请求使用相对地址，也支持代理子路径。

本次索引包括 R2R 的 train、val_seen、val_unseen、test，以及 RxR 的 train、
val_seen、val_unseen 的 guide 和 follower：R2R 16,844 条，RxR 154,222 条，
共 171,066 条指令样本、90 个 Scan 场景。已匹配 71,071 条 LeRobot 观测，
其中 R2R train 10,819 条、RxR train guide 60,252 条。全量后台生成已启动，网页显示实际进度；
其余 99,995 条缺少观测，保留指令和已公开 GT，不展示替代视频。
旧 InternNav 回放保留在历史索引中，网站详情已改用 LeRobot GT 视频。
这些是本机实际标注的计数，不代表所有原始版本的数据总规模。

GT 已接入 167,658 条指令记录，覆盖本机全部 R2R/RxR train、val_seen 和 val_unseen；
全部起点匹配成功。R2R test 的 3,408 条记录没有公开 GT。90 个场景已生成 200 张
高度切片底图。首页默认全部数据；“查看全部”会清除筛选及 URL 查询参数。

## GT 与详细俯视图

原先的 `reference_path` 只是稀疏导航路标。现在单独读取同目录中的
`{split}_gt.json.gz` 或 `{split}_{role}_gt.json.gz`，导入原始 `locations` 和 `actions`。
按 source、split、role、episode ID 匹配，并检查 GT 首点与标注起点，超过 0.5 m
即拒绝关联。页面保留文件路径、GT 原始键和实际起点偏差，方便核验。

详情中可以：

- 紧凑查看左侧俯视图与右侧 GT 长度、坐标点、动作数等统计；窄屏自动上下排列。
- 查看紫色 GT 折线、起点、目标，以及可选的棕色虚线参考路标。
- 滚轮或按钮缩放、拖动平移、聚焦路线或查看完整场景。
- 手动切换高度层；播放跨层路径时自动切换，也可取消跟随。
- 逐步查看 XYZ、累计三维距离与原始动作，展开完整坐标表和动作列表。
- 下载完整 GT JSON，或下载内嵌底图的独立 SVG，包含当前高度层、轨迹和比例尺。

底图来自对应场景的 `.navmesh`，采用 Habitat X/Z 平面，X 向右、Z 向下，Y 为高度，
单位为米。像素坐标为 `[(X-origin_x)/meters_per_pixel, (Z-origin_z)/meters_per_pixel]`。
默认网格 4 cm，超大场景最长边限制为 2400 像素，实际网格精度保存在底图元数据中。
浅色表示可通行区域，灰色表示不可通行或未覆盖区域；这是带导航碰撞约束的区域轮廓，
不是照片或建筑墙线。高度层根据 NavMesh 平面面积的高度峰值推断，每层切片范围
±0.75 m，并非建筑物正式楼层标签。楼梯和夹层边界可能落在切片外，跨层路径会分段绘制。

GT 坐标通常只记录位移。动作编码为 0=STOP、1=FORWARD、2=LEFT、3=RIGHT；只有
FORWARD 数量与坐标数减一相等且动作编码已知时，才生成动作到坐标的关联。
转向和 STOP 保持位置不变；其他情况按坐标播放，保留原始动作供检查。
俯视图每 125 ms 前进一个 GT 位置；LeRobot 视频以 8 FPS 编码，每条观测对应一帧。
两者都只控制演示速度，不代表采集时间戳；评测 GT 的位置序列不一定与 LeRobot 观测帧一一对应，两个播放器保持独立。
视频位于俯视图上方，直接使用 LeRobot 记录的轨迹与标签。

GT 与底图保存在独立 `runtime/vln-geometry.sqlite`，PNG 缓存位于 `runtime/topdown/`；
原标注和主索引保持只读。构建完成后原子替换侧索引，失败会保留上一版。
主标注索引重建后，需要再次运行 GT 构建命令；服务会拒绝关联旧标注快照的 GT。
地图按 NavMesh 文件信息和分辨率缓存，重复构建可复用。服务本身仍只依赖 Python 标准库，
Habitat、NumPy、Pillow 仅用于生成底图。

自定义侧索引位置可使用 `serve.py --vln-geometry /path/to/geometry.sqlite`。
如果只需要 GT，可省略构建命令的 `--scene-root` 和 `--habitat-python`。
自定义 GT 文件命名可在 source 中设置 `gt_pattern`（支持 `{split}`、`{role}`），
此时 source 的 `root` 必须为绝对路径，并重新构建主索引。
没有 GT 的 test 样本仍可查看场景和起点，页面不会生成假设路线作为 GT。

## 接入其他路径和数据版本

复制 `vln_sources.example.json` 为本地配置，修改 `root`。相对路径以配置文件所在
目录为基准解析。每个数据版本使用独立、稳定的 `id`；不要因为目录移动而改变这个 ID。

每个 source 支持以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 数据版本的唯一标识，用于稳定样本 ID 和媒体匹配 |
| `dataset` / `version` | 页面展示的数据集名称和版本 |
| `format` | `vlnce`、`r2r` 或 `rxr` |
| `root` | 标注根目录 |
| `pattern` | 相对于根目录的文件路径模板，允许 `{split}`、`{role}` |
| `splits` / `roles` | 明确列出要导入的划分和角色 |
| `connectivity` | 可选；原生 viewpoint 格式的 connectivity JSON 目录 |

支持的标注结构：

- **VLN-CE R2R / RxR**：`{"episodes": [...]}` 的 JSON 或 JSON.GZ，读取
  `episode_id`、`trajectory_id`、`scene_id`、`instruction`、`reference_path` 等。
- **原生 R2R**：JSON 列表，读取 `scan`、`path_id`、`path`、`instructions`。
  每条 instruction 展开为一个样本，同一路线的多条描述共享 route ID。
- **原生 RxR**：JSONL / JSONL.GZ，读取 `instruction_id`、`path_id`、`scan`、
  `path`、`instruction`、`language`。该适配器经过格式夹具测试，本机全量导入验证使用的是 VLN-CE 格式。

例如原生 R2R：

```json
{
  "id": "r2r-native-v1",
  "dataset": "R2R",
  "version": "native v1",
  "format": "r2r",
  "root": "/path/to/R2R/data",
  "pattern": "R2R_{split}.json",
  "splits": ["train", "val_seen", "val_unseen"],
  "roles": ["guide"],
  "connectivity": "/path/to/connectivity"
}
```

没有 connectivity 时保留 viewpoint 序列，页面明确提示无法绘制。只要路径中有一个
viewpoint 缺失坐标，就不绘制整条路径，避免把不连续的点错误连接。

## 数据含义与缺失处理

- **指令样本数**按归一化后的 instruction 记录计数。
- **参考路线数**按 source ID、split、role、scan、原始 trajectory/path ID 联合去重。
  多语言指令可以属于同一 guide 路线；follower 轨迹独立计数。
- **参考路径长度**是原始三维参考节点间的折线长度。**标注测地距离**仅显示原文件
  提供的 `info.geodesic_distance`，不从折线长度推断。
- VLN-CE 投影使用 Habitat X/Z 平面，原生 Matterport connectivity 使用 X/Y 平面。
  未生成底图时采用等比例坐标投影。Habitat 底图不叠加到原生 Matterport 坐标上。
- 指令词数按空白分隔，是轻量统计，不等于模型 tokenizer 的 token 数。
- test 中未公开或缺少的路径、目标显示为缺失；不填充虚构的路径或距离。
- RxR 词级时间戳中的 NaN/Infinity 在索引中转为 `null`，原始文件不变。
- JSON/JSON.GZ 按 episode 流式解析，避免将完整 RxR 训练文件同时加载进内存。
- 显式配置的文件缺失、格式错误、空目录或样本 ID 冲突时构建失败，保留上次发布的索引。
  成功构建后原子替换索引。不会扫描模型、图像目录来猜测数据集。

## LeRobot GT 视频与历史媒体

当前详情的视频为 470×280、8 FPS，来自 `/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912`。
`runtime/vln-videos.sqlite` 保存精确来源映射；服务默认自动读取，亦可通过 `--vln-videos` 指定。
成片保存到数据集的 `review/gt10_apos_opos_8fps_v1/train/<scan>/episode_<id>/`，包括 MP4、PNG 和逐帧 JSON。
首次查看自动生成，完成后持久缓存；缺少 LeRobot RGB 的样本显示明确说明。
构建命令和存储约定见 [LeRobot 视频说明](docs/lerobot_video.md)。

以下 `media_catalogs` 保留历史 InternNav 报告的导入兼容，当前页面不展示模型回放：

```json
{
  "source_id": "r2r-vlnce-v1-3",
  "label": "InternNav",
  "catalog": "../InternNav/internnav/visual/episode_site/public/episodes/catalog.json",
  "root": "../InternNav/internnav/visual/episode_site/public"
}
```

只有 source ID、split、scan、episode 和去除首尾空白后的 instruction 全部匹配才会关联。
支持对采样帧、运行轨迹图和 MP4 的索引。它们不作为 LeRobot GT 视频来源。

媒体服务只接受索引生成的 asset ID，支持 HTTP Range。公开静态目录只允许网页资产，
不会通过 URL 访问任意服务器文件。该模式沿用本地/可信代理部署，不包含账号系统。

## 筛选与导出

人工筛选表单已移除，页面不再读取或写入历史 localStorage 标记。

导出范围是当前所有筛选条件的交集，包含全部匹配页。每行包含来源定位、原始指令、
导航路径字段、数据版本和筛选条件。模型回放媒体和逐词时间戳
不打包进 manifest，可从原始标注定位。密集 GT 使用详情中的“下载完整 GT JSON”单独导出。
导出时筛选条件通过 POST 发送到本地服务，逐行交给浏览器下载，不将完整清单堆积在网页内存。

## 代码与验证

- `vln_catalog.py`：流式标注读取、格式适配、媒体匹配、SQLite 索引与查询。
- `build_vln_catalog.py`：索引构建入口。
- `build_vln_geometry.py` / `vln_ground_truth.py`：GT 流式导入、匹配校验和侧索引。
- `vln_topdown.py`：CPU NavMesh 高度切片与底图缓存。
- `vln_serve.py`：分页、详情、目录统计、导出和受限媒体服务。
- `build_vln_gt_videos.py` / `vln_gt_video.py`：LeRobot 来源映射、按需渲染队列及成片缓存。
- `vln_web/`：VLN 页面、路径绘图、8 Hz GT 视频及数据筛选。
- `serve.py --vln-db`：VLN 模式；原 `--catalog` 模式继续使用原网页。

从工作区根目录运行：

```bash
python -m unittest data_web.tests.test_vln_catalog data_web.tests.test_vln_geometry
node --test data_web/tests/test_vln_ui.js data_web/tests/test_ui.js
```

覆盖多语言与压缩格式、稳定 ID、路线去重、缺失数据、原子发布、精确媒体匹配、分页与
导出、路径遍历拒绝、Range 播放、代理子路径、坐标投影、GT 错配拒绝、动作对应、跨层分段、
独立下载和过期侧索引隔离。HTTP 测试需要允许本地回环套接字。

后续可按实际需求扩展 GT 第一视角采集、语义房间标注、同一路线多语言对照，以及
按统一 run 元数据接入 SR / SPL / nDTW 等评测对比。

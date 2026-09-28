# HM3D 全量共享资产与 ObjectNav 原始数据

## 支持范围

目前新增的是 **ObjectNav 原始数据查看、类别筛选、分页、目标详情及来源导出**，
网页入口为 `/#objectnav`。它不是 R2R/RxR 指令轨迹格式，不能直接提交给现有专家
回放器；三个新版本的 `can_collect` 和 `web_collection_accepted` 保持 `false`。
不要把“数据已下载 / 能浏览”称为 GT 或 LeRobot 采集验收。

| 网页数据集 | 共享标注相对目录 | 语义资产版本 |
|---|---|---|
| MP3D ObjectNav v1 | `datasets/objectnav/mp3d/v1` | MP3D 原生语义 |
| HM3D ObjectNav v1 | `datasets/objectnav/hm3d/v1` | HM3DSem v0.1 |
| HM3D ObjectNav v2 | `datasets/objectnav/hm3d/v2` | HM3DSem v0.2 |

以上目录均相对于 `VLN_SOURCE_ROOT`（未设置时使用 `VLN_RESOURCE_ROOT`）。
本机共享根为 `/home/beta/.local/share/vln-pipeline`，因此合并代码后 mentor
可以读取相同的输入，无需复制进个人仓库。索引不会改写原始 `.json.gz`。
不对导师的 API、Python 环境、数据库或既有 R2R/RxR 数据做修改。

官方来源：

- [Habitat-Lab 数据集说明](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)
- [HM3D 官方说明](https://github.com/matterport/habitat-matterport-3dresearch)
- [Habitat-Sim 官方下载实现](https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py)

## 共享场景布局和版本隔离

```text
scene_datasets/
  mp3d/                         # 保留导师已有场景
  hm3d/
    train/                      # v0.2 RGB / navmesh；v0.2 语义资产
    val/                        # 同上
    hm3d_basis.scene_dataset_config.json
    hm3d_annotated_basis.scene_dataset_config.json
    versions/v0.1/             # ObjectNav v1 专用，绝不覆盖 v0.2
      train/                    # v0.1 原生 RGB/navmesh 和语义，不复用 v0.2 几何
      val/
      hm3d_annotated_basis.scene_dataset_config.json
installation_manifests/habitat/
  <archive>.installed.json     # 来源 URL、归档哈希和全部已安装文件哈希
  status.json                  # 完成/失败/等待的包及真实场景覆盖
```

HM3D v0.2 的公开 train / val 合计 900 个场景（800 / 100）；100 个 test 场景
未公开，不伪称下载了完整 1000 场景。语义标注只覆盖其中一部分，不要求每个 RGB
场景都有语义。ScaleVLN 读取 `.basis.glb` 和 `.basis.navmesh`，不依赖语义渲染。
版本不同的语义文件可能同名，因此必须隔离。
[官方变更记录](https://github.com/matterport/habitat-matterport-3dresearch/blob/main/CHANGELOG.md)
还明确列出 v0.2 对部分场景的几何清理，因此 v1 的 RGB 和 navmesh 也从官方旧版
`hm3d-{train,val}-habitat.tar` 下载，独立安装在 `versions/v0.1`，不能链接 v0.2。
原始 ObjectNav v2 中的
`hm3d_v0.2/...` 仅在解析本机资源路径时映射到共享 `hm3d/...`；原始 `scene_id`
始终保留。v2 的 val_mini 原始场景路径使用 `minival/`，解析资源时映射到相同
场景的 `val/`，原始划分与路径仍不变。原始 v1、v2 scene dataset config 的内容不互相覆盖。

已有 `train/`、`val/` 到 `example/` 的场景链接会保留。同名文件必须 SHA-256
一致才允许复用；不一致即失败，绝不自动覆盖。源目录存在不代表下载完成。

## 下载、安装、续跑

Matterport 凭证属于私密部署配置，禁止写入 Git、命令行参数、URL、报告或日志。
用仓库外权限 `600` 的 dotenv 文件配置 `MATTERPORT_API_TOKEN_ID` 和
`MATTERPORT_API_TOKEN_SECRET`，下面传的是文件路径，不是密钥值。
只有 Matterport 官方 API 请求携带凭证；重定向对象存储不携带凭证。
已在聊天中粘贴的密钥建议下载结束后撤销并重新生成。

```bash
.venv/bin/python data_pipeline/download_habitat_assets.py \
  --credentials /path/outside/repository/matterport.env \
  --output /large-disk/habitat-downloads --workers 3

.venv/bin/python data_pipeline/install_habitat_assets.py \
  --downloads /large-disk/habitat-downloads \
  --source-root /shared/vln-pipeline --watch

# 三个 ObjectNav 包安装后运行；仅构建小型来源/计数清单
.venv/bin/python data_pipeline/objectnav_source.py \
  --source-root /shared/vln-pipeline
```

下载器支持 `--packages` 白名单选择（默认下载全部 17 个包）、1–4 个并发下载。
`.part` 续传前检查 ETag / 字节数；已完成包检查本地 SHA-256。
本地 SHA-256 用于传输后完整性和来源追溯，不冒充官方发布的校验和。
失败保留 `.part`；恢复网络后用同一命令重跑，不能删掉部分文件或换源后直接续传。

安装器只读取完成的下载清单，再校验归档大小和 SHA-256。白名单限制资产路径、
拒绝路径穿越和归档内链接/特殊文件；在目标文件系统暂存、校验冲突后逐文件原子
发布，最后写完成清单。保留至少 5 GiB 余量。`--watch` 每 30 秒检查新完成的包，
默认 24 小时退出，可用 `--timeout-hours` 修改。没有完成的包绝不先解压 `.part`。

本机后台任务为 `vln-habitat-assets-download.service` 和
`vln-habitat-assets-install.service`（user systemd transient units），关闭页面不影响
运行；机器重启后需重新执行命令，不承诺自动跨重启恢复。下载归档保留在
`/mnt/nfs/ljqa/vln-pipeline/downloads/habitat-assets-20260928`，安装状态以共享
`installation_manifests/habitat/status.json` 为准。状态有 `failed` 或非空 `pending`
时不能宣布全量验收成功。另有 `vln-habitat-legacy-download.service` 补充 v0.1 原版
几何，写 `legacy_download_summary.json`；两次下载的完成包集合一起用于安装结束判定。
`status.json` 还会逐文件统计 ScaleVLN 场景/episode 覆盖和各 ObjectNav 划分的
几何/语义就绪数量。ScaleVLN 全量应达到 795 / 795 场景和 155,098 条路线；这只是
源资产覆盖，并不代表逐条跑过 GPU 验收。ObjectNav 的 `can_collect` 仍为 false。

```bash
journalctl --user -u vln-habitat-assets-download.service -n 10 --no-pager
journalctl --user -u vln-habitat-assets-install.service -n 10 --no-pager
```

## ObjectNav 是如何接入的

1. `install_habitat_assets.py` 把三个官方 ZIP 放到独立版本目录；保留 train、val、
   val_mini（若存在），不改名成 VLN 的 val_seen / val_unseen。val_mini 是原验证集
   的子集，统计时单列，不能相加后称为去重总量。
2. `objectnav_source.py --source-root ...` 按官方安装清单逐文件验证，然后产生
   `objectnav_source_manifest.json`：场景、类别、真实条数、原始路径及哈希。
   不把数百万个 ObjectNav 任务塞进已有 VLN SQLite，不需要重建 Atlas 几何/视频索引。
3. `/api/objectnav/datasets` 返回三版本/划分/场景与实时资产就绪情况；
   `/api/objectnav/episodes` 按一个场景与类别分页；`/api/objectnav/episode` 按
   `source_row` 读取详情。查询值由清单和路径白名单限定，不允许任意文件读取。
4. `apps/web/objectnav.js` 渲染独立 `#objectnav` 页。原始 episode ID 可能在不同
   场景重复，记录身份同时绑定版本、划分、场景、内容哈希与文件行号。
5. `goals_by_category` 通过原始场景文件名 + 类别解析。物体中心和 view_points
   分别保留；详情预览每个目标前三个观测点，显式展示实际观测点总数。

目前 HM3D 原始 `shortest_paths` 为空；MP3D 部分原始任务有动作路径，但末尾
`null` 不等于经过验证的 STOP。浏览不制造语言指令、不生成路线、不生成
FM/RVQ/apos/opos，也不修改现有 terminal_ground 的 opos 定义。

若要开放“开始采集”，下一阶段应明确选择导航网格 oracle 或模型策略，独立记录
生成轨迹来源、目标观测位姿选择、实际到达/可见性、显式 STOP 和失败状态；再完成
真实 RGB/深度/位姿、四类监督及 LeRobot 独立验收，不能用目标中心拼伪 GT。

## 回归检查

```bash
.venv/bin/python -m pytest data_pipeline/tests/test_habitat_assets.py \
  data_pipeline/tests/test_objectnav_source.py tests/test_objectnav.py -q
node --check apps/web/objectnav.js

# 真实网页只读验收：三版本、各划分 API、类别筛选、分页与详情
.venv/bin/python data_pipeline/verify_objectnav_web.py \
  --base-url http://127.0.0.1:8801 \
  --browser-executable /path/to/chrome-headless-shell \
  --output test-results/NEW_OBJECTNAV_ACCEPTANCE
```

检查覆盖：三版本隔离、原始身份/类别/行号、分页、修改后的源拒绝读取、外部符号
链接与路径穿越拒绝、重复安装不覆盖、部分场景不误报就绪，以及 API 不把 ObjectNav
提交成现有 VLN 专家回放任务。浏览验收不占 GPU、不提交生产任务。

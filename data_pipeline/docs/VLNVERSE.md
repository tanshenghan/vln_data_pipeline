# VLNVerse / Kujiale 场景与 GT 采集

## 范围与资源

接入整个 [Eyz/VLNVerse_scene](https://huggingface.co/datasets/Eyz/VLNVerse_scene) 仓库。
本次固定 revision：`dde93090c80ccf9e290a51d84a6de2bb00412d81`，共 262 个场景目录、
233,102 个仓库文件、311,767,265,693 字节。含 `_fix` 后缀场景。
场景同步目录为工作站 `/mnt/fastssd/VLN_Data_Pipeline/scene_datasets/vlnverse`；
已有的共享场景仅作为只读复用来源，文件通过哈希校验后复制到项目目录。

参考路线来自独立的 [Eyz/VLNVerse_data](https://huggingface.co/datasets/Eyz/VLNVerse_data/tree/main/raw_data/final_splits)。
训练集和验证集有 `reference_path` 和 `goals`；已核对的 coarse/fine test 文件未提供这些字段。
场景有 USD 资源不等于该场景的每条导航指令都有公开参考轨迹。
网站展示全部场景的同步状态；没有公开参考轨迹的指令自动进入 LightNav 推理采集，保留原始 test 划分，不伪造公开路线或目标。

资源拆分依据：[上游数据准备说明](https://github.com/sihaoevery/vlnverse_emr/blob/vlnverse/docs/data_preparation.md)。

## 原生采集与坐标

`collect_vlnverse.py` 在独立 Isaac Sim 5.1 / Python 3.11 环境读取原生 USD；
复用 BetaNav 的几何、pointing、RVQ 编码、LeRobot 导出和独立校验。
Habitat 与 LeRobot 的依赖环境保持独立。

自动策略为 `public_reference_else_lightnav_vlm`，manifest 采样模式为 `vlnverse_auto_v1`。其中公开路线分支：

1. 使用公开参考路线中的位置点，校验起点、终点和源起始四元数。
2. 根据相邻非重复位置点计算行走朝向，插入不超过 15° 的原地转向采样。
3. 每个采样位置和朝向都在 USD 场景中实际渲染 RGB、camera-Z 深度，并读取实际世界变换。
4. 终点与公开 goal 相差不超过 0.05 米时记录显式终止命令；输入结束本身不产生 STOP。
5. 在独立 LeRobot 环境计算 FM-30 / RVQ-10 / apos / opos，并导出自包含数据集。

这是一种明确标记的参考路线运动学回放，不是上游完整动作序列，也不模拟机器人动力学或碰撞。
源轨迹的重复位置不用于猜测缺失转向；朝向采样规则独立记录。
不与 R2R/RxR 的原有专家动作回放混为同一采样分布。

USD 世界为 Z-up、米；平台变换为 `(x,y,z) → (x,z,-y)`，保持右手坐标系。
原始起始四元数按 `wxyz` 解释，机体局部轴映射为 BetaNav 的右、上、后。
相机仍固定为 470×280、HFOV 120°、高度 0.88 米、俯仰 0°；OpenCV 相机轴为右、下、前。
深度保存 float32 米值及独立有效性 mask。所有缺失和尾部监督规则沿用当前 BetaNav 契约。

源标注记录、标注文件 SHA256、USD 主文件 SHA256、坐标转换矩阵、采样规则、采集器哈希
和 RVQ 资源版本保存在采集 manifest；原始 episode 信息随正式 LeRobot 数据一并归档。

## LightNav 分支与审核边界

无公开 `reference_path` / `goals` 的样本使用源起始位姿和自然语言指令启动闭环推理。
每条 episode 单独建立 WebSocket 会话，连接配置的 `VLN_LIGHTNAV_ENDPOINT`（本机 `ws://127.0.0.1:8051`）；
reset 仅影响当前连接，不重启服务，不干扰其他客户端。服务部署与模型身份记录在冻结任务配置中。

模型仅接收 RGB、历史观测与指令。选择预测 RVQ-10 中首个能驱动前进或转向的 waypoint，
采用 0.1 秒控制积分、每步前进上限 0.25 米和转角上限 30°，丢弃横向分量。
这沿用本仓库 LightNav 速度控制约定；采集相机采用 BetaNav 契约的 470×280。
控制积分时间不等于训练样本的物理时间戳，最终 `physical_dt_s=null`。

原生 USD 可见网格作为静态三角形碰撞体；用半径 0.18 米的多高度球扫掠和向下支撑射线限制移动。
运动细分不超过 0.04 米 / 5°，受阻停止平移，仍允许转向，不滑移、不模拟动态机器人。
物理设置、相机和缺失 HDR 解析只写匿名 session layer，不修改下载的场景资源。
缺失的同名 `limpopo_golf_course_4k.hdr` 从已有工作站资产解析，保存真实文件 SHA256；
需要在资源根目录的 `render_assets/` 部署该文件，其哈希为
`d8c4f77cd64dae23dc7cc32665b6bcbc370716d3c99e255bf9359b6225862f36`。
其他未解析资源不会用虚构内容补齐。

默认 `lightnav_max_steps=500`，API 可设 1–1000；达到上限或连续 20 次无位移/转向则保留候选。
只有服务返回的明确 model STOP 才记录 `model_stop`；文件结束、阻塞、连接失败均不当作 STOP。
保存逐步模型响应与实际机体/相机变换，FM、RVQ、apos、opos 从实际轨迹和深度重算，
不将模型输出 token 复制为 GT。`opos` 指向实际终止位置；该位置是否符合指令由语义审核单独判断。

`review_vlnverse.py` 使用已有 Ark 配置中的 `VLM_MODEL` 和凭据，默认模型为
`doubao-seed-2-0-pro-260215`。最多 32 张按时间均匀抽样的观测（含首尾帧）与指令、终止方式、
碰撞帧 ID 一并提交，保存请求哈希、逐图哈希、原始响应、模型与用量。
已保存的响应在重试时复用；网络中断且未保存响应时，供应商是否已处理无法确定，重试可能再次计费。

审核输出 `pass / reject / uncertain`、目的地与路线判断、置信度、证据帧和限制。
仅 `pass` 且明确 STOP 的轨迹可进入正式导出；其他结果保存到 `candidates/`，网站可播放、下载并人工复核。
VLM 是抽样视觉审核，不保证所有帧正确，也不能代替隐藏 test 目标的官方评测。
`trajectory_source` 区分 `public_reference` 与 `lightnav_inference`；后者不是专家演示。

## 运行入口

正常使用网站“场景库”选择已完成同步的场景，在导航指令下点击“开始采集 · 公开路线”或“开始采集 · LightNav”。
任务经过环境、轨迹、资源完整性和磁盘容量预检后，进入既有工作站队列。
依次执行 `capture → vlm_review → annotate → validate_capture → export → validate_lerobot → preview → catalog`；审核候选在 `vlm_review` 后登记原始视频、报告与采集目录，不导出为完整训练数据。
采集与原有 Habitat 任务共用 GPU 互斥锁。关闭网页不会终止任务。

安装渲染环境：`bash configs/setup_isaac.sh`。要求 Python 3.11、兼容的 NVIDIA 驱动和 Linux glibc；
依赖固定为 Isaac Sim 5.1.0，使用官方 NVIDIA Python 索引。安装完成不自动开放采集，仍需真实验收：`.venv/bin/python configs/verify_vlnverse.py`。该命令在 `kujiale_0254` 真实渲染一条验证集路线，再标注、导出、隔离 capture、独立校验和生成检查视频；全部通过才写入就绪报告。

独立命令（路径由部署配置提供）：

```bash
# config.json 是平台冻结的任务配置，包含 source_records 和 source_ids。
OMNI_KIT_ACCEPT_EULA=YES data_pipeline/.venv/isaac/bin/python data_pipeline/collect_vlnverse.py \
  --config /path/to/run/config.json --output /path/to/new/capture \
  --scene-root /path/to/vlnverse/scenes --rvq-bundle /path/to/action_tokenizer

.venv/bin/python data_pipeline/review_vlnverse.py /path/to/capture \
  --env-file /path/to/private/seedance.env

# 仅在 vlm_review_summary.json 的 accepted > 0 时执行以下标注。
PYTHONPATH=src:. data_pipeline/.venv/lerobot/bin/python data_pipeline/collect_vlnverse.py \
  --output /path/to/capture --annotate
```

`--output` 采集时必须不存在；标注阶段只接受有 `render_complete.json` 的完整渲染结果。
失败的工作站任务保留日志和产物，由平台归档本次失败输出后重试。
VLNVerse 暂按明确选择的 episode 批量采集，不支持 `full_dataset` 的 Habitat 分片入口。

LightNav 分支还需通过真实网站任务验收后执行：

```bash
.venv/bin/python configs/verify_vlnverse_lightnav.py --run-id <实际完成的-LightNav-任务-ID>
```

该命令检查逐帧真实推理响应、原生观测、VLM 响应以及候选排除行为，保存
`runtime/vlnverse_lightnav_validation.json`。网页 LightNav 按钮同时要求此报告和推理端口可达。
首次部署使用 `.venv/bin/python configs/verify_vlnverse_lightnav.py --collect --episode-id kujiale_0010_36_1 --max-steps 80`。
该命令在工作站创建真实验收任务并调用相同 Worker 阶段，完成后保存报告再对外开放按钮；
仍要求场景文件已完整校验、LightNav 服务运行且 Ark 凭据已部署。

## 全量同步与恢复

`configs/sync_vlnverse.py` 固定仓库 commit，遍历根目录及每个分页子目录，
验证 LFS SHA256 或 Git blob SHA1。不能使用仓库 info API 的 `siblings` 列表作为完整清单：
大仓库的该字段可能截断，本次实际观测到截断。

```bash
.venv/bin/python configs/sync_vlnverse.py \
  --root /path/to/vlnverse/scenes --state runtime/vlnverse_sync \
  --repo-info runtime/vlnverse_repo_info.json \
  --reuse /path/to/existing/read-only/scenes --workers 24
```

`--repo-info` 来自 HF 的仓库 revision API，只使用其中的 commit SHA；
`--reuse` 可重复指定；默认 24 个文件下载线程。
逐文件下载至 `.partial`，校验通过后原子替换。失败继续记录并重试；
重启会复核已有完整文件，保留尚未完成的部分文件。
每个完整场景写出 `platform_asset_manifest.json`。网站可读取实时状态 `runtime/vlnverse_sync/status.json`。

本机由 `vln-vlnverse-sync.service` 持续托管。查看状态：

```bash
systemctl --user status vln-vlnverse-sync
journalctl --user -u vln-vlnverse-sync -n 30
```

同步完成与采集就绪分开管理。`runtime/vlnverse_environment_validation.json` 只应在真实 USD
渲染及独立 GT 校验通过后生成，不能以 Python 可执行文件存在来代替验收。

复制部署时，可在填写上述环境变量并保存仓库 revision JSON 后执行：

```bash
.venv/bin/python configs/install_vlnverse_sync.py --repo-info runtime/vlnverse_repo_info.json
```

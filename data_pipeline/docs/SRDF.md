# SRDF 接入：公开离散路线与连续轨迹分开管理

## 当前范围

当前接入 [SRDF 作者公开的 HM3D round-3 中间版本](https://github.com/wz0919/VLN-SRDF)，
不是论文最终 SRDF-20M；作者说明最终完整数据和 MP3D 生成指令已无法恢复。
默认选择 refined round-3 文件，不下载 round-0 的 6 GB 标注、模型、预计算特征或 RGB。
本机实际校验得到 **413,999 条指令 / 413,999 条路线 / 799 个 HM3D 场景**；
2026-09-28 这 799 个场景的 mesh/navmesh 均已在共享目录可读，但不等于有连续专家动作。
固定来源如下：

| 输入 | 仓库 / 固定 revision |
| --- | --- |
| `mantis.hm3d_round3_greedy_ndtw0.9_ranked_414k_rouge0.85.json` | `ZunWang/SRDF` / `785901fab701bd2cad1f47204d56523e86a8725d` |
| `r2r_preprocess_data.zip` 中的 connectivity | `OpenGVLab/ScaleVLN` / `8ef638db28d49184e23ad3edf8aa5281fd8db7e3` |

原始 instruction ID、path ID、指令文本、起始 heading、viewpoint 顺序全部保留。
同一条路线的多条指令不合并；catalog 的 episode ID 使用原始 instruction ID，
trajectory ID 仍使用原始 path ID。图节点坐标从 ScaleVLN 官方 connectivity 读取，
保留原坐标系，**不冒充 Habitat body/camera 位姿**。

公开文件不提供连续专家动作或逐帧位姿。网页只展示指令与离散参考路线；
即使所有 HM3D 场景都安装完毕，也不会把 `can_collect` 设为 true。
没有制造 `train_gt.json.gz`、STOP、动作时间戳、稠密轨迹或 LeRobot GT。

## 共享安装与索引

遵循 `VLN_SOURCE_ROOT`（缺省 `VLN_RESOURCE_ROOT`），不写死 mentor/开发者目录：

```text
datasets/SRDF/
  raw/<原始官方 JSON 和 ZIP>
  train/train.jsonl
  connectivity/<scan>_connectivity.json
  source_manifest.json
scene_datasets/hm3d/train/<scan>/<scene>.basis.glb
scene_datasets/hm3d/train/<scan>/<scene>.basis.navmesh
```

安装入口全部位于 `data_pipeline/srdf_source.py`：

```bash
.venv/bin/python data_pipeline/srdf_source.py download \
  --directory /absolute/download/srdf-public-round3

.venv/bin/python data_pipeline/srdf_source.py prepare \
  --annotation /absolute/download/srdf-public-round3/mantis.hm3d_round3_greedy_ndtw0.9_ranked_414k_rouge0.85.json \
  --graphs /absolute/download/srdf-public-round3/r2r_preprocess_data.zip \
  --output /absolute/shared/vln-pipeline/datasets/SRDF

.venv/bin/python configs/build_source_catalog.py
.venv/bin/python -m catalog.build_vln_geometry \
  --catalog runtime/vln.sqlite --output runtime/vln-geometry.sqlite
```

下载按官方 LFS SHA-256 校验，断点续传验证 Range。转换输出必须是新目录。
只读取 ZIP 中匹配已验证场景名的 graph，绝不按 ZIP 路径解压；精简 graph 只移除
浏览不用的邻接字段，原始完整 ZIP 保留用于审计。每个路线节点必须存在，重复 ID、
非有限坐标、缺失指令等会导致转换失败；全部检查结束后才写完成标记。
索引构建再次校验转换文件哈希；坏数据不会发布完成状态。

主索引更新会改变快照。按 [配置维护](../../docs/CONFIGURATION.md) 同步更新个人
geometry/video 索引并重启自己的 Atlas/API；不要动 mentor 的端口和数据库。
已有手工视频来源必须保留，不得只用平台任务替代它们。

## 网页与 API

- 数据浏览：数据集选择 **SRDF**，可按指令、原始 ID、场景搜索，查看离散路线。
- GT 页：新增 SRDF 选项，但开始按钮和容量检查禁用，显示连续轨迹尚未接通的原因。
- `/api/health` 的 `srdf` 单独报告源数据和场景覆盖，不影响原有 R2R/RxR/ScaleVLN 就绪状态。
- `/api/sources?dataset=SRDF` 提供分页；返回 `can_collect=false`。
- API 请求校验与来源选择双重拦截 SRDF 采集，不创建队列任务；也不支持云采集、
  持续采集和 Seedance。

这次不改变现有 GT 相机、apos/opos、RVQ-10/FM-30 定义。

## 接通连续环境版所需条件

[作者发布的 SRDF-CE 400k](https://huggingface.co/datasets/SongzeLi/SRDF_CE_400k)
采用人工审批。点击申请只代表提交成功，实际文件 GET 返回 403 且提示
`awaiting a review from the repo authors` 时仍不能下载。不要用来源不明的镜像绕过审批。
此次已确认该状态，但不将个人账号授权状态写死到代码里。

批准后应先检查固定 revision `f40f341f6e7329065018590a56b5793953d1cf29` 的
`merged_dataset_400k.jsonl`（约 1.58 GB）：确认是否包含完整起点、旋转、场景、
目标、动作、动作步长与转角、STOP 和帧对齐约定。不能因同属 SRDF/StreamVLN
就直接套用其他数据的 STOP 规则。当前未读取到这个受限文件，也未编造它的解析器。

随后接入当前回放链路，至少进行一条真实 Habitat 回放、路线/终点检查、
四类监督、独立 LeRobot 读回和网页采集验收，成功后才开放“开始采集”。
无需为了验证标注而先下载约 923 GB 的预渲染图片分卷。

## 验证

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest \
  data_pipeline/tests/test_srdf_source.py tests/test_srdf.py -q
```

公开源浏览验收不创建采集任务、不调用模型或付费 API。真实可采集验收仍等待连续源审批。

```bash
.venv/bin/python data_pipeline/verify_srdf_web.py \
  --base-url http://127.0.0.1:8801 \
  --browser-executable /absolute/path/to/chrome-headless-shell \
  --output /absolute/path/to/new-evidence-directory
```

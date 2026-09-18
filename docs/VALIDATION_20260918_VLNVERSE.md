# VLNVerse 接入与工作站验收（2026-09-18）

本次按“公开路线优先，没有公开路线则 LightNav 推理，VLM 审核”实施。
网站保留 VLN Atlas 布局；新增场景入口与任务结果使用同一套样式。

## 资源范围

- 完整镜像目标：`Eyz/VLNVerse_scene` revision `dde93090c80ccf9e290a51d84a6de2bb00412d81`。
- 分页遍历得到 **262 个场景目录、233,102 个文件、311,767,265,693 字节**，包括 `_fix` 场景。
- 全部 coarse/fine train、val、val_unseen、test 指令标注已部署。test 原始文件没有公开 reference_path / goals，保留缺失状态。
- 全量场景下载由 `vln-vlnverse-sync.service` 持续托管。本文写入时仍在下载，**不是全量资源已完成的声明**；实时进度以网站场景库和 `runtime/vlnverse_sync/status.json` 为准。
- 同步目标 `/mnt/fastssd/VLN_Data_Pipeline/scene_datasets/vlnverse`；已有共享 53 场景只读复用，并非采集范围上限。

## 真实工作站结果

| 验收 | 证据 | 结果 |
| --- | --- | --- |
| 原生 USD 传感器与 BetaNav 契约 | `runtime/vlnverse_environment_validation.json`；Kujiale 0254 的 30 帧采集 | RGB、camera-Z 深度、相机/机体位姿与标签通过；移开原始 capture 后，官方 LeRobot loader、自包含检查、几何重算与训练读取器通过 |
| 网站点击公开路线采集 | 任务 `e080fe31-ae3f-49fc-a343-aad0339e79c9`；`kujiale_0254_0_0`，train/coarse | 73 帧真实采集，VLM pass，标注、导出、独立校验、检查视频、目录登记完成 |
| 网站点击 LightNav 采集 | 任务 `a7d11b7e-204a-4bb4-9b45-330b8decec51`；`kujiale_0010_36_1`，test/coarse | 62 次真实推理与 62 帧观测；位移累计约 0.597 米，遇阻后停滞结束，VLM reject |
| 候选隔离 | 上述 LightNav 任务的 `capture/candidates/`、`trajectory_review/` | 未产生虚假 STOP、公开目标或正式训练导出；视频、RGB/深度/位姿、模型响应与审核证据保留可下载 |
| 语义状态显示 | `test-results/vlnverse-results/` | 已完成任务与审核结果分开显示；候选标记未通过，不标成 GT 视频，不允许发布为正式 GT 数据集 |
| 原网站回归 | `runtime/vlnverse_browser_latest.log` | 原索引仍为 171,066 条；桌面/窄屏检查通过，无 JavaScript 页面错误 |

LightNav 验收报告为 `runtime/vlnverse_lightnav_validation.json`。该报告证明推理、观测、VLM 和候选保留链路正常，
**不表示该试跑完成了导航目标**。当前 LightNav 试跑未通过语义审核，后续数据生产须逐轨迹检查，不能以流程完成率代替成功率。

已部署模型通过独立 WebSocket 会话调用，未重启既有 LightNav Docker 服务。
模型和动作码本哈希清单位于 `runtime/lightnav_model_provenance.json`，新任务冻结其内容。
LightNav 模型权重 SHA256：`ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18`。

## 自动检查

完整 `make test`：平台 / catalog / NavGen **185 项 + 25 个 subtest**，GT **119 项**，Node **19 项**通过。
新增检查覆盖未公开路线不被伪造、test 分支选取、坐标转换、参考终点、动作边界、无证据 STOP、
VLM 证据帧、审核通过但没有 STOP 的候选隔离、已保存响应复用、候选 GT 标识以及完整分页与文件哈希。

网页实际点击及播放证据保存在 `test-results/vlnverse-web-click/`、`test-results/vlnverse-lightnav-web-click/`、
`test-results/vlnverse-results/`。运行环境、密钥、模型、原始大文件、截图和测试输出不提交 Git。

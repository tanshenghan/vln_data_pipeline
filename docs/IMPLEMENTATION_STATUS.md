# 开发部署与验收状态

更新：2026-09-18。本阶段按 [总体计划](ARCHITECTURE_PLAN.md) 完成集成和工作站部署，逐项证据见下表；资源容量与范围边界单独列出。

网站：远程 VPN 使用 **http://192.168.66.2:8800/**，工作站同一局域网内使用 **http://192.168.1.66:8800/**。远程 Mac 的网络前提及 SSH 转发见 [访问说明](CONFIGURATION.md#远程-mac-访问)。默认主页保留原 VLN Atlas 的品牌、紫色配色、五项统计、筛选区、三栏目录、视频与 GT 俯视图。顶部进入 GT 采集、Seedance、任务审核和工作站状态。

## 架构要求对应结果

| 计划要求 | 实现与验收证据 |
|---|---|
| SSD 独立项目、三个上游来源固定 | `/home/beta/hongyu.yao/VLN_Data_Pipeline`；版本记录 `contracts/upstream_versions.json`，目录按计划划分 |
| GT 数据以 BetaNav 为准 | `contracts/beta_nav_v1.json` 固定实际导出字段、类型、形状、逻辑时间、相机与当前 pointing 规则；`contracts.check` 复用原独立校验器，旧规则不能冒充当前契约 |
| 工作站预先部署，无需用户手动安装或启动 | 独立 Habitat / LeRobot / NavGen 环境、90 个场景、R2R/RxR、RVQ 就绪；PostgreSQL/Redis 和 API/Worker/Beat/Atlas 常驻并随用户服务启动，linger 已启用 |
| 点击网页开始真实采集 | R2R `d79078a8-0210-45b3-9d8e-f7ab244012a5`、RxR `e7aa54ef-e92a-48a1-9cff-796fb13ad130`，实际回放、校验、导出、检查视频与目录登记完成 |
| 源 split、角色、语言和身份保留 | R2R train/val_seen/val_unseen、RxR guide/follower 与语言筛选；网页验证集任务 `cf5f85c3-9912-4e62-baf5-e77949b7c73f` 为 val_seen Episode 326、46 帧，最终 metadata 保留 val_seen |
| 自包含正式 GT 与四类监督 | 各完成任务 `dataset/meta/platform_validation.json`；源 capture 已改名隔离，官方 loader、完整数组、几何重算、码本哈希、DataLoader 均通过 |
| 关闭浏览器、Worker 重启不重复采集 | 网页任务 `3f794702-91dc-4d80-8ef8-861191e412a6` 在 capture 中重启 Worker，最终各阶段均一次执行；子进程持有跨 Worker 生命周期的 GPU 锁 |
| 预检环境、资源与容量 | `/api/preflight` 检查服务、GPU、源 GT、场景、码本、权限和流式统计的真实帧数；网页验证全量 train 容量不足被拒绝、单条可启动 |
| 实际进度、日志、分片暂停和续跑 | `ee508edd-7399-4e80-a8c5-7417437118d2` 通过网页启动 val_seen 全量计划，只执行两个受控分片；第一次暂停完成 1 片，续跑后第二次暂停完成 2 片，第一片内容及修改时间未变；测试后取消，未发布未完成的全量数据 |
| Seedance 实际生成与已有 VGGT | `a8f8458e-286c-4013-bcfd-99315e9c25f5` 完成真实视频、现有 VGGT、质量报告与打包；网络超时后复用同一供应商任务 ID |
| 自由任务、Prompt 预览及冻结复用 | 预览 `41565bac-9d58-4ade-a112-8bb07483387e` → 网页生成 `0f035774-eeab-45fc-bd66-2f80a96ac2fa`；两次 prepared_samples 完全一致，视频只提交一次 |
| 缺失仅标注，估计轨迹不是 GT | Seedance 详情与清单明确 `ground_truth=false`；可靠深度、可信标定、机体 GT 和四类监督显示未提供；估计轨迹在独立坐标视图展示 |
| 执行状态、质量和费用分别记录 | 两个实际生成样本均运行完成但语义质量未通过，原始产物保留；详情显示真实质量报告、供应商 ID、用量和估算费用，工作站页汇总估算费用 |
| 幂等、失败诊断、恢复和取消语义 | 请求 ID 防重、成功阶段复用、未知提交禁止重提、已有视频保留；平台测试覆盖。Seedance 按钮明确停止本地执行，说明已提交云端任务未被取消 |
| 原目录、生产目录、来源关系与版本查询 | 原索引 171,066 指令、92,355 参考路线、90 场景；生产目录在 PostgreSQL 单独登记，按来源、版本、split、语言、场景与质量查询，原始记录不会被生成结果覆盖 |
| 下载与审核发布 | 原始文件、GT 流式 tar、筛选清单；审核真实浏览器刷新后持久。R2R 发布版本 `14b7ba94-cd7b-4764-b6be-81eaf7441243`，包含 34 个文件哈希；缺失正式数据或独立校验结果禁止发布 |
| 索引快照与媒体 | 已验证 GT 自动接入原 Atlas；原来源不改写，同源最新视频展示、历史运行保留。主索引和几何索引来自同一快照 |
| Codeup 管理 | 本地 main 推送至指定 VLN/VLN_Data_Pipeline；后续提交使用 姚泓屿 `<yaohongyu@betainf.com>`，不改写最初导入提交的历史作者 |

## 验证记录

- 完整 `make test`：平台 / Data / NavGen 168 项及 25 个 subtest、GT 119 项、前端 Node 19 项通过。随后新增“取消已暂停任务”和“拒绝历史 pointing 规则”的检查；最终平台与契约专项 22 项通过。原 Habitat 模型 rollout 测试未迁入；2026-09-18 新增独立原生 USD LightNav 分支及专门测试。
- 只读真实浏览器验收：原始页面、手机宽度、视频加载、预览快照、Seedance 目录与导出清单，无 JavaScript 页面错误。
- 真正的网页写操作另验证：启动 R2R/RxR、启动验证集采集、容量预检、预览后开始 Seedance、分片暂停/续跑、审核和发布。
- 本机证据：`runtime/val_collection_evidence.json`、`runtime/shard_pause_evidence.json`、`runtime/review_release_evidence.json`、`runtime/contract_r2r_final.json`；截图与运行数据均不提交 Git。

## 已明确的容量和范围边界

- 2026-09-17 验收时工作站约 304 GiB 空余（当前以网站预检为准）。全量 R2R train 约 631,244 帧，保守峰值预算约 1,977 GiB，当前配置拒绝直接启动；小批量正常可用。需要更大的输出数据盘才能在该预算下执行全量 train。未为了验收实际生产整个 train。
- 原站截图的 17,002 个历史视频未迁移；当前原 Atlas 有三个本机真实 GT 源样本，四次已完成采集版本分别保留，页面显示真实数量。
- Seedance 的两条实际结果为 `rejected_vlm`，不能把“管线成功”宣称为“样本质量通过”。费用为配置单价估算，不代表服务方账单。
- 云端取消 API 未接入，停止本地执行可能仍有云端执行和费用；页面已明确。此行为遵循计划中按服务实际能力区分本地与云端状态的要求。
- 未补全 Seedance 缺失数据，未引入新模型训练、实体机器人部署或整套上游训练依赖。VLNVerse 新增调用已有 LightNav 服务进行仿真推理。

配置、维护与恢复说明见 [CONFIGURATION.md](CONFIGURATION.md)，真实运行详见 [验证记录](VALIDATION_20260917.md)。


## VLNVerse 扩展（2026-09-18）

已部署 Isaac 5.1 独立环境和 LightNav / VLM 采集分支。有公开参考路线则回放，无公开路线则推理；
审核通过且明确 STOP 才进入正式训练导出，其余保留为候选。两种入口均完成真实网站点击验收。
全量 262 场景镜像仍在后台同步，已完成校验的场景先开放使用。

新的完整检查为平台 185 项 + 25 subtest、GT 119 项、Node 19 项通过；
具体任务、质量结果、资源范围和浏览器证据见 [VLNVerse 验收记录](VALIDATION_20260918_VLNVERSE.md)。

VLNVerse 无人工审核必选步骤：自动审核与数据校验通过即标记“GT 通过”，发布不再要求人工保留记录。
任务详情直接展示模型、置信度、路线/目标判断与证据图片；原始报告折叠状态在自动刷新后保留。

# 配置与运行维护

平台配置读取项目根目录 `.env`，字段统一以 `VLN_` 开头。该文件不进入 Git；密钥应放在仓库外权限为 `0600` 的文件中，由 `VLN_NAVGEN_ENV` 指定。

| 配置 | 当前部署 / 默认值 |
|---|---|
| `VLN_RESOURCE_ROOT` | `/home/beta/.local/share/vln-pipeline`，场景、源数据、码本与生产结果 |
| `VLN_RUNTIME_ROOT` | 项目 `runtime/`，可重建索引、服务状态和缓存 |
| `VLN_DATABASE_URL` | `postgresql+psycopg:///vln_pipeline`，本机 Unix socket / peer 身份 |
| `VLN_REDIS_URL` | `redis://127.0.0.1:6379/5` |
| `VLN_HABITAT_PYTHON` | 资源根目录 `habitat/bin/python` |
| `VLN_LEROBOT_PYTHON` | 项目 `data_pipeline/.venv/lerobot/bin/python` |
| `VLN_NAVGEN_PYTHON` | `/home/beta/.cache/navgen/venv/bin/python` |
| `VLN_NAVGEN_ENV` | `/home/beta/.local/share/vln-pipeline/secrets/seedance.env` |
| `VLN_VGGT_REPO` | `/home/beta/.cache/navgen/vggt` |
| `VLN_GPU_ID` | `0` |
| `VLN_MINIMUM_FREE_GB` | `5`，基础启动门槛；不是全量数据的容量预估 |

生成密钥文件中的 `ARK_API_KEY` 仅由生成子进程读取。模型端点与可选价格沿用 `navgen_pipeline/config.py`，实际费用与估算费用必须区分。修改环境配置后，重启 API 和 Worker 使新任务读取新配置；不要修改已提交的任务参数来改变生成定义。

## 当前部署

网站：VPN 地址 `http://192.168.66.2:8800/`；同一局域网地址 `http://192.168.1.66:8800/`；工作站本机地址 `http://127.0.0.1:8800/`。

### 远程 Mac 访问

工作站的 `eno1` 地址是 `192.168.1.66`，WireGuard `wg0` 地址是 `192.168.66.2`。远程 SSH 会话连接的是后者；远程 Mac 不应假设可以访问前者所在的局域网。

Mac 连接工作站 VPN 后打开 [http://192.168.66.2:8800/](http://192.168.66.2:8800/)。服务监听 `0.0.0.0:8800`，已在工作站验证 VPN 地址返回 HTTP 200；Mac 到该端口仍取决于客户端 VPN 路由和网络访问策略。

如果能 SSH 但网页端口无法直连，在 Mac 终端建立转发：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18800:127.0.0.1:8800 beta@192.168.66.2
```

保持这个终端连接，浏览器打开 [http://127.0.0.1:18800/](http://127.0.0.1:18800/)。如平时使用 SSH 别名、自定义端口或跳板，将命令末尾的目标替换成已有连接配置。转发仅绑定 Mac 回环地址，网站和采集仍在工作站运行。

### 服务管理

API、Worker、调度器及 Atlas 已作为 beta 用户的 systemd 服务运行，启用 linger；PostgreSQL 和 Redis 为本机系统服务。网站采用原 VLN Atlas 的样式和数据浏览页面。新增功能通过顶部入口进入。

```bash
systemctl --user status vln-api vln-worker vln-beat vln-atlas
journalctl --user -u vln-worker -n 100
```

重新安装服务定义使用项目 `.venv/bin/python configs/install_services.py`。这不会安装场景资源或 Python 依赖，不能替代首次环境准备。API 依赖固定在 `apps/api/requirements.txt`；Habitat、LeRobot、NavGen 必须维持独立环境。

GT 导出环境为 Python 3.11、`lerobot==0.4.4`、`numpy==1.26.4`；CPU Torch 用于导出，GPU Habitat 用于采集。不能将导出环境直接升级到 NumPy 2.4，当前上游 datasets 的数组标量转换与其不兼容。

源索引配置保存于 `configs/sources.local.json`，不提交机器相关路径。主索引和几何索引必须使用同一快照。已验证的 GT 数据由 `workers/catalog_sync.py` 自动登记到 Atlas 视频集合；只登记产物，不复制或修改源数据。重采同一源样本时，Atlas 使用最近的已验证版本，历史任务仍保留自己的产物。

## 任务恢复

每次运行在资源根目录 `runs/<run-id>/` 保留配置、各阶段命令、进程身份、日志、退出记录与输出。数据库记录任务、审核和版本。

- 关闭或刷新网站不会停止已派发任务。
- Worker 重启后以进程 PID 和启动时间核对已运行的阶段；已有成功退出记录可恢复，避免再次执行。
- 全量 GT 暂停会停止派发新分片，当前已运行分片完成后进入 paused；续跑复用完成标记。小批量和已开始的合并在当前阶段完成后暂停。
- Seedance 持久化提交意图及供应商任务 ID；已获得 ID 的任务续跑复用原 ID。提交结果不明时禁止自动重复提交，须核对云端记录。
- 取消本地任务不等于取消已提交的云端生成；当前尚未接入供应商取消 API。
- 失败保留日志和已生成原始文件；下游 VGGT 或质量分析失败不会删除原视频。

## GT 原始参数文档

- [GT 数据格式](../data_pipeline/docs/LEROBOT_FORMAT.md)
- [GT 管线说明](../data_pipeline/README.md)
- [总体架构与验收要求](ARCHITECTURE_PLAN.md)

网页新增参数应映射到既有 GT 入口，不能改变坐标、时间、四类监督或各自 mask 的含义。跨环境校验只允许投影浮点值 `1e-9` 的绝对舍入误差，离散 token、状态和 mask 仍严格比较。

## Prompt 预览与产物目录

Seedance 页面可先创建 Prompt 预览任务，查看实际编译后的提示词。自由任务会调用规划模型，产生文本用量；此时不提交视频。点击“使用此 Prompt 配置生成”返回生成表单，再点击开始生成，后端复用预览快照。数量、时长、源样本或任务说明变化后，必须重新预览；任务名称可修改。预览不属于生产数据，不能发布数据版本。

规划模型的返回文本和用量缓存于各运行 `generation/planner_responses/`，解析失败后复用已有回复，不为相同输入重复请求。原始观测语义空字符串按缺失值处理，不生成置信度；已观察到的 `navigation` 类型别名映射为既有 `object_goal`。

Atlas 的“更多筛选 → 来源类型”可选择原始指令、全部生产产物、GT 或 Seedance。产物存于 PostgreSQL `data_samples`，与只读原目录分离。每个运行/样本有独立 ID，数据版本对应其运行 ID；同一源指令的多次采集或生成不会互相覆盖。源指令详情底部列出派生产物。原版首页的默认布局与原始数据查询保持一致。

数据集下载支持 `/api/artifacts/<id>?archive=tar`，流式打包，不先复制整个数据集；普通文件使用 `?download=true`。目录中的单个文件用 `?file=<relative-path>` 读取，服务器校验解析后的路径必须属于该数据集。

只读浏览器验收脚本（不会提交生成或采集任务）：

```bash
.venv/bin/python tests/browser_smoke.py \
  --run 0f035774-eeab-45fc-bd66-2f80a96ac2fa \
  --preview 41565bac-9d58-4ade-a112-8bb07483387e
```

可通过 `--url`、`--chromium` 和 `--output` 指定网站、浏览器及证据目录。测试依赖见 `tests/requirements.txt`，生成截图与 JSON 证据存放于被 Git 忽略的 `test-results/`。


## 容量、契约与检查

GT 页面可先“检查采集范围与容量”，点击开始时后端仍自动执行相同检查。检查公开 GT、所选语言和 ID、RxR 转向展开后的实际帧数、所有涉及场景、码本、服务运行状态、输出权限与空间。API 为 `POST /api/preflight`，成功返回 episode/帧/场景数量和保守峰值预算。全量 R2R train 的预算约 1,977 GiB，当前约 305 GiB 的 SSD 余量不足，需更大的数据盘；默认一条采集可直接运行。预算含预留空间，不代表压缩后最终大小。

[契约检查入口](../contracts/README.md) 复用官方独立校验。发布拒绝缺失产物或缺失 GT 校验报告的任务。历史上游参数见 [分片控制参数](../data_pipeline/docs/FULL_R2R_TRAIN.md#平台控制与划分2026-09-17)。

`make test` 分别在平台与 LeRobot 环境运行测试，并检查前端。`test_rollout_lightnav.py` 属于未迁入的模型推理，不计入采集管线测试；不会安装整套训练/推理栈以运行它。只读网页验收仍使用上面的 browser_smoke 命令。

## VLNVerse / Kujiale

完整场景同步、独立 Isaac 环境和采集参数见 [VLNVerse 配置](../data_pipeline/docs/VLNVERSE.md)。
`VLN_VLNVERSE_MANAGED_SCENE_ROOT` 指定经过校验的全量场景目录；`VLN_VLNVERSE_SCENE_ROOT` 可指定已有共享场景的只读目录；`VLN_ISAAC_PYTHON` 指定独立渲染解释器。

VLNVerse 自动采集采用公开路线优先、LightNav 回退、VLM 审核策略。`VLN_LIGHTNAV_ENDPOINT` 指向已有推理服务，
`VLN_LIGHTNAV_MODEL_PROVENANCE` 记录服务模型身份。API 的 `lightnav_max_steps` 默认 500、范围 1–1000；
VLM 复用 `VLN_NAVGEN_ENV` 内的 `ARK_API_KEY`、`ARK_BASE_URL`、`VLM_MODEL`，凭据不写入任务配置或 Git。
`VLN_LIGHTNAV_MODEL_MANIFEST` 可指定模型与动作码本 SHA256 清单，平台将其内容冻结到任务配置，优先于文字版本说明。

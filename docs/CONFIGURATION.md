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

网站：`http://192.168.1.66:8800/`，本机也可用 `http://127.0.0.1:8800/`。

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
- 暂停在当前阶段完成后生效；全量 GT 当前阶段为分片批处理，细化到停止派发分片尚待实现。
- Seedance 持久化提交意图及供应商任务 ID；已获得 ID 的任务续跑复用原 ID。提交结果不明时禁止自动重复提交，须核对云端记录。
- 取消本地任务不等于取消已提交的云端生成；当前尚未接入供应商取消 API。
- 失败保留日志和已生成原始文件；下游 VGGT 或质量分析失败不会删除原视频。

## GT 原始参数文档

- [GT 数据格式](../data_pipeline/docs/LEROBOT_FORMAT.md)
- [GT 管线说明](../data_pipeline/README.md)
- [总体架构与验收要求](ARCHITECTURE_PLAN.md)

网页新增参数应映射到既有 GT 入口，不能改变坐标、时间、四类监督或各自 mask 的含义。跨环境校验只允许投影浮点值 `1e-9` 的绝对舍入误差，离散 token、状态和 mask 仍严格比较。

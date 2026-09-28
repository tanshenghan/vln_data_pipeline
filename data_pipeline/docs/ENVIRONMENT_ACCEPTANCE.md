# R2R / RxR 工作站实测验收

使用平台 `.venv` 从仓库根目录运行，读取当前部署的 `.env`：

```bash
.venv/bin/python data_pipeline/verify_environment.py
# 可选：指定一条 RxR 专家样本
.venv/bin/python data_pipeline/verify_environment.py --dataset rxr --episode-id EPISODE_ID
```

| 参数 | 默认值 / 含义 |
|---|---|
| `--dataset` | `r2r`；也支持 `rxr` |
| `--split` | `train`；也支持 `val_seen`、`val_unseen` |
| `--role` | `guide`；RxR 还支持 `follower` |
| `--languages` | `all`；RxR 可选一个或多个 `en-US`、`en-IN`、`hi-IN`、`te-IN` |
| `--episode-id` | 可选；不指定时从本地源索引选择第一条匹配样本 |

先准备个人源索引与配置：`VLN_SOURCE_ROOT` 指向只读数据集和场景，
`VLN_RESOURCE_ROOT` 指向个人输出目录；`VLN_HABITAT_PYTHON` 和
`VLN_LEROBOT_PYTHON` 分别指向独立的已有环境，不能向共享环境安装依赖。
`VLN_RVQ_BUNDLE_ROOT` 可指向另一处可读的 LightNav `action_tokenizer`；
未设置时仍使用源根目录内的码本。Worker 禁止生成共享环境的 Python 字节码缓存。

验收会实际使用 GPU 渲染一条公开专家轨迹，校验原始采集，导出 LeRobot，
移走原始 capture 路径后用官方加载器和完整 GT 契约独立读回，再生成审核视频。
只有全部成功才将 `VLN_RUNTIME_ROOT/environment_validation.json` 中
`habitat`、`lerobot`、`complete` 设为 `true`。导入成功不算验收通过。
重新验收期间就绪标记为 false；失败保留阶段和错误，不能手工伪造通过标记。

每次写入新的 `VLN_RESOURCE_ROOT/validation/<dataset>-<id>/`，保存参数快照、
阶段命令与日志、原始采集归档、独立数据集、码本及来源哈希和审核视频；不改写源资源。
本命令不创建网页任务，也不证明 Celery 队列正常。通过后启动自己的 Worker、Beat、Atlas，
重启读取了新 `.env` 的 API，在网页中以数量 1 创建任务，验证完整队列链路。

本验收不代表 VLNVerse 的 Isaac / LightNav / VLM 链路已通过；它有独立验收入口。

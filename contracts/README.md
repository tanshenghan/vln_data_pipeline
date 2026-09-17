# BetaNav 契约

`upstream_versions.json` 固定迁入源码版本。`beta_nav_v1.json` 是该版本实际导出器在工作站默认 470×280 配置下的字段、dtype、shape 和时间定义快照，不是另一种导出格式。

正式入口使用独立 LeRobot 环境：

```bash
PYTHONPATH=src:. HF_HUB_OFFLINE=1 data_pipeline/.venv/lerobot/bin/python \
  -m contracts.check /path/to/dataset --report /path/to/new-report.json
```

报告路径必须不存在。先核对版本与字段，再调用原 `data_pipeline.validate_lerobot`：官方读取器、完整数组、独立 mask、几何重算、观测与码本哈希，以及无需源采集目录的 DataLoader 读取。平台新 GT 任务自动使用此入口。发布还要求正式数据集、成功的独立校验报告与“保留”审核。

历史定义可继续使用原只读校验器和浏览器检查；不能因为旧数据能读取，就将它发布为当前契约。Seedance 仅登记原始产物、质量与缺失说明，不经过完整 GT 契约，不生成填充监督。

契约语义的唯一详细定义见 [LeRobot 格式](../data_pipeline/docs/LEROBOT_FORMAT.md)。相机、时间或监督定义变化应建立新契约版本，保留旧快照。

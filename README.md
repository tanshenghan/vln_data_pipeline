# VLN Data Pipeline

工作站上的 VLN 数据生产与浏览平台。网站保留原 VLN Atlas 的布局与样式，在顶部接入 GT 采集、Seedance 生成、任务与审核和工作站状态。

GT 格式沿用 BetaNav，网页提交后在工作站真实运行 Habitat 采集、标注、LeRobot 导出与独立校验。Seedance 保留原始视频和现有 VGGT 估计轨迹；深度、可信标定和监督信息缺失时仅标注，不补全或伪造 GT。

远程 Mac 经工作站 VPN 访问：[http://192.168.66.2:8800/](http://192.168.66.2:8800/)。工作站同一局域网内使用 [http://192.168.1.66:8800/](http://192.168.1.66:8800/)。两种地址的网络前提和 SSH 转发方法见[访问说明](docs/CONFIGURATION.md#远程-mac-访问)。

R2R/RxR 小样本真实采集已通过，Seedance 实际生成、VGGT 和原始产物展示已验证，分片暂停恢复、容量预检、契约检查及审核发布已验证。全量 train 需要更大的数据盘，详见实施状态中的容量边界。

- [架构与验收计划](docs/ARCHITECTURE_PLAN.md)
- [实施与验证状态](docs/IMPLEMENTATION_STATUS.md)
- [配置及服务维护](docs/CONFIGURATION.md)
- [上游固定版本](contracts/upstream_versions.json)

生产资源位于仓库外，机器配置、密钥、运行结果和环境不提交 Git。

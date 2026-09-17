# 2026-09-17 工作站与网站验证

## GT 真实执行

R2R 与 RxR 均使用工作站 RTX 5090、资源目录中的 Habitat 环境和项目内 LeRobot 导出环境。任务 ID、实际结果和当前未完成项列于 `IMPLEMENTATION_STATUS.md`。最终报告在各任务 `dataset/meta/platform_validation.json` 中，界面可下载。

跨环境校验发现 Python/NumPy 版本不同导致小于 `1e-9` 的像素投影舍入差；只为连续浮点量设置绝对容差，token、mask、状态及字典字段仍精确检查。另将 LeRobot 环境的 NumPy 固定为 1.26.4，解决上游 datasets 的数组标量转换兼容性。

## 网站恢复

以迁入的 `catalog/vln_web/index.html`、`styles.css` 与 `app.js` 为基准。首页保留原 DOM 布局，直接加载原站样式，新增页面样式限制在平台区域；不再通过另一个侧栏页面嵌套 Atlas。

浏览器验证包含原站五项统计、筛选、样本详情、来源选择、GT/Seedance/任务入口、地图、视频实际加载与手机宽度。R2R Episode 10154 视频为 50 帧、8 Hz 检查播放，训练逻辑时间仍按 BetaNav 契约保持 1 fps。

当前视频索引关联两个本机已验证的 GT 样本。原站截图中的历史媒体数量尚未迁入，页面按真实索引展示，不填充截图中的数字。

## 密钥与供应商任务

密钥文件在仓库外，目录权限 0700、文件权限 0600。请求日志和数据库只保存运行配置、配置文件路径及供应商任务 ID，不保存密钥值。已验证鉴权及真实视频任务提交；后续网络超时属于状态查询失败，不能据此再次创建云端任务。

恢复机制测试覆盖：重复请求、Worker 确认前退出、失败任务重复投递、供应商任务 ID 复用、提交结果不明禁止重提，以及已提交任务的临时查询失败只重试读取。下游失败仍登记已存在的原视频及报告。

## 增补验收：预览、统一目录与恢复

- 原版网页发起 R2R 任务 `3f794702-91dc-4d80-8ef8-861191e412a6` 后关闭浏览器，并在 capture 阶段重启 Worker。最终 capture、validate_capture、export、validate_lerobot、preview 均完成且各仅有一个 attempt；未重复采集。
- 自由任务先通过网页创建预览 `41565bac-9d58-4ade-a112-8bb07483387e`，再使用同一预览在网页启动生成 `0f035774-eeab-45fc-bd66-2f80a96ac2fa`。两个 `prepared_samples.json` 内容完全一致，生成侧只有一个供应商任务提交记录。任务完成后，视频、质量、费用、独立估计轨迹和产物目录均经真实浏览器检查。
- 自由规划首次暴露空观测语义和任务类型别名的上游兼容问题。现在空观测字段保持 null，navigation 使用已有 object_goal 语义；相同规划回复可缓存复用，不通过伪造观测语义或换用固定模板规避错误。
- 产物目录浏览器筛选和下载清单覆盖两条 Seedance 结果；均明确 `ground_truth=false`，来源为空的自由任务归类为 Seedance/generated。其页面不显示 LeRobot GT 的说明或 GT 视频整理进度。
- 数据集 tar 下载验证包含实际文件内容，越界文件访问返回 403。

新增截图保存在本机 `runtime/prompt_preview.png`、`runtime/seedance_from_preview_completed.png` 和 `runtime/catalog_seedance.png`。这仍不是总体计划所有条目的最终验收。

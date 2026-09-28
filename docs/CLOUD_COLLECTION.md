# 云端空闲 GT 采集

网页的「采集方式」显示 **GPU0 / PRO6000 / B200 / H20**。持续采集可以同时勾选四者，GPU0 由本机 Celery 执行，云端由独立 `vln-cloud` 服务领取不同批次，使用闲时资源并自动让位。自定义数量仍选择单台设备。选择 R2R / RxR 和「持续采集」后点击「启用持续采集」。火山两个 H20 队列统一显示为一个 **H20** 选项；持续采集展开为已启用的 H20 队列，各自领取不同批次，自定义数量则自动选择就绪的 H20 队列。单批次 API 仍支持 `compute_pool=auto` 自动选择就绪资源池。已配置但尚未通过环境验证的设备可以保存到持续计划中，验证通过后自动加入；验证前不派发生产批次。本机是否有空闲 GPU 不影响云任务预检。

## 持续采集

1. 选择数据集、划分、角色和语言，勾选 GPU0、PRO6000、B200 中的一种或多种，采集数量选择「持续采集」，点击「启用持续采集」。默认覆盖该范围内所有未完成轨迹；指定 episode / 源样本时只采集指定范围。持续模式不受自定义数量限制，每批路线数由表单控制（默认 8）。
2. 每个已就绪设备空闲时领取不同批次，批次成功并登记校验产物后补充下一批。源样本领取记录和任务在同一数据库事务中保存，跨设备、重启和重叠计划不会重复领取；已有正式 GT 校验成功的样本也会跳过。已派发批次固定在原设备续跑。
3. 计划保存在数据库中，关闭浏览器或重启调度器后仍然有效。当前范围采完后进入「本轮采完，等待新增」，继续检查来源目录中相同筛选范围的新增轨迹，不反复重采已完成样本。新增场景/源文件仍需部署到云端并通过源文件哈希核对。
4. 页面底部「持续采集计划」显示进度和最近批次，可暂停、继续或停止整个计划。暂停后停止补充批次并释放已在采集的 GPU；继续沿用断点。单独手动暂停的批次不会被计划的继续按钮悄悄恢复。停止保留已有产物，确认云任务停止后释放未完成样本的领取记录。
5. 前台任务排队、监控失联、镜像/数据未验证时不派发；磁盘空间按每批复核，不够时等待。普通执行失败达到原有重试上限后，计划显示「需要检查」，停止补充，处理后手动重试。

`POST /api/collection-plans` 接收 GT 配置及 `compute_pools` 数组，要求 `Idempotency-Key`；本机、云端、混合选择分别对应 `execution=local/cloud_idle/mixed`。`GET /api/collection-plans` 查看计划；`POST /api/collection-plans/preflight` 检查范围；`POST /api/collection-plans/{id}/{pause|resume|retry|cancel}` 控制计划。云端和混合计划支持 R2R / RxR；ScaleVLN 当前仅本机可采集，云端 HM3D 和源数据仍需另行部署验收。原有 `/api/runs` 单批次接口保留，不接受 `execution=mixed`。

## 调度行为

- 火山调用 ML OpenAPI `2024-07-01`，强制 `ResourceConfig.Preemptible=true`、`Priority=1`，关闭资源保留和平台自动创建后继重试任务。
- 百度调用 AIHC v2，强制 `idleTask=true`，不传 `priority`（平台自动固定为 `lowest`），关闭额外容错重试。需要**全托管资源池**；旧版自运维资源池不静默降级成普通任务。
- 默认每轮检查后等待 15 秒再检查所有配置的队列，实际周期还包含 API 查询耗时。其他用户的非闲时任务处于排队/启动状态时，本平台的云采集停止，原生闲时调度也可以直接回收资源。百度同时读取训练任务、所有用户的开发机和队列级等待数量；火山同时读取训练任务、开发机和在线服务，包含在线服务等待扩容的实例。查询不按创建者过滤，分页不完整或无权读取时停止派发。本采集器自己的闲时排队不会触发自我让位。已有前台任务正常运行且无任务排队时，原生调度器决定剩余资源能否分配。判断依据是调度状态，不能把已分配但 GPU 利用率低的卡当作空闲。
- 连续观察空闲 120 秒后启动；让位后冷却 120 秒。默认每个资源池最多一个单实例采集任务；通过 `max_active_jobs` 控制同时执行的任务数量。单个任务固定使用模板申请的 GPU，当前执行器使用容器内 GPU 0。不要配置多副本或多卡规格，额外的卡不会自动分工。`workers` 是现有单 GPU 上的分片并行数。
- 手动暂停/取消与资源让位分开记录。手动暂停后必须手动续跑；自动让位会恢复。只有云端确认旧任务进入终态，才允许创建下一次任务。
- 监控、鉴权、分页或未知状态异常时停止派发。云容器必须持续看到控制器续写的租约；默认租约 90 秒，过期或要求停止后最多等待 15 秒，再终止整个采集进程组（TERM 后最多 3 秒 KILL），释放 GPU。云平台自己的回收可以更快。
- `CreateJob` 超时或控制器在提交后重启时，按已持久化的唯一任务名找回原任务，不重复提交。长时间找不到时会显示「云提交结果待核对」。**不要删除绑定来重试**：先在供应商控制台/云审计确认提交结果。当前无“强行重发”按钮。
- 已完成的阶段与完整分片保留。被强杀的当前分片/当前未完成阶段重做，不承诺帧级恢复。普通运行失败最多重试 3 次；明确抢占不消耗此次数。
- 单个批次首次分配后固定在该资源池恢复，不将断点复制到另一朵云。持续计划自动把未完成轨迹拆成互不重叠的批次分给所选设备；原有 `full_dataset` 单任务仍在一个资源池完成。

控制台的「空闲资源出借／闲时出借」与任务的「使用闲时资源」是两个不同开关。火山明确允许在本队列有足够空闲配额时提交闲时任务，无需其他队列开启出借；百度也支持优先使用本队列空闲资源，再使用其他队列已出借的资源。因此，不要仅凭「出借未开启」就修改队列的共享策略。截图中的未分配 GPU 数量也不能替代实时排队检查：即使尚有未分配 GPU，只要监控范围内有非闲时任务排队，采集器仍会让位。

## 上线所需信息

| 项目 | 火山引擎 | 百度百舸 |
| --- | --- | --- |
| 凭证 | AccessKey ID / SecretAccessKey，可选 STS token | AccessKey ID / SecretAccessKey，可选 STS token |
| 位置 | Region、API endpoint、ProjectName、队列 ID | Region、API endpoint、全托管队列 ID；API `resourcePoolId=aihc-serverless` |
| 监控范围 | `monitor_queue_ids` 列出需要让位的队列 | 同左；PRO6000、B200 分别配置 |
| 运行规格 | 单 GPU 规格 ID、采集镜像 | 单 GPU 的资源描述符、采集镜像 |
| 存储 | 挂载的 vePFS/CFS 等持久化 POSIX 盘 | 挂载的 PFS/CFS 等持久化 POSIX 盘 |

这里需要云平台控制面的 **AK/SK**，不是方舟或千帆模型推理 API Key。子账号需要查询监控范围内**所有用户任务**、创建闲时任务以及停止自己采集任务的权限；只能列出自己的任务的凭证不足以实现主动让位。无需给采集容器云管理密钥，也无需向云端开放 PostgreSQL/Redis。应用调用停止接口时只使用数据库中记录、且任务名核对一致的采集任务 ID。

注意区分「地域」「可用区」「资源池 ID」「队列 ID」：火山的「可用区 C」不能推断为北京，签名所用 Region 必须从控制台顶部或详情页网址确认。百度资源池详情中的 `aihc-...` 是资源池 ID；队列详情顶部也使用 `aihc-...` 格式，但这是另一类 ID。不要用资源池 ID 填 `queue_id`。百度全托管训练 API 的 `resourcePoolId=aihc-serverless` 是接口规定值，与控制台展示的实际资源池 ID 分开记录。截图识别到的部署信息可保存于被 Git 忽略的 `configs/cloud_inventory.local.json`，在得到凭证后以只读 API 查询结果复核。

百度 API 的资源池下还区分 `Physical` 物理队列和 `Elastic` 弹性任务队列，两者可能同名。`queue_id` 应填写 `Elastic` 子队列的实际 API ID（实测为 `aihcq-...`），不能只按截图推断前缀。`DescribeQueue` 即使传入子队列 ID，也可能返回含 `children` 的物理父队列；控制器按 ID 精确定位子队列。`cprom-...` 是监控实例 ID，不能用于提交任务。资源/开发机查询使用 GET 与 `version: v2`，训练 API 使用 POST 与 `X-API-Version: v2`。

### 1. 配置密钥和资源池

凭证放在仓库外 `~/.config/vln-pipeline/cloud.env`，权限设为 `600`：

```dotenv
VOLCENGINE_ACCESS_KEY_ID=...
VOLCENGINE_SECRET_ACCESS_KEY=...
BAIDU_ACCESS_KEY_ID=...
BAIDU_SECRET_ACCESS_KEY=...
# 可选：VOLCENGINE_SESSION_TOKEN / BAIDU_SESSION_TOKEN
```

```bash
cp configs/cloud_pools.example.json configs/cloud_pools.local.json
cp configs/cloud/volcengine-job.example.json configs/cloud/volcengine.local.json
cp configs/cloud/baidu-job.example.json configs/cloud/baidu-pro6000.local.json
cp configs/cloud/baidu-job.example.json configs/cloud/baidu-b200.local.json
```

填写资源 ID、区域、镜像、路径；实际 JSON 和密钥文件被 Git 忽略。模板中的 `REPLACE_...` 必须全部替换。优先从该队列一个验证成功的单实例任务导出创建请求，保留真实资源类型、镜像与存储配置。PRO6000/B200 的资源描述符从你的队列中读取，不根据显示名称猜测。控制器会覆盖名称、启动命令、闲时策略和重试策略。

使用百度 CCR 企业版私有镜像时，可在资源池中填写 `baidu_registry_id`（`ccr-...`）和 `baidu_registry_username`。控制器在每次提交前用管理 AK/SK 申请有效期一小时的镜像拉取密码，仅加入本次云 API 请求，不写入任务配置、断点或模板；抢占后再次启动会获取新密码。镜像域名必须与配置的 CCR 实例及地域一致。只读 `--check` 不申请密码。账号需允许该 CCR 实例的临时凭据接口。

### 2. 准备镜像与共享数据

仓库提供 R2R/RxR 镜像配方，读取当前配置的 Habitat/LeRobot 环境，不向共享环境安装依赖：

```bash
# conda-pack 只安装到独立构建工具目录
.venv/bin/python -m pip install --target runtime/cloud-image-tools conda-pack==0.8.1
PYTHONPATH=runtime/cloud-image-tools:. .venv/bin/python configs/prepare_cloud_image.py --output runtime/cloud-image
docker build --build-arg RUN_UID="$(id -u)" --build-arg RUN_GID="$(id -g)" \
  -t vln-collector:r2r-rxr-v1 runtime/cloud-image
```

若无法连接 Docker Hub，可用 Ubuntu 官方 ECR 镜像：构建时加 `--build-arg BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:22.04`。生产部署固定镜像 digest。快照包含实际安装的 Python 环境、源码和 SHA-256 清单，不包含源场景、产物、`.env` 或 `data_pipeline/config.py`。需要平台 Python 3.10、LeRobot Python 3.11；构建会执行无 GPU 的导入检查。修改代码后可用 `--refresh-code` 更新同一构建目录，再重新构建。镜像构建成功后，还需推送至云任务可访问的私有镜像仓库。

镜像在固定路径包含本仓库源码和平台、Habitat、LeRobot 的独立 Python 环境（版本沿用当前项目），以及 ffmpeg、图形驱动所需用户态库。用 `worker_env` 配置解释器和只读 `VLN_SOURCE_ROOT`。镜像必须固定版本，并分别在目标 GPU 上执行真实采集验证；不能仅凭训练任务能运行就假设 Habitat/EGL 或 Isaac 渲染兼容。

默认没有任何 `validated_datasets`。验证 R2R/RxR 后填入相应列表；VLNVerse 还需 Isaac、完整 USD 资源、HDR、可达的 LightNav（如需推理）、VLM 凭证文件和对应验证，完成后才能添加 `vlnverse`。VLM 凭证用受控文件挂载，通过 `VLN_NAVGEN_ENV` 引用，勿写进启动命令。

本实现采用**共享 POSIX 存储传递配置、控制、断点和产物**：

```text
工作站 local_storage_root  ─── 同一个持久化目录 ─── 云容器 remote_storage_root
├── .vln-volume-id
└── runs/<run-id>/
    ├── config.json / cloud_control.json
    ├── stages/                  每阶段日志和成功标记
    ├── dataset.work/shards/     已完成分片
    ├── dataset/                最终 LeRobot 数据集
    └── cloud_result.json        原子提交的产物清单
```

需要提前通过专线/VPN 等使工作站挂载该云存储，或使用下面的 SSHFS 桥接。双方必须看到**同一份实时文件**，支持原子 rename 和一致的读写权限；云容器使用的原生 PFS 必须支持跨容器文件锁。镜像按持久化盘写入者的数字 UID/GID 构建并以该用户运行；默认与控制器一致，普通账号桥接时则与远端登录账号一致。云任务配置不能强制改成 root，否则原子写入的私有控制/结果文件无法互读。源目录也需允许该 UID/GID 只读访问。单独在两边建同名目录或异步 rsync 镜像不满足要求；对象存储 FUSE 不适合作为本协议的控制/锁目录。当前没有内置跨云数据上传下载代理；若无法共享挂载，需先接通存储或增加独立传输后端。

在真实挂载目录创建 `.vln-volume-id`，内容与池配置的 `storage_id` 相同；云容器也会核对卷标和任务配置。不要在挂载断开时自动创建卷标，避免把工作站本地盘误认为云盘。源数据以只读方式单独挂载；R2R/RxR 启动前会核对源 episode/GT 文件 SHA-256，防止两端使用不同版本。

如果工作站无法直连 PFS，可以通过已挂载 PFS 的开发机建立 SSHFS 桥接。`configs/install_cloud_storage.py` 安装独立 `vln-cloud-storage` 用户服务，关闭文件及目录缓存，使用同步写入和 OpenSSH 的原子替换。开发机必须保持运行；连接丢失时控制器停止续租，云端按租约退出。此方式不提供高可用网关，产物传输速度也受 SSH 链路限制。

```bash
cp configs/cloud_storage.example.json configs/cloud_storage.local.json
# 填写已有 SSH 地址、专用密钥路径、已核对的 known_hosts、共享目录和数字 UID/GID。
# 远端需已有对应数字用户（无需登录权限）、可执行 setpriv 和 sftp-server。
# mountpoint 必须提前创建且属于当前用户；远端目录和卷标需在真实 PFS 上创建。
.venv/bin/python configs/install_cloud_storage.py --config configs/cloud_storage.local.json
```

可通过 `--sshfs-binary /absolute/path/to/sshfs` 安装自行从发行版软件包解出的客户端，无需改动系统 Python。服务不会创建远端账户、目录、卷标或修改其他 SSH 密钥。启用前必须实测双方原子替换后立即可读，且私有文件的 UID 与镜像一致。**SSHFS 不承担跨机器锁**：控制器在本机 runtime 目录取单实例锁，云容器彼此通过原生 PFS 的 `flock` 互斥；禁止在桥接端手工运行 `cloud_entry`，也不要部署第二个控制器写同一组断点。云任务通过 `pfsl2`（以实际集群返回类型为准）直接挂载 PFS 子目录，不经过 SSHFS 渲染或导出。

接入第二朵云时使用独立配置、挂载点和服务名，例如 `--config configs/cloud_storage.volc.local.json --service-name vln-cloud-storage-volc`，保留已有百度桥接。如果开发机使用普通账号，配置 `"sftp_mode": "login"`：直接以 SSH 登录账号运行 SFTP，通过 `idmap=user` 将远端 UID/GID 映射为本机控制器用户，无需 sudo。此时镜像的 `RUN_UID` / `RUN_GID` 必须取远端 `id -u` / `id -g` 的值；配置文件中的 `uid` / `gid` 仍填本机控制器的值。默认 `setpriv` 模式维持原行为，要求 SSH 账号能够切换为控制器对应的远端数字用户。

火山单实例采集使用 `RuntimeConfig.Framework=Custom`。实测可采用 root 镜像用户完成平台初始化，再在资源池配置 `run_as_user`（例如 `"1022:1024"`），由调度器通过 `setpriv` 切换身份后打开日志并启动采集；镜像需包含该数字用户和 `setpriv`。采集过程与持久化文件仍使用普通账号。该字段留空时沿用镜像的默认用户，不改变已有部署指纹；变更为其他身份会冻结旧任务，需在停机核对后重新验收。

镜像、路径、原生模板等配置一旦用于任务便冻结校验。修改它们后不会直接复用旧断点；应恢复原配置后续跑，或创建新任务。停用/移除资源池会先停止其已有采集任务。

### 3. 检查并启动服务

完成验证后填写 `validated_datasets` 并设置 `enabled=true`。本机 `.env` 可配置：

```dotenv
VLN_CLOUD_POOLS_FILE=configs/cloud_pools.local.json
VLN_CLOUD_CREDENTIALS_FILE=/home/ljqa/.config/vln-pipeline/cloud.env
VLN_CLOUD_POLL_SECONDS=15
```

```bash
.venv/bin/python -m workers.cloud_scheduler --check  # 只读验证配置与云 API
.venv/bin/python configs/install_services.py --only vln-cloud
systemctl --user restart vln-api vln-worker vln-beat # 使用实际部署的服务名
systemctl --user status vln-cloud
```

`--only vln-cloud` 只添加云调度服务，保留现有端口与其他服务配置；不带 `--only` 则安装全部服务，适用于新部署。若开发站使用 `vln-dev-api` 等自定义服务名，重启对应服务。API、Worker、Beat 都应加载新代码后才启用云资源池，防止旧版恢复器把云任务派到本机。控制器只处理网站已提交的 `execution=cloud_idle` 任务，启动服务本身不会采集任意数据。也可前台运行 `.venv/bin/python -m workers.cloud_scheduler`。`--once` 会真实执行一轮调度，不是预览。

网页采集表单的「采集方式」勾选 PRO6000 / B200，检查范围并启用持续采集。`GET /api/compute-pools` 显示连接情况；任务列表和详情使用相同的设备名称，详情还显示云任务 ID、让位状态和分片进度。所有任务状态、产物浏览、审核发布继续使用现有 API。

## 验收

离线回归：`.venv/bin/python -m pytest tests/test_cloud_collection.py tests/test_collection_plans.py -q`。覆盖原生请求约束、完整分页、排队让位、停止确认后恢复、手动暂停/取消、重复投递、提交超时找回、监控失败、租约超时、进程组终止、配置改变及重试上限。

实际云验收需要上述接入资料：每个 GPU 池先采集一个 episode；再用小分片任务测试抢占，确认 GPU 确实归还、已完成分片的内容/mtime 不变、续跑没有重复样本、正式数据通过独立契约校验。另测关闭控制器后的租约退出和“手动暂停不自动恢复”。未经这些实测，不把资源池标记为已完成云端验收。

可用以下操作命令排队验证 R2R/RxR 镜像与数据。它通过正常的闲时调度器提交每个数据集一个 episode，持续等待前台资源需求消失；不会绕过排队或申请普通优先级任务。

```bash
.venv/bin/python -m configs.verify_cloud_pool --pool baidu-pro6000 --datasets r2r rxr
.venv/bin/python -m configs.verify_cloud_pool --pool baidu-b200 --datasets r2r rxr
```

验收任务 ID 与冻结配置记录在 `runtime/cloud-validation/<pool-id>.json`，命令重启复用原任务。只有原生闲时任务成功、产物完成独立校验、执行配置仍相同时，才将相应数据集加入 `validated_datasets`。有其他任务排队时，验收也会等待；可由用户服务托管该命令，空闲后自动验证和启用。手动暂停验收任务仍须手动恢复，验收命令不覆盖人工控制；失败或取消则不启用。这个命令只验证镜像和数据，不代替暂停续跑、租约故障等调度验收。

百度可能在同一任务 ID 下重新申请被回收的闲时资源；控制器保留该任务绑定，依靠 PFS 上的跨容器锁和断点恢复。若任务最终失败，回收原因从当前实例状态读取，不用历史实例的抢占记录掩盖后续应用错误。正常的应用失败仍受重试次数限制。

## 官方接口依据

- [火山预付费闲时任务](https://docs.volcengine.com/docs/ml-platform/prepaid-idle-custom-tasks?lang=zh)
- [火山 ML OpenAPI Python SDK 模型](https://github.com/volcengine/volcengine-python-sdk/tree/master/volcenginesdkmlplatform20240701)
- [百度 AIHC 创建训练任务](https://cloud.baidu.com/doc/AIHC/s/Hmayv96tj)（`idleTask`、`queueID`、v2 请求头）
- [百度 AIHC 使用闲时资源](https://cloud.baidu.com/doc/AIHC/s/mmnobxpu9)
- [百度 AIHC 查询任务列表](https://cloud.baidu.com/doc/AIHC/s/xmayvctia)、[查询详情](https://cloud.baidu.com/doc/AIHC/s/Kmayvejf0)
- [百度资源描述符与数据结构](https://cloud.baidu.com/doc/AIHC/s/Imb0gupts)
- [百度队列详情](https://cloud.baidu.com/doc/AIHC/s/Jmfcjhru0)、[开发机列表](https://cloud.baidu.com/doc/AIHC/s/Nmbkpgnrm)
- [百度 CCR 镜像临时凭据](https://cloud.baidu.com/doc/CCR/s/Ql6bxdxv6)

接口与 SDK 于 2026-09-28 核对；账号所在区域和集群组件版本仍须在真实接入时确认。

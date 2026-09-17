# 公网访问与运行维护

2026-09-17，用户明确同意公网传输后，已通过 Cloudflare Quick Tunnel 开放网站：

**https://crucial-prot-coupled-hdtv.trycloudflare.com/**

可直接分享根地址，无需访问者使用 localhost、SSH 转发或连接本机网络。
根页面默认列出全部 R2R/RxR 指令；带 query 的链接保留筛选，点击“查看全部”清除筛选。
页面、标注、地图和请求的视频经 Cloudflare HTTPS 隧道传输，原始 Parquet 留在当前服务器。

这是临时分享入口，依赖服务器、网站进程和隧道进程保持运行；不是固定域名部署。
重启网站通常可以保留链接，重启 Quick Tunnel 会分配新地址。固定网址需要配置命名隧道和域名。
参见 [Cloudflare Quick Tunnels 官方说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

## 网站服务

先确认旧服务是否仍占用 8788，不要重复启动。服务器上的命令：

```bash
python -u /mnt/pfs/tanshenghan/vln4project/data_web/serve.py \
  --vln-db /mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln.sqlite \
  --vln-videos /mnt/pfs/tanshenghan/vln4project/data_web/runtime/vln-videos-all.sqlite \
  --host 127.0.0.1 --port 8788
```

网站保留本地监听，由隧道连接它；访问者使用上方公网地址。
服务启动时核对全部数据源指纹，准备完成后才监听。网站重启不影响独立运行的视频批量任务。

## 公网隧道

官方 cloudflared 2026.9.1 已安装在 `data_web/runtime/tools/cloudflared`，
SHA-256：`03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc`。
需要重新创建隧道时运行：

```bash
/mnt/pfs/tanshenghan/vln4project/data_web/runtime/tools/cloudflared tunnel \
  --url http://127.0.0.1:8788 --no-autoupdate --protocol http2
```

使用输出中的新 HTTPS 地址，并更新本文件及 `runtime/public_access.json`。
不要为了刷新网页而重启隧道；当前网站静态文件刷新即可生效。

## 视频后台任务

当前任务使用 8 个 CPU worker，处理全部 71,071 条已验证观测，状态持久保存。
查看 `runtime/vln-video-batch.pid` 的进程及 `runtime/vln-video-batch.log`，
或请求公网 `/vln-api/video-progress`。任务独立于浏览器，关闭网页不会中止渲染。
断点续跑命令、数据源配置、原始文件和成片目录见 [lerobot_video.md](lerobot_video.md)。
不能将 71,071 条来源映射解释为全部视频已生成；以进度中的 ready/failed 为准。

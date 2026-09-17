# GT 渲染器代码快照

为使 Codeup 的 data_web 仓库可独立构建视频，本目录随网站发布渲染器源代码。
工作区维护源为 `LightNav-0/visual/render_episode_gt.py`，本次快照与其逐字节一致。
后续从维护源同步，不在两个位置分别实现不同算法。

- 版本日期：2026-09-17
- SHA-256：`45d98c0d60b488f190e3aa98cd83920f4953491a2d08933d7e30fc276510b5cf`
- 依赖：见仓库根目录 `requirements-vln-video.txt`
- CPU 测试：`tests/test_render_episode_gt.py`、`tests/test_render_lerobot_gt.py`

构建视频索引时传入 `--renderer /absolute/path/to/data_web/visual/render_episode_gt.py`。
网站运行只需标准库，视频编码需要安装视频依赖；无需 Habitat、Torch 或 GPU。

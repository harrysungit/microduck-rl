# 官方源码、扩展仓和版本更新

2026-11-06 的仓库整理把旧 fork 拆成官方依赖与扩展包，减少维护整份 fork 的负担。
2026-11-18 核对：本扩展仓基线 `3285093` 跟踪 71 个文件；官方依赖固定版本 `8d0db749`
跟踪 241 个文件。两者都没有 sparse checkout，也不是 shallow clone。文件数量不同是设计，不是同步漏文件。

| 位置 | 内容与作用 |
|---|---|
| 本仓库 `harrysungit/microduck-rl`（连字符） | HD-1910、ROCm、任务注册及评测工具 |
| 安装时 `.deps/microduck_rl` | 官方完整源码；实际路径以安装环境为准 |
| 已有开发环境 `micro_duck/training/repos/microduck_rl` | 历史安装的完整官方依赖 checkout |
| `src/mjlab_microduck/`（官方依赖内） | 官方任务、奖励、机器人模型等 |
| `env/pins.toml` 与 `pyproject.toml` | 官方依赖 commit，必须一致 |
| `docs/upstream/AGENTS.develop.md` | 官方手册参考快照，不是安装版本也不是自动更新文件 |

旧的 `harrysungit/microduck_rl`（下划线）fork 已删除。历史 checkout 的 origin 可能仍指向它；
识别官方来源请看 upstream 和固定 commit，不能把过期 remote 当成当前依赖缺失。
2026-11-19 已把现有开发环境依赖 checkout 的 origin 修正为官方地址，固定 commit 未升级。

## 怎么知道官方更新了

```bash
python scripts/check_upstream.py --out-dir logs/upstream-review
```

检查从官方 Git 获取 develop，输出当前固定版本、远端版本、提交列表、全文件变化清单和重点源码 diff。
同时读取官方 `pollen-robotics/microduck-policies` 的发布目录，对比 `docs/upstream/policies.json`
中各个 ONNX 的 SHA256，发现模型新增、移除或替换。因此源码没变、只换发布模型也能发现。
只取公开目录和哈希，不下载或部署模型；私人 W&B 试验和其他未列明发布渠道不在检查范围。
默认缓存是 `.cache/upstream-review.git`，不改安装依赖、不切换工作分支、不修改 pins。
退出码 0=同版本或已按准确版本记录审查，3=存在差异待评审，2=检查失败。网络失败不会伪装成无更新。
本仓库队列启动时执行检查；如已审查而决定暂留旧版本，可传
`--reviewed-upstream <完整远端SHA> --upstream-review-note '暂缓原因/对照报告位置'`。
模型文件变更还需传 `--reviewed-policies <目录revision>`，源码审查不能替代模型变更审查。
若官方又前进，旧的确认不能放行。官方 `train` 命令本身不会执行这个检查，手动训练先按上面检查。
这是每次运行时检查，并没有创建后台定时通知。

升级应在单独分支：保存差异报告 → 检查任务/奖励/模型/导出/API变化 → 同时更新两处 pin →
在独立环境安装 → 配置与导出对照 → 64×5烟雾检查 → 原版本与新版本同条件评估和视频。
正在训练的环境不能原地升级。发现新版策略可下载核验来源，但官方 XL330 模型不能直接当本机 HD-1910 候选。

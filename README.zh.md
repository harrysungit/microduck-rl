# microduck-rl

[English](README.md) | [中文](README.zh.md)

[Microduck](https://github.com/pollen-robotics/microduck) 的训练，支持更多舵机和 GPU，建立在 Pollen Robotics 的
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) 之上。

这不是 Pollen 的仓库。它是一个依赖 microduck_rl（原样不改）的包，通过 mjlab 的 `mjlab.tasks` 入口，
在官方任务旁边注册额外的 [mjlab](https://github.com/mujocolab/mjlab) 任务。`train`、`play` 和导出的用法和上游一样。

它加了什么：

- **飞特 HD-1910 舵机**：执行器模型（BAM 的飞特控制律，配 LuwuDynamics 的 HD-1910 M6 参数，固件 P = 5，
  含电源压降）、质量模型（每颗舵机 21 g，整机 0.82 kg），以及官方任务的 HD-1910 版。
- **ROCm**：借助 [AMD 的 Warp HIP 移植](https://github.com/AMD-Ecosystem/warp/tree/amd-integration-halo)，
  让 mjlab 和 MuJoCo Warp 跑在 AMD GPU 上，并修掉拖慢它的几处问题：在 Radeon 8060S（gfx1151）上吞吐是
  直接移植的 7.3 倍。见 [benchmark/](benchmark/README.md) 和 [docs/ROCM.md](docs/ROCM.md)。
- **策略算数之前的检查**：验收门在训练前声明一个技能"能用"指什么，训练后给出结论；另有独立的 sim2sim，
  在普通 MuJoCo 里运行，不经过训练用的世界。

2026-11-18：SitStand lowstd 在仿真和实机试验发现头壳触地，旧验收结论已撤回，不能作为上机候选。
训练好的策略暂未发布。开始训练前阅读 [训练规则](AGENTS.md)、[仓库与官方更新](docs/REPOSITORIES.zh.md)
和 [坐站新方案](docs/SITSTAND_PLAN.zh.md)。

## 技能

以下保留历史技能检查结果；坐站已撤回。自动指标通过不替代完整动作视频的人工核验。

| 技能 | 任务 | 验收（训练世界） | Sim2sim（MuJoCo C） |
|---|---|---|---|
| [走路并起身](examples/walk/README.md) | `Mjlab-VelStand-Flat-MicroDuck-HD1910` | ACCEPTED，1 条警告 | 通过 |
| [起身](examples/standup/README.md) | `Mjlab-StandUp-Flat-MicroDuck-HD1910` | ACCEPTED | 通过，46/48 |
| [坐下与站起](examples/sitstand/README.md) | `Mjlab-SitStand-Flat-MicroDuck-HD1910` | 已撤回：头壳触地 | 旧姿态检查漏检 |
| [踢球](examples/kick/README.md) | `Mjlab-BallKick[Left]-Flat[-Backlash]-MicroDuck-HD1910` | ACCEPTED（草案契约） | 通过 |
| [低头捡物](examples/ground_pick/README.md) | `Mjlab-GroundPick-Flat[-Backlash]-MicroDuck-HD1910` | ACCEPTED（草案契约） | 还没跑 |

Roulade（前滚翻）能训练，但没通过它的契约，见[路线图](docs/ROADMAP.md)。

## 任务

15 个 HD-1910 任务，都是官方任务换上 HD-1910 机器人：

- `Mjlab-Velocity-{Flat,Rough}[-Backlash]-MicroDuck-HD1910`
- `Mjlab-VelStand-Flat[-Backlash]-MicroDuck-HD1910`、`Mjlab-VelStand-Rough-Backlash-MicroDuck-HD1910`
- `Mjlab-StandUp-Flat-MicroDuck-HD1910`、`Mjlab-SitStand-Flat-MicroDuck-HD1910`
- `Mjlab-GroundPick-Flat[-Backlash]-MicroDuck-HD1910`
- `Mjlab-BallKick-Flat-MicroDuck-HD1910`、`Mjlab-BallKickLeft-Flat[-Backlash]-MicroDuck-HD1910`
- `Mjlab-Roulade-Flat-MicroDuck-HD1910`

带齿隙的版本用稠密雅可比：34 个自由度时 MuJoCo Warp 会选稀疏路径，在这里更慢。VelStand 的行为克隆
从本地 HD-1910 检查点取老师（`HD1910_STAND_EXPERT`、`HD1910_WALK_ANCHOR`）。

## 安装（ROCm，gfx1151）

需要 Linux、`/opt/rocm` 下的 ROCm 7.2（Warp 用它编译）和 Python 3.12。

```bash
git clone https://github.com/harrysungit/microduck-rl.git
cd microduck-rl
bash env/setup.sh          # .venv/、.deps/、.cache/ 都在仓库内
source env/activate.sh
python env/check_env.py    # 设备、一个 Warp kernel、ROCm 补丁、任务
```

`env/pins.toml` 固定所有来源：Pollen 的 microduck_rl、AMD 的 Warp 分支、gfx1151 的 torch 夜间版，以及
microduck_rl `uv.lock` 里的 Python 包。`env/activate.sh` 把所有缓存放在 `.cache/` 下，并打开 HIP graph 捕获和
PyTorch TunableOp。

其他 GPU：ROCm 补丁在 CUDA 和 CPU 设备上不起作用。在可用的 microduck_rl 环境里装上本包
（`pip install --no-deps -e .`）应该就够了，但没有测过。

## 训练

```bash
source env/activate.sh
python scripts/check_upstream.py --out-dir logs/upstream-review
# 先审阅上游变化；以下原生 train 命令不会自行执行该检查。
train Mjlab-Velocity-Flat-MicroDuck-HD1910 --env.scene.num-envs 4096 --env.sim.njmax 160 \
    --agent.max-iterations 10000 --agent.logger tensorboard
```

`--env.sim.njmax` 很重要：mjlab 默认每个 world 1500 个约束行，约是走路实际用量的 30 倍，降低预算会快很多。
预算有保护：某个 world 要超出时，训练以 `ConstraintBudgetExceeded` 停下，而不是悄悄丢掉约束行。

五个技能从零开始、无人值守地训完（gfx1151 上约 31 小时）：

```bash
nohup setsid python scripts/queue_runner.py --queue queues/hd1910_skills.json > /dev/null 2>&1 < /dev/null &
```

队列给每个阶段设时间上限和卡死检测，收敛即停，串起热启动和续训，然后评估、录视频、跑验收门。

## 检查一个策略

```bash
# 验收：在导出的 ONNX 上跑技能的契约，干净世界和训练世界各一次
python scripts/acceptance.py --task Mjlab-StandUp-Flat-MicroDuck-HD1910 --onnx policy.onnx \
    --checkpoint model_5000.pt --spec getup --out-dir logs/results/standup_acceptance

# Sim2sim：同一个 ONNX 在普通 MuJoCo（CPU）里跑，用它自己的舵机环
python scripts/sim2sim_assets.py --out-dir logs/sim2sim/assets
python scripts/sim2sim.py --onnx policy.onnx --assets logs/sim2sim/assets --out-dir logs/results/walk_sim2sim
```

结论是 ACCEPTED、FAILED 或 UNVERIFIED。UNVERIFIED 表示没有失败项，但有东西没检查到，不算通过。
契约见 [scripts/acceptance_core.py](scripts/acceptance_core.py)。

部署时用 Pollen 的 `scripts/export.py` 导出，它把观测归一化写进 ONNX。接口以及在 HD-1910 机器人上的不同之处见
[docs/DEPLOY.md](docs/DEPLOY.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/microduck_rl/tasks/` | 任务注册，以及它应用的运行时补丁 |
| `src/microduck_rl/actuators/hd1910/` | HD-1910 执行器、质量模型、机器人配置、BAM 参数 |
| `src/microduck_rl/patches/` | `rocm.py`、`perf.py`、`fixed_std.py`：对 mjlab、Warp 分配器用法和 rsl_rl 的运行时补丁 |
| `env/` | 固定的来源、安装、激活、环境检查、MuJoCo Warp 的两处修补 |
| `scripts/` | 评估、验收、sim2sim、训练队列、吞吐和性能分析 |
| `queues/` | `scripts/queue_runner.py` 的训练队列 |
| `examples/` | 每个技能一个目录：配方、数据、局限 |
| `benchmark/` | gfx1151 上的吞吐 |
| `docs/` | 部署接口、路线图、ROCm 说明 |
| `tests/` | CPU 测试：`python -m pytest tests` |

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 和 [NOTICE.md](NOTICE.md)。本仓库不含 3D 模型文件：机器人的网格随 Pollen 的
microduck_rl 提供，适用那里写明的许可。

# Throughput on a Radeon 8060S (gfx1151)

Measured with `scripts/bench_throughput.py` and full training runs, GPU idle
otherwise. Env steps per second count every world. Power is the APU package
power (PPT) sampled with `scripts/sample_gpu.py`. The fixes are described in
[docs/ROCM.md](../docs/ROCM.md).

## One environment step, velocity task (XL330), 4096 worlds

| Configuration | Per step | Env steps/s | Against the row above |
|---|---|---|---|
| Plain port | 790 ms | 5,185 | — |
| + caching allocator | 161 ms | 25,387 | 4.9x |
| + `njmax` 128 | 129 ms | 31,870 | 1.26x |
| + no origin marker sites | 113 ms | 36,141 | 1.13x |
| + TorchScript fusion off | 77 ms | 53,306 | 1.47x |
| + HIP graph capture | **70.9 ms** | **57,749** | 1.09x |

## Training loop, velocity task (XL330), by number of worlds

Env steps per second of the whole loop (collection and PPO update).

| Worlds | Plain port | Caching allocator | First four fixes | All six fixes | Power, efficiency (all six) |
|---|---|---|---|---|---|
| 1024 | 4,062 | 14,298 | 16,338 | — | — |
| 2048 | 4,625 | 19,924 | 23,786 | 28,978 | 84.8 W, 342 steps/J |
| 4096 | 5,126 | 22,716 | 31,417 | **37,286** | 95.3 W, **391 steps/J** (plain port: 123) |
| 8192 | — | 21,872 | 36,549 | **41,826** | 103.0 W, 406 steps/J |

## The same 187 iterations, 4096 worlds

| Run | Wall time | Median env steps/s | Final mean reward |
|---|---|---|---|
| Plain port | 59.4 min | 5,138 | 91.0 |
| Caching allocator | 13.3 min | 23,008 | 88.2 |
| First four fixes | 9.6 min | 32,194 | 99.8 |
| All six fixes | **8.0 min** | **38,545** | 95.1 |

The learning curves overlap by iteration; the final rewards differ within
the run-to-run spread of GPU training.

## HD-1910 tasks, 4096 worlds, all fixes

| Task | Env steps/s (training loop) |
|---|---|
| `Mjlab-Velocity-Flat-MicroDuck-HD1910` | 38,300–38,700 (three 187-iteration runs) |
| `Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910` (34 DOFs, dense Jacobian) | 16,800 |
| `Mjlab-StandUp-Flat-MicroDuck-HD1910` | 32,200 |
| `Mjlab-GroundPick-Flat-MicroDuck-HD1910` | 36,500 |
| `Mjlab-BallKick-Flat-MicroDuck-HD1910` | 29,700 |
| `Mjlab-SitStand-Flat-MicroDuck-HD1910` | 25,200 |
| `Mjlab-VelStand-Flat-MicroDuck-HD1910` | 13,500 |

Medians of each skill's training run, with that stage's constraint budget
from `queues/hd1910_skills.json` (VelStand needs 400 rows per world, the walk
160). A full 10,000-iteration HD-1910 walk takes about 7 h.

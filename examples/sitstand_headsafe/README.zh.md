# 坐站头部安全训练：实验配方

这是独立的有界训练配方，继承本扩展包的 HD1910 SitStand 基础任务。目录存在、烟雾测试通过、GPU奖励上升，都不表示动作成功或模型可上机。本配方不包含模型、设备校准、私有运行目录或下载它们的凭据；仅显式执行命令时才读取外部校准并注册实验任务。

## 准备与输入

先按仓库安装说明准备固定版本训练环境，阅读根 AGENTS、`docs/TRAINING_WORKFLOW.zh.md` 和 `docs/SITSTAND_PLAN.zh.md`。需要 NumPy、SciPy、PyTorch、MuJoCo/mjlab、rsl-rl、onnx/onnxruntime、pytest 与 ffmpeg。使用与任务兼容的61维观测、14维原始 actor 输出 ONNX，包含可兼容 MLP 和观测归一化；不支持任意模型架构。

所有设备文件与产物置于仓库外。`--calibration` 为设备自己的 TOML：`[[joints]]` 中按名称提供 `min_rad/max_rad/zero_raw/direction`，名称见 `calibration.py`。这里没有可代用的设备数值。物理目标高度、供电及BAM是本次HD1910名义实验条件，迁移其他硬件必须重新测量与验证。

`--anchor-data` 是经过独立接触/视频审查的 NPZ，键为 `observations:(N,61)` 和 `actions:(N,14)`；actions 为原始确定性 actor 输出，不能填写滤波后目标或实际角度。入口会逐行运行所选teacher，核对全部标签误差≤1e-3，并将teacher、anchor、calibration、配方源码及设置hash绑定写入 `bootstrap.json`。标签匹配只能证明来源一致，不能证明采集动作安全。

可用以下诊断采集器生成完整13秒站→坐→站视频及anchor；它使用GPU，请先确认没有并发训练：

```bash
python examples/sitstand_headsafe/collect_anchor.py \
  --teacher /external/policy.onnx --calibration /external/device.toml \
  --out /external/anchor-review
```

采集器记录全程，失败也保留，并生成异常慢放。它仅控制步采样几何间隙，头接触使用物理子步锁存；报告是GPU一致性诊断。必须另用本仓CPU全几何/子步审计和正常速度视频核验，不能仅凭其报告批准模型。默认从站姿开始；另用 `--initial-posture sit` 检查坐姿reset，用 `--profile operating` 覆盖显式operating条件。

## 检查与运行

先检查本配方实际使用的完整碰撞场景；基础SitStand的groundcontact前检不能代替它：

```bash
python scripts/sim2sim_assets.py --out-dir /external/cpu-assets
python scripts/pose_preflight.py --scene allcollisions --assets /external/cpu-assets \
  --calibration /external/device.toml --out-dir /external/fullcollision-hold --video
```

实测此场景的SIT固定控制平衡约65.05mm，groundcontact约61.5mm。历史V2–V5沿用了61.5mm目标，
虽宽容差下可通过核心动作检查，仍应在新物理对齐实验中统一修正奖励、分类、command高度和FK初态，
不能只改其中一个值。当前目录保留对应历史实验实现与来源记录，不能称为已完成最终目标对齐的模型。

以下CPU检查不启动训练或仿真：

```bash
python -m py_compile examples/sitstand_headsafe/*.py
python -m pytest examples/sitstand_headsafe/test_recipe.py -q
python examples/sitstand_headsafe/train_headsafe.py \
  --calibration /external/device.toml --teacher /external/policy.onnx \
  --anchor-data /external/anchor-review/teacher_rollout.npz \
  --out /external/input-check --validate-only
```

先运行64环境×5更新烟雾测试；该步骤会启动GPU：

```bash
python examples/sitstand_headsafe/train_headsafe.py \
  --calibration /external/device.toml --teacher /external/policy.onnx \
  --anchor-data /external/anchor-review/teacher_rollout.npz \
  --num-envs 64 --iterations 5 --out /external/smoke \
  --reviewed-upstream <准确审查的源码SHA> \
  --reviewed-policies <准确审查的模型目录revision> \
  --upstream-review-note '保留固定依赖的理由及审查记录'
```

`--profile nominal`显式把两个观测项的IMU错位上限设为0°；`--profile operating`设为2°，并恢复有界随机化：躯干/头COM±3mm、质量.95–1.05、关节摩擦/armature .9–1.1、encoder bias±.0075rad、脚摩擦.7–1.3，无push，无按轮数加难。实际配置写入params与profile.json，不能仅凭profile标签推断条件。旧实验仅删除events而保留观测IMU6°的条件不属于这里的nominal定义。`--validate-only`不构建仿真，因此不证明任何profile下动作通过。`--profile`及两种锚系数和warmup轮数都绑定在恢复身份中。

入口先执行官方更新检查；未知或未审查更新会在训练开始前退出。已审查版本再次变化仍需重新核对。每个输出目录必须不存在，不会覆盖旧实验。

只有配置、几何、导出与动作核验满足条件后，才考虑≤1024环境×200更新的小试。超过5更新必须提供 `--review-every` 和 `--assets`；例如 `--iterations 200 --review-every 100 --assets /external/cpu-assets`。检查边界会导出ONNX并调用仓库 `scripts/sitstand_check.py` 生成完整视频/接触审计；任何非PASS均停止追加更新。自动PASS仍待人工视频核验，不自动部署。

默认探索std=.05、固定anchor系数10、当前PPO状态在线teacher锚系数25、teacher学习率3e-5、前10更新只训critic、命令目标高度ramp=.4s。在线锚使用冻结原teacher为当前storage状态标注；前10更新在PPO和两种BC梯度都加入之后把actor梯度置None，避免Adam动量或hook继续更新actor。它们是诊断参数，并非已证实的最优配置。每块更新后断言teacher观测归一化完全未变。BC梯度在PPO backward之后、同一次Adam更新前加入并重新裁剪actor梯度。

## 初始化与恢复

默认是 **teacher warm start**：只导入actor与冻结的观测归一化；critic、optimizer和计数从零开始。它不是从零训练，也不是恢复teacher原训练。

`--resume /external/previous/model_final.pt` 是 **本配方检查点恢复**：保留actor、critic、optimizer、迭代/全局步计数，并再次显式冻结观测归一化；必须提供相同teacher、anchor、校准、配方源码和训练设置，hash不一致拒绝恢复。新的 `--iterations` 表示追加更新数。仿真episode状态与随机数流未序列化，恢复从新的端点episode开始，不能声称逐位连续。只加载自己信任的检查点。

常规周期检查点和最终检查点均内嵌recipe绑定信息。不要用旧实验检查点冒充这里的resume；更换teacher或数据应开新warm-start run。改变归一化策略需要显式修改配方并记录新实验身份，当前CLI不提供静默解冻。

## 已知约束与交付边界

当前配方的用户头部命令范围固定为0。真实合同会让固件在活动SitStand网络期间把四维头部请求夹到0；
模型仍能自主动头平衡。外加head命令的压力回放绕过了该合同限幅，只验证坐站/触地安全，
没有独立追踪误差评分。开放可控头部需要另做范围课程及完整验证，不能把合同写成未训练范围。

- 保留来源实验的13秒端点reset与3/8秒反向/返回命令、每子步头接触失败锁存、奖励失败屏蔽、完整头颈几何间隙和符号。它不是安全中间状态课程。
- 当前训练端头接触解析neck完整子树（当前4个body，额外子body自动覆盖）；bearing_roll是右髋yaw，独立记录接触/间隙，合法坐姿按含该右髋的8个躯干/髋/腿body逐物理子步检查，要求法向力>.05N的躯干或左右双侧支撑，连续1秒并满足端点关节误差<.3rad。脚独撑低蹲不能算坐姿。GPU与独立CPU实现仍需交叉核验，`valid_transition_count`不是最终验收。
- 两端各50%初始化，固定头部指令。仅在nominal条件选择best仍不够：必须另跑独立operating条件回归和留出种子，覆盖真实传感误差、目标处理及完整站→坐→站，再检查行走切入与返回。训练随机化出现过类似条件不等于完成该验证；nominal通过的模型仍可能在operating条件碰头。也没有证明训练后模型超过teacher。
- actor/ONNX一致性不包含外部motor目标处理。导出的输出仍须按校准、7.4/V补偿、电压EMA、头/腿低通.5/.7、限位与整数编码器量化处理；纯CPU测试验证该实现与独立NumPy参考的输入输出一致性。
- `runtime_state.py` 由本项目已有部署对齐实现复制必要TargetState/VoltageState，并自包含保留同源算法；没有对私有路径的导入。
- 每次模型验证必须交付正常速度完整站→坐→站视频、失败样例与必要慢放，附模型/checkpoint/hash、仿真条件和关键时刻。人工核验前统一标为“自动检查完成，待人工核验”，不能作为可上机候选。
- 本目录不写入设备校准值；运行时的params、日志、检查点和视频可能含设备私密信息，留在仓库外，不随配方提交。



## 仅供审查的导入包

`package_review.py` 从所选ONNX、配对checkpoint、本次真实 `params/env.yaml`/`agent.yaml`、`initialization.json` 和归档 `source/` 构造新合同；不读取旧deployment-contract，不改ONNX，不自动选择模型，不调用机器人服务。YAML中的Python标签只解码成数据，不执行构造函数。运行时算法必须匹配本配方已审查的AST身份，不能把旧流水线声明套到已变化代码。

```bash
python examples/sitstand_headsafe/package_review.py \
  --onnx /external/run/policy.onnx --checkpoint /external/run/model_final.pt \
  --params /external/run/params \
  --task Mjlab-SitStand-Flat-MicroDuck-HD1910-HeadSafeEndpointsV2-20261118 \
  --evaluation-summary /external/final-evaluation.json \
  --name 'HeadSafe review' --out /external/review-package
```

最终报告必须有匹配ONNX的 `policy_sha256`、明确的 `automatic_pass:true` 与 `operating_pass:true`，并提供非空 `declared_checks` 与 `rows`。所有预先声明的检查和实际rows必须一一对应、无重复/遗漏，不能只抽取通过的核心条件。递归检查整份报告，任何层级的FAIL/FAILED/UNVERIFIED、头接触、automatic_pass非true或任何*_pass=false都阻止final包装；operating通过不能掩盖stress/留出/切换失败。若报告含checkpoint或校准hash也必须匹配。脚本总是输出 `review_only:true`、`human_review:pending`、`deployment_approved:false`，不会因自动通过而声称人工批准。设备校准仅记录hash，校准值和校准路径不进入合同。

可选 `--validator /path/to/x86/robotd` 调用 `validate-policy` 离线子命令；若ORT需要动态库，由调用者设置 `ORT_DYLIB_PATH`。输出验证日志与ZIP位于外部 `--out`。它证明合同可解析、网络形状/一次零观测推理可运行，不能证明动作安全或当前设备设置一致。

没有最终候选时，只能使用 `--diagnostic-only` 测格式；此时summary必须明确 `diagnostic_only:true, operating_pass:false`，包名/状态标为非候选，不能把格式探针冒充自动安全验收。默认模式拒绝该证据。

合同使用固件已支持的 `hd1910-v1` 和真实新SitStand任务名。`runtime_target_pipeline`诚实声明外部RuntimeAction处理；当前该适配器不会自动强制设备头/腿滤波、电压或坐站窗口设置，导入前须另行核对。slot切换可能保留滤波历史，仍需行走切入/返回测试。用户完成视频核验后才考虑import；本脚本永远不执行import或启用动作。

## 本配方实现检查记录

公共入口已实跑operating 64环境×5更新（critic warmup为2更新），并用相同源码/输入/设置恢复追加2更新：迭代5→7、全局控制步120→168，optimizer步计数持续增长，所有actor归一化buffer逐位不变、count始终0；两次actor/ONNX最大误差分别约4.77e-7和4.47e-7。随后新增的打包辅助脚本不改变训练算法；恢复身份严格绑定源码，历史run使用各自归档源码。

现有15项CPU测试及编译通过，650行teacher标签完整核对误差0。现成x86固件loader已成功验证显式diagnostic-only格式探针。以上仅证明配方训练/恢复/导出/打包流程可运行，未将烟雾模型进行动作验收或认定为候选；最终模型仍须独立operating、留出、完整视频及人工核验。

## 头部命名审计更正

官方固定版本明确注明：`bearing_roll`是`trunk_base`下的右髋yaw，仅因历史错误保留在HEAD_BODY_NAMES/DR列表。它不是头部。旧归档与较早的本配方GPU烟雾报告将“neck子树+右髋”的superset误标为全头颈；约14.9–22mm这类旧最小值可能由右髋贡献，不能作为真实头部间隙。旧报告保持原样，不追写成新审计。

当前审计与未来配方使用父子树确定真实头部（neck、neck_pitch、yaw_roll_motion、jaw_soft，并覆盖未来新增子体），只对该范围施加碰头门禁；独立报告右髋与历史superset统计。`bearing_roll`接地可成为右侧髋支撑，不能冒称碰头。历史失败的`jaw_soft/top_head_shell`碰撞仍然属于真实头部，不能借此次更名撤销失败。

为复现既有物理随机化，operating的历史五body COM DR保持原分布，变量改为LEGACY_HEAD_DR_BODIES并附scope说明；`sim2sim.sample_params`的历史DR列表也不改。旧scope的GPU烟雾仍标为历史superset；现已另行执行真实neck-subtree-v2的teacher双初态传感器运行和operating64×5数值烟雾，不能用旧run替代新证据。

### 真实头部范围GPU补充核对

原teacher ONNX `1c94934c6421ec72ed48079cea74966868a7a59f7d732e0207d0f7bdaeed3c50`，在显式nominal（IMU错位0°）、校准限位/滤波/电压/量化处理下，从站/坐两初态各运行完整13秒，分别录制站→坐→站与坐→站→坐正常速度视频。实际几何范围为4个neck子树body、52个geom；独立legacy右髋字段与8槽合法支撑传感器均正常运行。两次均记录2次有效转换、0真实head接触、0legacy髋接触；控制步采样的真实head最小间隙约85.159/83.896mm，legacy右髋约21.598/21.691mm。间隙是控制步采样，不冒称子步全程最小值；接触锁存逐物理子步更新（每次2600更新）。视频仍待用户人工核验。

另用真实head范围源码执行operating 64环境×5更新、critic warmup2数值烟雾，源hash已绑定：normalizer count0、全部导入buffer保持不变，actor/ONNX误差约5.96e-7。新烟雾ONNX `9b1d15cfd0db747c3ca3df8ecd5fb4b29036859fe9117f03b4f43d4ff9c93ce2` 仅是数值流程产物，没有对它完成动作验收、没有将其标为候选，也未部署。

### 最终证据的完整行要求

`declared_checks`使用非空唯一check_id列表，或每项包含`condition`与整数`seed`；每个row必须使用相同身份。所有rows都要求明确 `automatic_pass:true`、`legal_sit_support_pass:true`，且有完整`safety`：

- `head_scope_version:"neck-subtree-v2"`、`head_scope_source:"parent-link neck subtree"`；真实head body至少包含neck/neck_pitch/yaw_roll_motion/jaw_soft，排除bearing_roll，head_geometry_count为正整数。
- `head_contact:false`、`head_contact_substeps:0`、`first_contact_s:null`；缺项不能用顶层PASS替代。
- physics_substeps与expected_physics_substeps均为正整数且相等，coverage必须明确为1；min_clearance_m有限且非负。

整份报告中的失败行即使被标为非核心或放在更深的结果列表里，也不能被隐藏。缺审计数据、仅operating_pass的局部summary、历史superset范围和不完整physics覆盖都不能包装成final。已经发现的strong/stress失败应原样披露，不得通过删条件、删rows或改顶层标志获得通过；目前此类模型只能生成明显DIAGNOSTIC ONLY的审查材料。

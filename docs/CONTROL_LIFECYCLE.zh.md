# HD-1910 执行器的控制时钟（显式可选）

`microduck_rl.actuators.hd1910.lifecycle` 提供独立的执行器与动作组件。
导入不会注册任务，也不会替换已有执行器、动作或公共坐站配方的默认配置。
采用时须注册新的任务名、保存配置和源码身份，按新的训练条件进行训练或明确的跨任务 warm start；
不能把旧 checkpoint 当作同条件 resume，也不能回写旧模型的评估结果。

## 为什么重置也需要检查

固定依赖的 `ManagerBasedRlEnv.reset()` 在派生 `_reset_idx()` 写完姿态后调用
`scene.write_data_to_sim()`，再执行 `sim.forward()`。前者会依次执行执行器的
`get_command → apply_delay → compute`，即使没有调用 `sim.step()`。
因此只看时间、物理步计数或单独的 DelayBuffer 单元，不能证明完整环境没有推进控制状态。

完整环境的零物理步捕获发现：原重置清除了目标，最终姿态已经写入，但首次静态写入仍会将零绝对关节目标
送入延迟历史并计算 BAM；平滑目标和前次力矩随之改变，第一条策略动作的电压估计也会读取这份力矩。
局部重置后的全场景写入，还可能改变未重置环境的延迟、平滑目标和力矩，而那些环境的时间和姿态没有改变。
这不是“从前一姿态错误播种”：懒初始化使用的是最终真实位置，之后的控制计算才推进状态。

## 行为与使用范围

`LifecycleHd1910Actuator` 保留原 HD-1910/BAM 控制律、参数、延迟分布和摩擦计算。
只有动作的 `apply_actions()` 授权后，下一次写入才执行原延迟和原计算各一次，随后撤销授权。
安装版循环每个物理子步都会调用 `apply_actions()`，因此不是每个控制帧只计算一次。

没有授权的重置或派生量刷新写入不追加延迟、不计算 BAM、不更新摩擦预算或前次力矩，只返回缓存的控制力矩。
重置只清选中环境的缓存，保留其它环境最后一次真实物理计算的控制。原父类的 reset 行为保留；
没有为猜测的模型差异额外清理摩擦预算、阻尼或摩擦随机化参数。

该方案适用于当前单线程 Python `apply_action → write_data_to_sim → sim.step` 循环，
一个动作项完整拥有一个非齿隙 HD-1910 执行器。它没有验证整个 Python 循环的图捕获、并发控制写入、
多个动作项共享执行器、齿隙执行器或任意其它自定义执行器。
没有跳过 Scene 路径或外力写入；同一控制通道存在其它竞争写入者时，需要另定控制所有权。

## 标准位置动作的显式配置

只在新的任务配置中调用，且保留该任务原来的全部其它设置：

```python
from microduck_rl.actuators.hd1910.lifecycle import (
    with_lifecycle_actuator,
    with_lifecycle_joint_position_action,
)

# cfg 是新任务自己的配置；不要修改共享机器人常量。
cfg.scene.entities["robot"] = with_lifecycle_actuator(cfg.scene.entities["robot"])
cfg.actions["joint_pos"] = with_lifecycle_joint_position_action(cfg.actions["joint_pos"])
```

第一个 helper 深拷贝机器人配置，只替换已知的单个非齿隙 `Hd1910ActuatorCfg` 的构建类型。
第二个只接受标准 `JointPositionActionCfg`；遇到未知动作子类会拒绝，避免丢失校准、低通、量化、
电压补偿或其它自定义处理。所有原配置字段的值保持不变。
标准位置动作仍调用原 `JointPositionAction.apply_actions()`，包括目标侧 `encoder_bias` 抵消，再授权控制写入。

对于已有的自定义动作，应显式定义 `class NewAction(PhysicsWriteActionMixin, ExistingAction)`，
并让新的配置 `build()` 在保留原构建验证后返回 `NewAction`；不能只改一个可能被自定义 `build()` 忽略的
`class_type` 字段。Mixin 在构造时核对唯一执行器和完整关节所有权；原动作的处理、重置和偏置语义仍由原类负责。
动作与执行器必须成对启用。只替换其中一个不构成正确配置。

## 验证范围与后续要求

公共 CPU 回归使用真实 HD-1910/BAM 的构造、初始化、延迟、计算和重置方法，以 CPU 张量容器承载模拟器存储，
检查授权路径与原父类逐位一致，以及未授权重复写入、随机状态、摩擦字段、局部重置、缓存别名和配置保持。
这不是完整 Warp 桥接、GPU 图或动态模型验收。

移植所依据的隔离实现还完成过完整 GPU 零物理步捕获与 64 环境 × 5 更新检查，确认全重置、局部重置以及每个
真实控制帧的四次物理子步授权；这些是独立实验的证据，不冒称本公共副本重新完成了 GPU 或训练验证。
这些是控制生命周期证据；模型动作能力和上机资格仍需独立验证。

采用新配置后仍应重新记录实际实例类、源码/配置哈希、全重置和局部重置前后完整状态，验证真实物理子步
对应一次延迟与计算，再完成训练烟雾、导出对照、独立动作评估和完整视频人工核验。
首次策略目标填充也不等于 CPU `measured` 或原生测位预装，更不证明 CPU/GPU/实机物理完全一致。

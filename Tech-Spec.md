# D435i真实深度预处理技术规格

## 输入输出契约

```text
uint16 (240, 424)
-> depth_scale 转米
-> 无效/超距填充 2.0 m
-> 4×4 area downsample
-> float32 (60, 106)
-> [:-2, 4:-4]
-> bicubic, align_corners=False
-> float32 tensor (1, 58, 87)
-> (depth / 2.0) - 0.5
```

## 实现任务

1. 新增硬件无关的`depth_processing.py`。
2. 将视觉节点改为424×240并读取真实depth scale。
3. 检查帧号和相机时间戳单调前进，超时、缺帧或旧帧不发布。
4. 删除手工crop、3米归一化和adaptive average pooling。
5. 保持ROS控制话题兼容，并用32FC1发布米制网络输入预览。
6. 用纯Python单元测试验证形状、无效值和常量归一化。

## 约束与风险

- ROS节点仍按现有架构发布深度而不执行视觉网络。
- 视觉节点测试环境可能没有ROS/RealSense，因此硬件无关部分必须可独立测试。
- 真实相机验收已经在来源仓库完成；本轮只做迁移后的静态与纯函数验证。

## 验证记录

- 2026-07-27：8个聚焦深度测试通过。
- `depth_processing.py`和`visual_extreme_parkour.py`通过Python 3.8语法检查。
- 未连接D435i、ROS图或真实机器人；没有运行控制节点。

## Dry-run输出路由

`resolve_output_topics()`是输出边界的唯一事实来源：

- dry-run：返回随机LowCmd测试话题，两个Sport API话题均为`None`。
- 真实模式：保留`/lowcmd`、`/api/robot_state/request`和
  `/api/sport/request`。
- Sport API发布函数在构建消息前再次检查`self.dryrun`，形成第二道防线。
- 2026-07-27：3个输出路由测试和8个深度处理测试全部通过；
  `unitree_ros2_real.py`等入口通过Python 3.8语法检查。

## Jetson环境收敛

- 复用`~/parkour/parkour_venv`的NVIDIA CUDA Torch，不通过PyPI重装Torch。
- `rsl_rl`以`--no-deps -e`挂载仓库内置版本。
- 删除控制路径中未使用的torchvision和cv2导入，避免为了假依赖改动
  Jetson Torch或引入OpenCV/libgomp冲突。
- `scripts/setup_jetson.sh`只安装内置包并执行完整导入/CUDA验收，不启动
  ROS节点、相机或电机输出。

## 真实接管状态

```text
Sport Mode + model warm-up
-> L1 release/hold gate
-> CheckMode
-> latch fresh measured joint pose
-> create own /lowcmd publisher (publishing still blocked)
-> ReleaseMode (only when active)
-> CheckMode verification
-> authorize real LowCmd publishing
-> publish latched-pose hold immediately
-> 3 s latched-pose startup ramp
-> policy_prime: 0.5 s, >=10 proprio frames, >=5 depth-GRU frames
-> stand_hold
-> Y rising edge
-> policy_engagement: 1 s quintic blend, <=0.05 rad/cycle
-> policy: <=0.10 rad/cycle, continuous input watchdog and output constraints
```

MotionSwitcher RPC每次2秒超时，最多3次。publisher对象可以在ReleaseMode前创建，
但真实发布授权只在CheckMode确认为空后打开。任何RPC或输入新鲜度检查失败时，
预创建publisher不得发送保持命令或退出motor-off。原生publisher endpoint可能因DDS
发现缓存继续显示数秒，该endpoint存在本身不再阻塞接管。

`policy_prime`开始时清零上一动作和episode历史、重置视觉GRU hidden state，并在
默认站姿下重新采样。LowState或深度年龄超过0.25秒时清空本轮样本并重新预热；
遥控器只在Y请求当帧检查0.25秒新鲜度。L2恢复完成后走同一预热路径。接入保护以
最后站姿目标为起点。`last_actions`始终保存上一帧actor原始输出，接入和输出保护
只约束LowCmd目标，不能改写actor历史。

策略动作边界统一执行以下顺序：

```text
raw_action (12 finite values)
-> policy_clip = normalization.clip_actions / control.action_scale
-> clamp to [-policy_clip, +policy_clip]
-> target_q = default_q + clipped_action * control.action_scale
-> policy engagement blend/slew guard
-> continuous target-step, joint-limit, and PD-torque intersection
-> LowCmd joint mapping
```

`self.actions`保存actor原始输出，因为训练环境在延迟和裁剪之前写入动作历史。输出
保护后的执行动作只进入飞行记录。每个policy周期先验证LowState和深度年龄不超过
0.25秒，再将请求目标与上一命令`±0.10 rad`、机械关节范围及由最新`q/dq/Kp/Kd`
推导出的PD力矩可行区间求交。深度超时进入站姿恢复；可行交集为空时恢复轨迹也
无法同时满足约束，因此与LowState超时一样发送motor-off尾帧并锁定在
`emergency_stop`。

运行时使用250周期控制环形缓冲和50帧视觉环形缓冲，不在控制循环写磁盘或打印完整
tensor。L2、R2、异常和退出时将原始/实际动作、请求/下发/实测关节状态、IMU、足端力、
输入年龄、视觉统计和循环耗时写入`--flight-log-dir`下的时间戳NPZ。

实时观测使用`std_msgs/msg/String`话题`/extreme_parkour/runtime_status`，QoS深度为1，
schema version为1。控制记录完成后用单调时钟限制为每0.5秒最多发布一次；JSON包含：

```text
phase, dryrun, real_lowcmd_authorized, engagement_active
input_age_ms(low_state, remote, depth)
body(roll_deg, pitch_deg)
feet(force, contact)
joint(measured_q, commanded_q, measured_dq, max_tracking_error)
policy(max_abs_raw_action, max_request_command_delta)
motor(temperature, lost, tau_est, max_abs_tau_ratio, max_abs_pd_tau_ratio)
loop_ms(last, p50, p95, max, samples)
```

状态构建器是无ROS纯函数，所有数值必须有限，累计`lost`必须为非负整数，力矩限值必须
为正。发布端复用本周期已经取得的LowState、命令和内存环形缓冲，不读取原始深度、不
写磁盘；序列化或publisher异常由观测边界捕获并限频报告，不得传播到控制状态机。
`scripts/read_runtime_status.py`只创建订阅端，默认输出单行摘要，并以transient-local
QoS读取`/rosout`中`unitree_ros2_real`的实时及缓存事件；`--json`输出完整状态JSON，
`--logger-name`可覆盖日志过滤名。

2026-07-28使用ROS 2 Jazzy隔离domain 232联调：合成controller的schema v1状态和
`/rosout` warning均被读取端实时显示。57个Python 3.8测试、相关语法检查和diff检查
通过；Foxy Jetson及真实控制节点话题仍待部署后验证。

2026-07-27静态验证：43个深度、映射、输出隔离、上下文、接入保护、飞行记录和
回放几何测试通过；相关文件通过Python 3.8语法检查和`git diff --check`。新policy
prime、Y接入保护、飞行记录和控制循环p95尚未在Jetson dry-run或真机上验收。

## model_38300关节与接触边界

Isaac内部资产使用`FL/FR/RL/RR`，但训练环境在actor输入和动作边界执行自反重排。
因此所有actor输入、历史观测、默认关节角和动作统一使用：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

Unitree电机顺序也是`FR/FL/RR/RL`，因此actor索引到真机索引的映射为：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

LowState和LowCmd不再做左右腿二次重排，方向系数保持全`+1`；脚端力也保持
`FR/FL/RR/RL`。测试必须引用训练actor契约，并用一热动作验证第0维进入FR电机0，
避免自洽但错误的往返测试和对称站姿掩盖问题。

2026-07-27验证：6个关节映射测试及全套43个单元测试通过。使用真实验收D435记录
的一帧视觉输出和导出actor离线前向时，旧映射与修复后物理电机目标的最大差值为
0.343127 rad、L2差值为0.587423 rad；动作全部有限。控制相关Python文件通过
Python 3.8语法检查，未连接ROS图或真机执行修复后的控制节点。

## 接触观测与Y接入故障修复

训练环境的接触观测语义是：

```text
current = norm(contact_force) > 2
filtered = current OR previous
observation = filtered ? +0.5 : -0.5
previous = current
```

真机端保持actor的`FR/FL/RR/RL`顺序，以原始脚端力`>5`判为当前接触，并使用相同的一帧
逻辑或记忆。policy prime重置历史时，previous从全false开始，禁止用全true掩盖
第一帧实际状态。阈值5是根据本次飞行记录的承重值8～32和明确离地值0确定的部署
参数，不等同于仿真力单位，Jetson验收时必须复核。

Y入口按同一时刻的LowState检查：四足当前接触、`|roll|/|pitch| <= 8°`、
`max(|measured_q-commanded_q|) <= 0.2 rad`。prime开始时锁存12个电机`lost`计数，
恒定的历史非零值允许通过；任一计数增长会更新基线并重新开始完整0.5秒prime。
prime完成后，`stand_hold`在每个50 Hz控制周期继续比较最新计数与上一周期基线；增长
必须在没有Y事件时就自动回到prime。这样一次增长只有在随后连续0.5秒稳定后才重新
进入`stand_hold`，不会在数分钟后按Y时被误判为当下故障。Y事件处理前先执行该观察，
随后仍按同一LowState重复入口检查，因此Y当帧的新增长继续被拒绝。温度字段只要求
可记录，不使用经验温度阈值阻断Y；其他入口检查失败仍回到policy prime，不进入策略。

`PolicyTransitionGuard`将前1秒目标限制在接入起点`±0.3 rad`和0.05 rad/周期，
并用后级实际下发目标作为下一周期基准。完整1秒后固定退出接入态，取消额外目标步长
限制以恢复训练时的动态响应，持续输出保护只执行机械限位和PD力矩约束。运行期速度检查使用URDF标称
上限的0.5%相对测量容差，避免浮点边界和传感器量化导致误停。按用户确认，接入后不使用
roll/pitch、足端接触或温度自动motor-off；
LowState反馈超时仍是硬停止条件。R2保留为人工motor-off急停，L2始终执行原有的
一秒站姿渐变。

飞行记录`format_version=2`，在v1字段上新增`contact_state(4)`、
`motor_temperature(12)`、`motor_lost(12)`和`motor_tau_est(12)`。这些字段只进入内存
环形缓冲，磁盘写入仍只发生在L2、R2、异常或退出路径。

2026-07-27本地验证：19个真实控制安全测试及全套43个单元测试通过；修改后的运行
模块通过Python 3.8语法检查，`git diff --check`通过。尚未同步Jetson，也未连接
ROS图、D435i或真机。

## Unitree边界S2S技术规格

硬件无关的边界模块定义LowState和LowCmd字段等价结构。生产节点从ROS消息提取数组
后调用同一组纯函数，Isaac回放则从PhysX tensor合成相同结构：

```text
Isaac q/dq/contact FL/FR/RL/RR
-> Unitree LowState q/dq/foot FR/FL/RR/RL + gyro + quaternion(wxyz)
-> decode_low_state
-> build_policy_proprio + update_proprio_history
-> exported actor
-> prepare_policy_action
-> PolicyTransitionGuard
-> constrain_policy_target
-> encode_low_cmd
-> Unitree motor target FR/FL/RR/RL
-> Isaac target FL/FR/RL/RR
-> actor-space action accepted by env.step
-> PhysX PD
```

LowCmd回到PhysX时必须同时计算显式Isaac目标和`env.step()`需要的actor动作，并逐帧
断言环境内部重排后的PD目标等于显式Isaac目标，避免回放自身形成自洽的二次重排。
接触使用PhysX足端合力的模作为合成Unitree标量，再执行生产阈值5和一帧逻辑或；
这只验证顺序和滤波逻辑，不代表真机力标定。

回放状态机按50 Hz仿真时间运行：共享`RemoteEdgeTracker`生成并消费L1上升沿，随后
150周期实测位姿到默认姿态的五次渐变，再用25周期默认站姿prime重建本体历史和5次
视觉GRU状态；只有新的Y上升沿通过入口检查后才开始1秒策略接入。LowState每周期更新，
深度每5周期更新，生产0.25秒新鲜度门槛仍执行。L1前虽然PhysX需要内部保持目标，
日志中的LowCmd授权必须保持false。

单箱物理高度场可叠加固定种子的块状均匀噪声，参数为幅度、块尺寸和种子。噪声先写入
地面，箱体顶面随后恢复为精确20 cm；默认幅度为0，实机前鲁棒性验收使用±5 mm、
0.2 m噪声块和种子17。

Viewer中A/B/C分别使用蓝/绿/红车身；C只在回放脚本内部注入旧
`[3,4,5,0,1,2,9,10,11,6,7,8]`关节重排和`[1,0,3,2]`足端重排。绿色B是唯一
边界验收对象。默认NPZ目录为`~/extreme-boundary-s2s`，记录阶段、根状态、原始动作、
请求/下发目标、LowState/LowCmd电机数组、Isaac目标、接触、估算力矩、reset和通过标记。

运行配置使用`EXTREME_BOUNDARY_COMPARISON=abc|fixed`、
`EXTREME_BOUNDARY_GUARDS=full|mapping`、`EXTREME_REPLAY_STEPS`和
`EXTREME_BOUNDARY_LOG_DIR`；不添加ROS依赖，也不发布消息。`mapping`仍保留Y后1秒
接入渐变，但不执行稳态步长/关节/力矩交集，只用于隔离边界，不构成真机验收。

2026-07-28修复前参数扫描发现三个独立问题：0.05 rad稳态限速使绿色边界在箱体前
侧翻；箱体后缘计算多加了1米；到达最后目标后的成功终止被计为意外reset，随后旧环境
硬编码的0～0.9 rad关节reset噪声又制造了二次安全故障。保持0.05 rad接入限速，稳态
使用0.10 rad并把力矩上限收紧到URDF的`23.7/23.7/35.55 Nm`时，绿色边界到达
`x=3.55 m`，越过含0.2米裕量的真实后缘`x=3.40 m`。实现必须修正这些验收语义后
重新运行，不得把成功终止后的reset状态纳入策略安全结论。

2026-07-28修复后，Python 3.8 CPU PhysX的`fixed/full`和`abc/full`均在第426步由
绿色边界越过`x=3.40 m`后正常结束。`fixed/full`的接入/稳态最大目标变化分别为
0.05/0.10 rad，最大估算力矩比例0.8535，无reset、保护故障、机械限位越界或
LowCmd到PhysX目标差异。红色旧映射在入口检查被拒绝，故保持站姿。

同日模拟遥控与粗糙地面验收：L1第5步、Y第183步，绿色边界在±5 mm物理地面噪声
上第422步越过`x=3.40 m`，最大力矩比例0.8812，无reset、fault或目标越界。

同机非headless短跑在创建场景前由Isaac Gym进程以退出码139结束；`DISPLAY=:0`可访问，
RTX 5070 Ti和580.159.03驱动可被`nvidia-smi`识别，CPU headless正常。当前证据只能确认
Viewer入口代码和颜色配置存在，不能确认该旧版Isaac Gym图形栈支持当前GPU/驱动；
可视验收需换到已知兼容的Isaac Gym工作站完成。

共享边界及既有控制链共48个Python 3.8单元测试通过，相关脚本语法检查和
`git diff --check`通过；三个traced资产SHA-256保持不变。

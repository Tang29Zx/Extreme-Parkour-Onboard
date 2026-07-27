# MEMORY.md

## 项目上下文

- 本地路径：`/home/tang/Extreme-Parkour-Onboard`。
- 当前仓库：`Tang29Zx/Extreme-Parkour-Onboard`；原始上游为
  `change-every/Extreme-Parkour-Onboard`，迁移起点为 `master` 提交
  `6f136271c36e3c9ba8c8f9e4e4205f4e16e2d24c`。
- 目标硬件：Unitree Go2、Jetson Orin NX、Intel RealSense D435i。
- 可复用的已验收实现来自 `/home/tang/parkour-go2-deploy`。

## 长期约束

- D435i推理输入固定为`424×240` Z16、30 FPS，不旋转、不镜像、不翻转。
- 必须读取设备`depth_scale`转米；`0`、`65535`、非正值和超过2米的深度按2米处理。
- 先做`4×4`面积降采样得到`60×106`，再执行`[:-2, 4:-4]`和bicubic缩放得到
  `58×87`，最后按`0～2 m`归一化；bicubic之后不增加截断。
- Isaac内部资产顺序是`FL/FR/RL/RR`，但训练环境在actor输入和动作边界使用
  `[3,4,5,0,1,2,9,10,11,6,7,8]`重排。因此`model_38300`的actor契约实际是
  `FR/FL/RR/RL`，与Unitree电机和脚端力一致；LowState、LowCmd和足端力都必须
  identity映射。导出包装器没有内嵌重排。
- 2026-07-27全链路复核推翻了早先“模型顺序是`FL/FR/RL/RR`”的判断。二次重排
  会交换前后腿的左右动作与动态观测；对称默认站姿和映射往返测试无法发现该错误。
  重新同步Jetson并通过dry-run前禁止再次真机接管。
- 不存储凭据，不自动运行真实电机入口。

## 已完成迁移（2026-07-27）

- 新增纯函数`depth_processing.py`，并将视觉节点切换到已验收的
  `424×240 -> 60×106 -> 58×87`、0～2米训练等价路径。
- 视觉节点已删除手工crop、3米归一化、adaptive average pooling、RealSense
  滤波器和无关的`ros2_numpy`/CRC依赖；预览话题现在发布米制`32FC1`网络输入。
- D435i采集会拒绝超时、缺帧、帧号未前进或相机时间戳未前进的数据。
- 8个聚焦测试、Python 3.8语法检查和`git diff --check`通过；尚未在该仓库中
  重新连接D435i或运行ROS节点。

## Dry-run输出边界（2026-07-27）

- dry-run的LowCmd只发布到随机`/lowcmd_dryrun_<id>`，不使用真实`/lowcmd`。
- dry-run不创建Sport Mode和robot-state API发布端，函数边界也会拒绝请求。
- `--nodryrun`保持上游真实话题、模型和策略数学，接管时必须通过真实输出保护。
- 3个输出路由测试通过；未连接ROS图或真实机器人执行控制节点。

## Jetson环境（2026-07-27）

- 现有CUDA虚拟环境位于`~/parkour/parkour_venv`，Unitree ROS消息工作空间位于
  `~/unitree_msgs_ws`，CycloneDDS工作空间位于`~/cyclonedds_ws`。
- 上游`rsl_rl`包含未使用的torchvision导入，控制入口包含未使用的cv2导入；
  已删除两个假依赖，不为此改动Jetson的NVIDIA Torch。
- 使用`scripts/setup_jetson.sh`做一次性内置包安装和环境验收；新终端使用
  `source scripts/jetson_env.sh`。
- Foxy的`setup.bash`会读取可选的未定义变量，因此包装脚本不能在source ROS环境时
  开启Bash `nounset` (`set -u`)。

## 真实接管保护（2026-07-27）

- 真实模式在模型warm-up和L1连续1秒确认前不创建`/lowcmd` publisher；L1确认后、
  ReleaseMode前预创建publisher和消息缓冲，但此时真实发布仍被硬门槛禁止。
- MotionSwitcher使用CheckMode/ReleaseMode/CheckMode确认链；模式已为空时跳过
  ReleaseMode。
- ReleaseMode前锁存新鲜、近似静止的实测关节位姿；最终CheckMode确认为空后
  立即发布该位姿保持命令。原生publisher endpoint的DDS销毁延迟不再阻塞接管，
  消除了已观测到的约5.75秒无有效低层保持命令空窗。
- 接管后从锁存关节位姿用3秒五次平滑过渡到策略默认姿态，再保持0.5秒重建至少
  10帧本体历史和5帧视觉GRU状态；LowState或深度超过0.25秒会重新预热，遥控器
  只在Y请求当帧检查。L2恢复后走同一流程。
- Y后的第一帧严格保持站姿，随后用1秒五次渐变和0.05 rad/周期上限追赶策略目标；
  稳态继续执行0.10 rad/周期、机械限位和PD力矩约束。`last_actions`始终保存上一帧
  actor原始输出，与训练端在延迟和裁剪前写动作历史的语义一致。
- policy每周期检查LowState、深度、关节位置和速度；深度或输出保护失败走站姿恢复，
  LowState超过0.25秒则发送motor-off尾帧并进入硬停止。
- L2只用1秒退回站姿，不在`/lowcmd` publisher存在时恢复Sport Mode；R2和
  进程退出发送10帧motor-off。
- 预创建publisher后，最终CheckMode为空前，普通腿命令和退出motor-off都不得发布。
- 控制节点不再逐帧打印yaw、动作tensor和分段耗时；最近250个控制周期及50次视觉
  更新保存在内存，L2恢复完成、R2关电机后、异常或退出时写入时间戳NPZ。
- 32个测试、Python 3.8语法检查和diff检查通过；新policy prime、Y接入保护、飞行
  记录和Jetson循环p95尚未在Jetson dry-run或真机验收。

## 权重来源

- `traced/`已经替换为`parkour-go2-deploy/models/model_38300`的完整配套包，
  `base_jit.pt`、`vision_weight.pt`和`config.json`必须保持同一版本。
- 三个文件的SHA-256依次为
  `b556db21f30f17e83fcd240b6f0d534b72f255aec42109cb3c223ac8ad756fb4`、
  `897758d6e775419993973fc6f75c096631267b51332d6ed0d1bbb576d9551793`和
  `4056bb5cf389a9827d57d23afc9c4a57aeb3a44c30a62c5ac8500de3ce3a77d7`。
- 目标加载链已验证视觉输出`(1, 34)`、拼接观测`(1, 114)`、动作`(1, 12)`，
  且所有输出有限；真实机器人行为仍需重新同步Jetson并先做dry-run。
- 上游示例权重仍可从Git恢复，不单独保存副本。

## 2026-07-27真机故障诊断

- `~/extreme-control.log`记录约5秒策略运行：216帧动作全部有限，L2在1秒内正常退出，
  没有Python异常、NaN或控制报错。但该版本按错误的`FL/FR/RL/RR`actor假设记录
  和下发数据，因此日志不能证明左右腿映射正确。
- 218个计时样本中p50约15.63 ms、p95约50.89 ms，14帧超过20 ms；首次warm-up
  样本不属于稳定控制周期，运行期仍存在少量超期，需要后续降低逐帧终端输出。
- 停止控制后`/lowcmd` publisher为0。连续1500帧LowState显示RR/RL hip温度约
  `69/68°C`，随后降至`67/66°C`，其余腿约`37～42°C`；RL thigh和RL calf的
  `lost`持续为5且未增长，其余电机为0。
- BMS约31.45 V、95% SOC，电芯约4098～4102 mV，未见明显电池异常。
- 当前证据支持后髋高温或RL历史通信丢失与故障灯相关，但Unitree公开消息文档未
  定义`lost`值语义或温度告警阈值，不能仅凭该字段确定最终故障码。冷却、正常
  重启并采集无自定义控制的基线前，禁止再次真机接管。
- 后续日志确认另一次故障发生在策略启动前：Sport Mode于17:28:46释放，自有
  `/lowcmd`直到17:28:52才创建，约5.75秒空窗内后腿先塌下；随后从塌下位姿恢复
  站姿触发红灯。原生Sport Mode正常且该次没有策略动作，根因转向低层接管时序，
  因此删除“等待原生publisher endpoint消失”门槛，改为预创建但确认释放前禁发。
- 修复接管空窗后的支架测试中，释放确认到锁存位姿首帧仅约5.6 ms且3秒站姿正常；
  用户按Y后发生侧翻，但`~/extreme-control-handoff.log`在`Startup ramp complete`
  后停止，没有记录Y或动作，无法复原该次精确LowCmd。Jetson代码和三个模型资产
  哈希与本地一致，事后LowState约为roll -2.9°、pitch -0.1°、12电机`lost=0`。
- 上一次完整策略日志显示同一模型第一帧相对站姿最大目标跳变约0.453 rad，且左右
  明显不对称；同时warm-up到Y之间的10帧本体历史和视觉GRU状态长期未刷新。因此
  后续真机前必须先通过状态重建、Y短时接入保护和飞行记录的dry-run验收。

## model_38300动作裁剪契约（2026-07-27）

- 训练源码将`normalization.clip_actions=1.2`解释为缩放后的关节残差上限，并用
  `clip_actions / action_scale`计算策略空间阈值；`action_scale=0.25`时策略动作裁到
  `±4.8`，最大关节残差为`±1.2 rad`。
- 训练端PD目标和参考部署均使用
  `target_q = default_q + clipped_action * action_scale`；保护后目标反算动作使用
  `(commanded_q - default_q) / action_scale`。
- 训练环境在动作延迟与裁剪前更新`action_history_buf`，观测时重排回actor顺序；
  因此`self.actions`和`last_actions`必须保存actor原始输出。裁剪后请求和保护后执行
  动作只用于目标计算与飞行记录。

## 导出策略单箱回放坐标（2026-07-27）

- `legged_gym/scripts/replay_traced.py`的CPU深度射线必须基于每个环境实际选择的
  `env_origins`计算箱体世界坐标，不能将箱体固定在第一行赛道的世界坐标。
- 修复前固定seed会让机器人出生在第二行：真实碰撞箱约在`x=20～21.2 m`，视觉箱
  仍在`x=2～3.2 m`，策略因此在`x=19.89 m`卡住。修复后同一1000步回放在第250步
  已到`x=22.16 m`并越过单箱，坐标纯函数测试通过。

## Y后向左侧翻记录与修复（2026-07-27）

- 飞行记录`extreme-flight-20260727-194701-319410.npz`曾被分析为包含Y后43帧接入和
  51帧L2恢复。第一帧目标无跳变、最大单周期变化0.05 rad、动作有限、控制p95约
  8.74 ms、深度年龄最大约40 ms；这些事实排除了明显时序和非有限输出，但记录字段
  按错误映射标记，不能排除左右腿二次重排。
- Y前roll/pitch约`-2.29°/-1.40°`，Unitree原始脚端力按`FR/FL/RR/RL`为
  `[32,22,29,11]`。旧阈值25会把FL和RL写成离地；训练端使用低阈值并将当前接触与
  上一帧逻辑或。该接触错误和关节二次重排同时存在，不能只凭侧翻方向区分贡献。
- 真机接触阈值改为5并增加训练一致的一帧接触记忆。该值来自本次日志承重脚最低
  8～11、离地脚为0的样本，不是Unitree官方标定值，部署前需在目标地面复核。
- policy prime和Y瞬间新增四足承重、8°姿态、0.2 rad关节跟踪门槛；高温按用户
  要求只记录、不阻断。`lost`是累计计数：prime以当前值为基线，只在继续增长时
  重启，不能要求绝对值为0。接入前1秒增加相对起点±0.3 rad包络并使用0.05 rad
  周期限速；稳态使用0.10 rad并持续执行机械限位和PD力矩约束。
- 用户明确不要接入期或L2基于倾斜角自动motor-off：接入后的姿态、支撑和温度只
  记录，R2仍是人工motor-off急停，L2保持原有的一秒站姿渐变。
- 飞行记录升级为v2，新增接触判定、12电机温度、`lost`和估算力矩。19个聚焦测试
  在Python 3.8容器中通过，相关运行文件语法检查和diff检查通过；修改尚未同步Jetson
  或做硬件验收。
- 20:08后的L1站姿连续观测为roll约`-2.42°`、pitch约`-1.60°`，四足力
  `[32,26,30,12]`，最大关节跟踪误差约0.145 rad。RR calf的`lost=5`在496帧内
  恒定且电机正常跟踪，证明绝对`lost=0`门槛会误阻Y。遥控器空闲时不持续发布，
  因此prime只检查LowState/深度，Y事件当帧单独检查遥控器新鲜度。

## sim2real全链路复核与修复（2026-07-27）

- 权威训练源码位于`/home/tang/extreme-parkour-go2`。Isaac DOF顺序是
  `FL/FR/RL/RR`，`step()`先把actor动作重排到Isaac顺序，观测又把关节、动作历史和
  接触重排回`FR/FL/RR/RL`；`save_jit.py`没有在导出模型中加入重排。Onboard原始
  提交`6f13627`的identity映射与该契约一致。
- 修复当前控制端的关节、默认角、动作和足端identity映射。真实验收D435记录的一帧
  视觉输出经`base_jit.pt`前向后，旧二次重排与修复后物理电机目标最大相差
  0.343127 rad，L2相差0.587423 rad，说明该错误足以造成明显左右不对称控制。
- `last_actions`恢复为上一帧actor原始输出。policy每周期增加LowState/深度新鲜度、
  关节位置和URDF速度检查；速度检查保留0.5%相对测量容差。LowState超时硬停止，
  其他保护失败进入站姿恢复。接入首1秒使用0.05 rad/周期，之后以0.10 rad/周期与
  机械限位及PD力矩可行区间求交；交集为空时也必须硬停止。
- 全套43个单元测试通过，相关文件通过Python 3.8语法检查和`git diff --check`；
  导出actor离线前向全部有限。没有运行Jetson dry-run、ROS图或真实机器人。
- 仓库内置`legged_gym`是较旧训练副本，未完整包含生成`model_38300`的动作延迟、
  观测噪声和批量深度实现。它可用于当前回放，但重新训练前必须以权威训练仓库同步
  并重新验证，不能假定内置源码可复现现有权重。
- MEMORY中提到的历史飞行文件当前不在本机仓库；除已有摘要外无法重新核验原始列。
  真机前仍需关注后髋66～69°C和历史`lost`计数，并在承重支架上完成新版本验收。

## Unitree边界S2S（2026-07-28）

- 新增消息无关的LowState/LowCmd边界，生产ROS节点和Isaac回放共用同一套关节、足端、
  IMU、53维proprio和LowCmd转换；Isaac原始顺序与Unitree顺序不再由两个入口重复实现。
- 边界回放默认提供蓝色直连、绿色正确边界、红色旧二次重排三条赛道；红色故障只在
  回放脚本存在。每帧显式验证LowCmd目标经环境内部重排后与PhysX PD目标逐维一致。
- B停机由三个问题叠加造成：0.05 rad被错误地永久用作稳态限速；渐变器按候选目标推进
  而未接收后级实际下发目标反馈；URDF速度上限被当成无浮点/测量容差的硬断言。现改为
  首1秒0.05 rad、稳态0.10 rad，渐变器每周期提交实际下发目标，并使用0.5%速度容差。
- Go2关节和力矩限值集中到边界模块，力矩使用URDF的`23.7/23.7/35.55 Nm`。回放
  关闭旧环境0～0.9 rad reset噪声，箱体通过阈值按真实后缘加0.2米裕量计算为
  `x=3.40 m`，到达目标后的成功终止不再计为意外reset。
- Python 3.8 CPU PhysX的`fixed/full`和`abc/full`都在第426步由绿色B越过箱体后缘；
  单路最大接入/稳态目标变化0.05/0.10 rad、最大估算力矩比例0.8535，无reset、fault、
  机械限位越界或LowCmd到PhysX目标差异。红色C因旧二次重排导致入口跟踪误差超过
  0.2 rad而被拒绝，因此保持站姿，这是预期故障对照而非脚本卡死。
- S2S不验证DDS、CRC、固件、电机通信或真实脚端力尺度；`full`通过只解除映射和控制
  数学阻塞，不构成真机策略接管验收。
- 当前RTX 5070 Ti、580.159.03驱动主机的非headless Isaac Gym在Viewer初始化阶段
  退出139，虽然X Display可访问且CPU headless正常；三色Viewer尚未实际渲染验收，
  需要使用已知兼容Isaac Gym的GPU/驱动环境。
- 修复后全套51个Python 3.8单元测试、相关语法检查和`git diff --check`通过；
  `fixed/full --headless`和`abc/full`验收均为PASS。没有启动ROS、DDS、Jetson
  dry-run或真机输出。离线三路回放保存在
  `~/extreme-boundary-s2s/replays/unitree-boundary-abc-repaired.mp4`。

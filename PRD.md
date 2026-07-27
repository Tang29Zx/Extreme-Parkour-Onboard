# D435i真实深度预处理迁移

## 用户目标

用已在Jetson和真实D435i上验收的深度路径替换仓库原有的`640×480`手工裁剪、
3米归一化和自适应平均池化，同时保持控制节点接收的`/forward_depth_image`
为展平的`58×87`归一化深度。

## 验收标准

- 相机请求`424×240` Z16、30 FPS并读取设备`depth_scale`。
- 原始深度严格按`424×240 -> 60×106 -> 58×87`处理。
- `0/65535/超2米`按2米处理，结果全部有限。
- 相机超时、缺帧、帧号停滞和时间戳停滞均拒绝发布。
- 不旋转、不镜像、不翻转；bicubic后不新增截断。
- `/forward_depth_image`仍包含`58×87=5046`个浮点数。
- 聚焦测试、语法检查和`git diff --check`通过。

## 非目标

- 不修改策略网络内部的关节顺序、模型加载和策略观测数学；只在Unitree消息
  边界执行模型顺序与真机顺序转换。
- 不连接相机或机器人，不运行真实电机命令。

## 回滚

本次修改集中在新增`depth_processing.py`、测试以及`visual_extreme_parkour.py`，
可按聚焦diff逐文件撤销，不影响模型文件。

## Dry-run输出隔离

- 默认dry-run只发布到随机`/lowcmd_dryrun_<id>`话题。
- dry-run不创建`/api/robot_state/request`或`/api/sport/request`发布端。
- dry-run中即使按下遥控器按键，Sport Mode请求也必须在函数边界被拒绝。
- `--nodryrun`保持上游的真实话题和策略数学，但必须经过下述接管保护。

## 真实接管保护

用户目标是保留上游模型、观测、关节顺序和控制增益，只在真实输出边界
增加可验证的安全状态。

验收标准：

- 模型warm-up必须发生在Sport Mode释放前。
- 真实接管要求L1先释放、再连续按住1秒，并要求LowState和遥控器新鲜。
- CheckMode为空时不重复ReleaseMode；ReleaseMode后必须再次CheckMode确认为空。
- L1确认后、ReleaseMode前预创建真实`/lowcmd` publisher和消息缓冲，但在
  CheckMode确认为空前，任何真实LowCmd发布（包括退出时motor-off）都必须被拒绝。
- ReleaseMode前锁存新鲜且近似静止的实测关节角；CheckMode确认为空后立即用该
  位姿发送第一帧保持命令，不等待原生publisher endpoint从DDS图中消失。
- 接管后以锁存位姿为起点，用3秒五次平滑过渡到策略默认站姿。
- 启动站姿和每次L2恢复完成后，必须保持默认站姿至少0.5秒，重建至少10帧
  本体历史和5帧视觉GRU状态；LowState、遥控器或深度超过0.25秒时重新开始预热。
- Y只在预热完成后生效。进入策略的前1秒使用五次渐变，每周期目标变化不超过
  0.05 rad；与请求目标追平后取消保护，稳态策略保持上游目标直通。
- 接入保护期间，策略观测中的上一动作必须反映实际下发目标，而不是未执行的原始动作。
- dry-run使用相同的站姿预热和Y接入状态机，但继续隔离真实LowCmd与Sport API。
- 控制周期不逐帧打印tensor；最近5秒控制和视觉状态保存在内存，L2、R2、异常或
  Ctrl+C时写入NPZ飞行记录。

风险与约束：MotionSwitcher的CheckMode空状态是低层控制权释放的授权事实；DDS
endpoint的销毁时间不再作为授权条件。首次真机验收仍必须使用承重支架。若ReleaseMode
或其后的CheckMode失败，节点不得发布任何真实LowCmd。

回滚方式：恢复“释放后等待外部publisher清空再创建自身publisher”的顺序；该顺序
会重新引入无低层保持命令的空窗，回滚版本不得用于无支撑真机接管。

Y接入保护的回滚边界：可以移除0.5秒状态预热、1秒渐变和0.05 rad限速以恢复上游
直通，但已观测到第一帧约0.453 rad不对称目标跳变并导致支架侧翻；回滚版本不得再次
真机进入policy。

## 策略动作裁剪一致性

`normalization.clip_actions`定义缩放后的关节残差上限。策略输出必须先除以
`control.action_scale`换算策略空间裁剪阈值，再乘缩放系数转换为关节残差：

```text
policy_clip = clip_actions / action_scale
clipped_action = clamp(raw_action, -policy_clip, policy_clip)
target_q = default_q + clipped_action * action_scale
```

验收标准：

- `clip_actions=1.2`、`action_scale=0.25`时，策略动作裁剪上限为`±4.8`，任一关节
  残差不得超过`1.2 rad`。
- 稳态策略的`self.actions`和下一帧`last_actions`必须保存裁剪后动作，不能保存
  原始网络输出。
- Y接入保护期间，`last_actions`继续由实际下发目标按同一加号映射反算；保护追平后
  切换为裁剪后策略动作。
- 飞行记录同时保留原始动作、裁剪后请求目标和保护后的实际动作。
- 非有限动作、非法裁剪上限或非法缩放系数必须在LowCmd发布前拒绝。

非目标：本次不修改策略模型、动作维度、关节顺序、PD增益、机械限位或接入渐变参数。

回滚方式：恢复临时的策略空间`±1.2`裁剪会把关节残差进一步限制为`±0.3 rad`，
但会偏离训练动作分布；该版本只能用于明确接受能力损失的保守诊断。

## model_38300关节映射修复

替换后的`model_38300`沿用本地Parkour训练资产的`FL/FR/RL/RR`顺序，Unitree
LowState/LowCmd协议使用`FR/FL/RR/RL`顺序。上游自带权重使用的恒等映射不再适用。

验收标准：

- 关节状态在LowState输入边界按`[3,4,5,0,1,2,9,10,11,6,7,8]`转换为模型顺序。
- 策略目标在LowCmd输出边界用同一映射转换回Unitree电机索引。
- Unitree脚端力`FR/FL/RR/RL`按`[1,0,3,2]`转换为模型的`FL/FR/RL/RR`。
- hip/thigh/calf内部顺序和全`+1`方向保持不变。
- 用12个互异值验证映射和逆映射，禁止用左右对称站姿作为唯一证据。
- 修复只做离线测试；重新同步Jetson并通过dry-run前禁止真实接管。

回滚方式：恢复`unitree_ros2_real.py`的映射改动并删除纯函数映射模块及其测试；
如果继续使用`model_38300`，回滚后不得运行真机模式。

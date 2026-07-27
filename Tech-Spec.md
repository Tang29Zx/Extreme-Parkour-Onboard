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
-> policy (upstream target passthrough after target convergence)
```

MotionSwitcher RPC每次2秒超时，最多3次。publisher对象可以在ReleaseMode前创建，
但真实发布授权只在CheckMode确认为空后打开。任何RPC或输入新鲜度检查失败时，
预创建publisher不得发送保持命令或退出motor-off。原生publisher endpoint可能因DDS
发现缓存继续显示数秒，该endpoint存在本身不再阻塞接管。

`policy_prime`开始时清零上一动作和episode历史、重置视觉GRU hidden state，并在
默认站姿下重新采样。任一输入年龄超过0.25秒时清空本轮样本并重新预热。L2恢复完成
后走同一预热路径。接入保护以最后站姿目标为起点；保护期间的`last_actions`由实际
下发目标反算，追平策略请求后才切换为裁剪后策略动作。

策略动作边界统一执行以下顺序：

```text
raw_action (12 finite values)
-> policy_clip = normalization.clip_actions / control.action_scale
-> clamp to [-policy_clip, +policy_clip]
-> target_q = default_q + clipped_action * control.action_scale
-> policy engagement blend/slew guard
-> LowCmd joint mapping
```

`self.actions`在稳态保存`clipped_action`。接入保护改变最终目标时，使用
`(commanded_q - default_q) / action_scale`反算实际动作，以保证当前观测和10帧历史
都只包含实际执行语义。纯函数测试覆盖`±4.8`动作上限、`±1.2 rad`残差上限、加号映射
往返和非法输入拒绝。

运行时使用250周期控制环形缓冲和50帧视觉环形缓冲，不在控制循环写磁盘或打印完整
tensor。L2、R2、异常和退出时将原始/实际动作、请求/下发/实测关节状态、IMU、足端力、
输入年龄、视觉统计和循环耗时写入`--flight-log-dir`下的时间戳NPZ。

2026-07-27静态验证：34个深度、映射、输出隔离、上下文、接入保护和飞行记录测试
通过；相关文件通过Python 3.8语法检查和`git diff --check`。新policy prime、Y接入
保护、飞行记录和控制循环p95尚未在Jetson dry-run或真机上验收。

## model_38300关节与接触边界

所有策略输入、历史观测、默认关节角、动作和保护状态统一使用训练/仿真顺序：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

Unitree电机顺序是`FR/FL/RR/RL`，因此模型索引到真机索引的唯一映射为：

```text
[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
```

LowState只在读取边界转换一次，LowCmd只在写入边界转换一次。方向系数保持全
`+1`。脚端力从Unitree的`FR/FL/RR/RL`按`[1,0,3,2]`转换为模型顺序。
纯函数模块必须覆盖互异值往返和脚端力重排测试，避免对称值掩盖左右腿交换。

2026-07-27验证：5个关节映射测试与原有20个深度、输出路由和真实接管测试
全部通过；控制相关Python文件通过Python 3.8语法检查，`git diff --check`通过。
未连接ROS图或真机执行修复后的控制节点。

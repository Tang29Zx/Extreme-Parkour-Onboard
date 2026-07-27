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
-> ReleaseMode (only when active)
-> CheckMode verification
-> external /lowcmd publishers == 0 for 0.5 s
-> create own /lowcmd publisher
-> 3 s measured-pose startup ramp
-> stand_hold
-> Y rising edge
-> 1 s policy engagement ramp + 0.05 rad/cycle slew limit
-> policy
```

MotionSwitcher RPC每次2秒超时，最多3次。任何RPC、输入新鲜度或publisher
独占检查失败时，在创建真实`/lowcmd` publisher前终止进程。

2026-07-27静态验证：9个真实接管纯函数测试、3个输出路由测试和8个
深度处理测试全部通过；两个控制入口通过Python 3.8语法检查。尚未
在Jetson ROS图上执行MotionSwitcher或创建真实`/lowcmd` publisher。

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

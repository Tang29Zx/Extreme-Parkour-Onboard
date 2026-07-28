# Deployment Code of Extreme Parkour on Unitree Go2

This repository provides an **unofficial implementation** for deploying the project [Extreme Parkour with Legged Robots](https://github.com/chengxuxin/extreme-parkour) on the **Unitree Go2** quadrupted robot. The original work was developed for A1 robots and does not provide the deployment code. 

## Key Contributions

- Add detailed comments throughout the training code. Documented the previously unexplained **observation** vector.

- ~~Add camera randomization during training to accout for Go2's movable camera (unlike A1's fixed camera).~~

- Provide train weights and deployment code for Unitree Go2.

## Deployment Instructions

#### Environment Setup
Make sure the environment is properly set up on your Go2 robot, including rclpy, torch and unitree sdk.

Jetson上首次同步仓库后执行一次：

```bash
cd ~/Extreme-Parkour-Onboard
bash scripts/setup_jetson.sh
```

该脚本不从网络安装Torch或torchvision，只将仓库内置`rsl_rl`挂载到已有
`~/parkour/parkour_venv`，并一次性检查CUDA、ROS、Unitree消息、RealSense、CRC和
控制/视觉入口。此后每个新终端先执行：

```bash
cd ~/Extreme-Parkour-Onboard
source scripts/jetson_env.sh
```

#### Hardware Setup
Install the **Intel RealSense D435i** depth camera on the Go2. Verify that the captured images resemble the simulation (can be checked using `rviz`).

#### Deployment Steps
1. Connect to the Go2 robot wirelessly via SSH (wired connection is also ok).
```bash
ssh unitree@<go2_ip_address>
```

2. In the first terminal, start the visual node:
```bash
python3 visual_extreme_parkour.py --logdir traced
```

该节点使用已经过真机验收的固定深度路径：D435i以`424×240`、Z16、30 FPS采集，
使用设备`depth_scale`转成米制深度，将`0`、`65535`和超过2米的值置为2米，
再执行`4×4`面积降采样、训练裁剪和bicubic缩放，最终向
`/forward_depth_image`发布展平的`58×87`归一化深度。该路径不旋转、不镜像、
不翻转，且bicubic之后不额外截断。`/camera/forward_depth`同步发布米制
`32FC1`网络输入用于检查。

3. In a second terminal, start the controller node:
```bash
python3 run_extreme_parkour.py --logdir traced --mode parkour --nodryrun
```

真实模式只能在承重支架上首次验收。启动时节点先加载并warm-up模型，
此时不发布`/lowcmd`。按日志提示先释放L1，再连续按住L1 1秒；节点会在
ReleaseMode前预创建自己的publisher并锁存实测关节角，但保持真实发布禁用。
CheckMode确认Sport Mode已释放后，节点立即发送锁存位姿保持命令，不等待
原生publisher endpoint从DDS图中消失。

`traced/`中的`model_38300`在Isaac内部使用`FL/FR/RL/RR`，但训练环境在actor边界
已经重排为`FR/FL/RR/RL`。该顺序与Unitree电机和脚端力一致，控制节点不得再次
交换左右腿。更换其他权重时必须从其训练源码和导出包装器确认actor边界，不能只凭
对称站姿或映射往返判断兼容。

- 接管后先等待3秒站姿渐变，再等待0.5秒本体历史和视觉GRU预热；看到
  `Policy prime complete`后才可按Y。
- policy prime和按Y瞬间都会确认四足承重、roll/pitch不超过8°、最大关节跟踪
  误差不超过0.2 rad且12个电机`lost`计数没有继续增长；历史非零计数允许通过。
  prime只要求LowState和深度持续新鲜，按Y瞬间才检查遥控器。不满足时保持站姿并
  打印拒绝原因；电机温度只写入飞行记录，不阻止进入策略。
- prime完成后等待Y期间仍以50 Hz观察累计`lost`计数；发现增长会立即自动重新prime，
  连续稳定0.5秒后重新接受Y，不会把几分钟前已经稳定的增长延迟到按Y时才拒绝。
- 短按并释放 **Y** 进入策略；前1秒以五次曲线接入且每周期最多变化0.05 rad，
  第1秒相对起点还限制在0.3 rad以内。接入完成后以0.10 rad/周期持续限制目标变化、机械
  关节范围和估算PD力矩；策略观测中的`last_actions`保持上一帧actor原始输出。
  接入后不根据姿态自动motor-off。
- policy运行中每周期检查LowState、深度、关节位置和速度。深度过期或状态越界时
  进入站姿恢复；LowState超过0.25秒没有更新，或不存在同时满足步长、关节和力矩
  约束的目标时执行motor-off硬停止，因此必须先在承重支架验收这些故障路径。
- 短按 **L2** 退出策略并用1秒回到站姿；不会在`/lowcmd` publisher
  存在时自动恢复Sport Mode。L2不会根据倾斜角自动motor-off。
- **R2** 是低层接管后的电机关闭急停，机器狗会失去支撑力。
- `Ctrl+C`退出时也会发送10帧motor-off命令，因此只能在支架承重时停止。
- `--flight-log-dir`可指定飞行记录目录，默认`~/extreme-flight-logs`；L2、R2、
  异常或退出时会保存最近5秒数据并在日志中打印文件路径。记录包含12电机温度、
  `lost`、估算力矩和四足接触判定。

控制节点运行后，可在第三个终端只读查看2 Hz实时状态：

```bash
cd ~/Extreme-Parkour-Onboard
source scripts/jetson_env.sh
python3 scripts/read_runtime_status.py
```

输出同时包含`unitree_ros2_real`的实时`/rosout`事件；状态单行包含控制阶段、真实
输出授权、策略接入状态、LowState/遥控器/深度年龄、机身roll/pitch、四足接触、
关节跟踪误差、最大温度、`lost`计数、力矩比例和控制周期。使用`--json`查看完整
12关节数组。该脚本只订阅
`/extreme_parkour/runtime_status`和`/rosout`，不创建控制publisher，也不代替退出时
保存的NPZ飞行记录。控制节点使用其他logger名称时通过`--logger-name`指定。

## Notes and Tips

#### Policy selection:
Modify in `run_extreme_parkour.py`:
```bash
base_model = 'your_base_model.pth'
vision_model = 'your_vision_model.pth'
```

#### Walk Mode:
It is recommended to test the walk mode to verify your model and camera setup:
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk --nodryrun
```
Use ``--mode walk`` or ``--mode parkour`` to switch between **walking** and **parkour** mode, as they were trained as separate tasks in the original work. You can aslo use
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk
```

不加`--nodryrun`时，节点只向随机的`/lowcmd_dryrun_<id>`话题发布测试
消息，不创建Sport Mode API发布端；遥控器按键只切换节点内部状态，
不会调用机器狗的Sport Mode或真实`/lowcmd`。短按L1后，dry-run会执行与
真机相同的3秒站姿、0.5秒policy prime和Y接入保护，便于先验收状态机。

## Unitree边界S2S回放

`replay_unitree_boundary.py`在Isaac Gym中合成LowState并接收LowCmd等价数据，
不会启动ROS、DDS或真机输出。默认Viewer显示三条相同赛道：蓝色是actor直连基准，
绿色经过修复后的完整Unitree边界，红色注入旧版左右腿二次重排故障。

该Viewer只能在安装Isaac Gym的x86_64 Ubuntu NVIDIA工作站运行，不能在Jetson或
RViz中查看。旧版Isaac Gym可能不兼容RTX 50系和较新驱动；若非headless在Viewer
初始化阶段退出139，应换用已验证的Isaac Gym显卡/驱动环境，headless结果不能替代
视觉验收。示例环境使用本机Python 3.8 Isaac Gym配置：

```bash
cd /home/tang/Extreme-Parkour-Onboard
export PATH=/home/tang/miniconda3/pkgs/ninja-1.13.2-h171cf75_0/bin:$PATH
export LD_LIBRARY_PATH=/home/tang/miniconda3/envs/hybrik/lib
export PYTHONPATH=/home/tang/extreme-parkour-go2/isaacgym/python:$PWD:$PWD/rsl_rl
export PARKOUR_RESOURCES_DIR=/home/tang/extreme-parkour-go2/legged_gym/resources

EXTREME_REPLAY_STEPS=1000 \
EXTREME_BOUNDARY_COMPARISON=abc \
EXTREME_BOUNDARY_GUARDS=full \
EXTREME_GROUND_NOISE_M=0.005 \
EXTREME_GROUND_NOISE_PATCH_M=0.2 \
EXTREME_GROUND_NOISE_SEED=17 \
/home/tang/miniconda3/envs/hybrik/bin/python \
  legged_gym/scripts/replay_unitree_boundary.py \
  --task go2 --device cpu --pipeline cpu
```

加入`--headless`执行无窗口验收；使用`EXTREME_BOUNDARY_COMPARISON=fixed`只运行绿色
边界。`EXTREME_BOUNDARY_GUARDS=mapping`保留1秒接入渐变但关闭稳态输出约束，只用于
隔离顺序、观测和LowCmd往返问题，不能代表真机安全链。默认`full`会执行生产端每周期
输入检查、接入期0.05 rad/周期、稳态0.10 rad/周期、机械限位和估算PD力矩约束。
关节速度检查对URDF标称上限保留0.5%测量容差，超出该容差仍会停止策略。
回放使用与生产端相同的遥控器上升沿逻辑，自动注入一次L1和一次Y；L1前不授权
LowCmd，Y只能在3秒站姿和0.5秒prime完成后生效。`EXTREME_GROUND_NOISE_M`控制
物理高度场的对称噪声幅度，噪声块尺寸和随机种子分别由
`EXTREME_GROUND_NOISE_PATCH_M`、`EXTREME_GROUND_NOISE_SEED`设置。

每次运行都会在`~/extreme-boundary-s2s`保存无pickle NPZ；可用
`EXTREME_BOUNDARY_LOG_DIR`覆盖目录。日志同时包含actor、Unitree motor和Isaac顺序、
原始动作、请求/下发目标、接触、估算力矩、reset、保护故障和最终验收结果。完整验收
失败时脚本以非零状态退出，而不会把reset或保护硬停隐藏为成功。

2026-07-28的Python 3.8 CPU PhysX验收中，绿色B在第426步越过箱体后缘
`x=3.40 m`，结果为PASS；接入期最大目标变化0.05 rad、稳态0.10 rad、最大估算
力矩比例0.8535，且无reset、保护故障或LowCmd/PhysX目标差异。三路`abc/full`回放也在
第426步由绿色B完成，红色C因旧二次重排造成入口跟踪误差而保持站姿。

2026-07-28进一步使用±5 mm、0.2 m噪声块和种子17运行模拟遥控完整链：L1在第5步
接管，第183步Y通过，绿色边界第422步到达`x=3.4076 m`。结果为PASS，0 reset、
0 fault，接入/稳态最大目标变化0.05/0.10 rad，最大估算力矩比例0.8812。

to perform a more conservative test. This command runs the policy without sending actions to the motors — useful for verifying perception and inference without physical movement.

## Performance 
The Go2 is capable of climbing over obstacles up to **40 cm** in height. Video will be provided soon.

## Acknowledgments
This repository is based on modification of [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour). Special thanks to the original authors for their open-source contribution.

## Contact
I am a beginner in robotics, and warmly welcome feedback and contributions to improve this repository. For questions, suggestions or collaboration, please open an issue or contact me directly.

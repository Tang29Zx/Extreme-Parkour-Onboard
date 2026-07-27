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

`traced/`中的`model_38300`策略包使用训练顺序`FL/FR/RL/RR`；节点在LowState/LowCmd
边界与Unitree的`FR/FL/RR/RL`顺序互换，并同步重排脚端力。更换其他权重时
必须重新确认其训练关节顺序，不能仅凭对称站姿判断兼容。

- 接管后先等待3秒站姿渐变，再等待0.5秒本体历史和视觉GRU预热；看到
  `Policy prime complete`后才可按Y。
- 短按并释放 **Y** 进入策略；前1秒以五次曲线接入且每周期最多变化0.05 rad，
  与策略目标追平后恢复上游目标直通。
- 短按 **L2** 退出策略并用1秒回到站姿；不会在`/lowcmd` publisher
  存在时自动恢复Sport Mode。
- **R2** 是低层接管后的电机关闭急停，机器狗会失去支撑力。
- `Ctrl+C`退出时也会发送10帧motor-off命令，因此只能在支架承重时停止。
- `--flight-log-dir`可指定飞行记录目录，默认`~/extreme-flight-logs`；L2、R2、
  异常或退出时会保存最近5秒数据并在日志中打印文件路径。

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
to perform a more conservative test. This command runs the policy without sending actions to the motors — useful for verifying perception and inference without physical movement.

## Performance 
The Go2 is capable of climbing over obstacles up to **40 cm** in height. Video will be provided soon.

## Acknowledgments
This repository is based on modification of [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour). Special thanks to the original authors for their open-source contribution.

## Contact
I am a beginner in robotics, and warmly welcome feedback and contributions to improve this repository. For questions, suggestions or collaboration, please open an issue or contact me directly.

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
- 上游自带权重使用仓库当前的`FR/FL/RR/RL`恒等映射，但替换后的
  `model_38300`来自本地Parkour训练链，模型/仿真顺序是`FL/FR/RL/RR`。
  当前代码已在LowState和LowCmd边界使用
  `dof_map=[3,4,5,0,1,2,9,10,11,6,7,8]`，脚端力也必须从Unitree
  `FR/FL/RR/RL`按`[1,0,3,2]`重排为`FL/FR/RL/RR`。
- 2026-07-27首次替换`model_38300`时遗漏上述映射：默认站姿仍可正常，但策略
  观测和动作残差会交换`FL/FR`及`RL/RR`。该版本曾触发真机异常/故障灯；
  修复后25个纯函数测试通过，重新同步Jetson并通过dry-run前禁止再次真机接管。
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

- 真实模式在模型warm-up和L1连续1秒确认前不创建`/lowcmd` publisher。
- MotionSwitcher使用CheckMode/ReleaseMode/CheckMode确认链；模式已为空时跳过
  ReleaseMode。
- 外部`/lowcmd` publisher必须连续0.5秒为零；15秒超时时不创建自己的
  publisher，日志会列出剩余endpoint。
- 接管后从实测关节位姿用3秒五次平滑过渡到策略默认姿态。Y只在
  `stand_hold`后生效，策略目标再用1秒渐变和0.05 rad/周期限速接入。
- L2只用1秒退回站姿，不在`/lowcmd` publisher存在时恢复Sport Mode；R2和
  进程退出发送10帧motor-off。
- 9个真实接管纯函数测试通过；尚未在Jetson上运行MotionSwitcher接管链或真实
  `/lowcmd`。

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

- `~/extreme-control.log`记录修复映射后的约5秒策略运行：216帧动作全部有限，
  LowState/LowCmd映射为`FL/FR/RL/RR -> FR/FL/RR/RL`，换算后的请求关节目标
  位于配置关节范围内；L2在1秒内正常退出，没有Python异常、NaN或控制报错。
- 218个计时样本中p50约15.63 ms、p95约50.89 ms，14帧超过20 ms；首次warm-up
  样本不属于稳定控制周期，运行期仍存在少量超期，需要后续降低逐帧终端输出。
- 停止控制后`/lowcmd` publisher为0。连续1500帧LowState显示RR/RL hip温度约
  `69/68°C`，随后降至`67/66°C`，其余腿约`37～42°C`；RL thigh和RL calf的
  `lost`持续为5且未增长，其余电机为0。
- BMS约31.45 V、95% SOC，电芯约4098～4102 mV，未见明显电池异常。
- 当前证据支持后髋高温或RL历史通信丢失与故障灯相关，但Unitree公开消息文档未
  定义`lost`值语义或温度告警阈值，不能仅凭该字段确定最终故障码。冷却、正常
  重启并采集无自定义控制的基线前，禁止再次真机接管。

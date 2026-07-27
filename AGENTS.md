# AGENTS.md

## 项目说明

- 项目：Extreme Parkour Onboard Go2
- 技术栈：Python 3.8、ROS 2 Foxy、Unitree ROS 2、RealSense、PyTorch
- 当前目标：将已验收的 Go2 D435i 深度预处理迁入现有 Onboard 推理链路。

## 开发约定

- 沟通和文档使用中文；代码、标识符和代码注释使用英文。
- 真机命令默认 dry-run；未经用户明确授权，不执行 `--nodryrun`。
- 模型、相机录制、日志和二进制权重不自动修改或提交。
- 修改保持聚焦，不重构无关训练代码。

## 验证

- 语法：`python -m py_compile depth_processing.py visual_extreme_parkour.py`
- 聚焦测试：`python -m unittest tests.test_depth_processing -v`
- 输出路由：`python -m unittest tests.test_output_routing -v`
- 真实接管纯函数：`python -m unittest tests.test_real_control_safety -v`
- 关节边界：`python -m unittest tests.test_joint_mapping -v`
- Jetson环境：`bash scripts/setup_jetson.sh`（仅在Jetson上执行）
- Diff：`git diff --check`

## Git

- 修改前后检查 `git status --short` 和聚焦 diff。
- 不自动提交或推送。
- 不使用破坏性 reset、clean、rebase 或强制推送。

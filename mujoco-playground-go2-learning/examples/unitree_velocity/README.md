# Go2 / Go1 速度指令跟踪例程

本例在 MuJoCo Playground 源码上增加独立的速度跟踪任务，默认复用 Go2 MJX 模型，学习停止、前进、后退、左转、右转。它不是已训练好的步态控制器。

先读 [手把手教程](../../docs/四足速度跟踪手把手教程.md)，再用 [结构说明](../../docs/环境与配置结构说明.md) 查阅配置和源码。

## 文件与入口

| 文件 | 负责什么 |
| --- | --- |
| `environment.py` | Go2 模型适配、指令采样和观测同步；Go1 备用任务 |
| `run.py` | 检查、默认姿态仿真、训练、评估和键盘交互 |
| `training_metrics.py` | 完整重置后保留终止回合的奖励和超时记录 |
| `test_*.py` | 指令、重置、奖励统计、进度和显示格式的检查 |

所有命令从**项目根目录**执行，而不是本文件所在目录。

## 在新机器准备环境

本机已经安装完成时，无需重装，直接激活现有 `.venv`。

新机器建议使用 Linux、Python 3.12 和可用的 NVIDIA 驱动。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，克隆共享仓库后进入项目：

```bash
git clone https://github.com/302992/-.git ~/github-shared
cd ~/github-shared/mujoco-playground-go2-learning
uv --no-config sync --locked --all-extras
source .venv/bin/activate
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

`uv.lock` 保存本机使用的版本；同步会联网安装依赖，CUDA 训练应看到 `gpu`。不必先用 `pip -U` 升级整套依赖。视频输出还需要系统提供 ffmpeg 和可用的 EGL 图形环境；桌面查看器使用 GLFW。

## 准备外部模型

模型网格不随这个教学项目上传。本机已经通过软链接使用 `~/mujoco_models/mujoco_menagerie`。

在新机器上，可以让 Playground 加载内置运动任务以获取其固定版本的 Menagerie：

```bash
python -c "from mujoco_playground import registry; registry.load('Go1JoystickFlatTerrain', config_overrides={'impl': 'jax'})"
python -m examples.unitree_velocity.run inspect --robot go2
```

首次运行会访问 GitHub 下载资产。模型的默认位置是 `mujoco_playground/external_deps/mujoco_menagerie/unitree_go2/`。

也可使用自己已有的模型目录，显式指定：

```bash
python -m examples.unitree_velocity.run inspect \
  --robot go2 \
  --model-dir "$HOME/mujoco_models/mujoco_menagerie/unitree_go2"
```

该目录需要包含 `go2_mjx.xml`、`scene_mjx.xml` 和 `assets/`。采用显式模型目录时，也在 `stand` 和首次 `train` 命令中传入同样的 `--model-dir`；恢复和评估会读取实验保存的实际模型路径，可用参数覆盖位置。

模型来源：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)。Go2 适配在内存中进行，采用平地脚部接触简化；详细限制见教程。

## 常用操作

```bash
python -m examples.unitree_velocity.run train --help
python -m examples.unitree_velocity.run inspect --robot go2
python -m examples.unitree_velocity.run stand --robot go2
```

训练命令、完整参数解释、当前/总轮次、训练与评估奖励区别，请按 [教程第 5、6 步](../../docs/四足速度跟踪手把手教程.md) 操作。学习配置默认 1024 个训练环境；显存、吞吐量与学习效果需分别观察。

## 自动检查

快速检查（不需要 GPU 或机器人模型）：

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  examples/unitree_velocity/test_progress.py \
  examples/unitree_velocity/test_training_metrics.py
```

包含本地 Go1/Go2 模型的集成检查：

```bash
JAX_PLATFORMS=cpu python -m pytest -q examples/unitree_velocity
```

缺少对应机器人模型时，模型集成检查会跳过。已验证版本与历史 GPU 试跑结果见教程末尾；本机的日志、视频和训练检查点保存在 `logs/`，不包含在仓库中。

## 来源与许可证

上游项目为 [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)，本副本基线提交 `d59156e`。保留上游源码、[Apache-2.0 许可证](../../LICENSE) 和引用说明。外部机器人模型遵循各自目录的许可证。

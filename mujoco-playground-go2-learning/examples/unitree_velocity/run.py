"""Guided Unitree velocity experiment. Run with python -m examples.unitree_velocity.run."""

# Environment variables must be set before importing JAX / MuJoCo.
# ruff: noqa: E402

import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import threading
import time
from decimal import ROUND_DOWN, Decimal, localcontext
from pathlib import Path

# Set these before importing JAX / MuJoCo. The native viewer needs GLFW instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from .environment import COMMANDS, command_vector, make_env
from .training_metrics import wrap_for_training


def training_wrapper(env, **kwargs):
    """Fully reset tasks and preserve terminal rewards/timeout diagnostics."""
    return wrap_for_training(env, **kwargs)


def model_digest(env):
    """Identify the effective XML and external assets independently of location."""
    digest = hashlib.sha256()
    xml = (
        env._adapted_xml.encode("utf-8")
        if hasattr(env, "_adapted_xml")
        else Path(env.xml_path).read_bytes()
    )
    for payload in (
        xml,
        *(
            item
            for name, data in sorted(env._model_assets.items())
            for item in (name.encode("utf-8"), data)
        ),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def require_gpu():
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "未检测到 JAX GPU。先检查 nvidia-smi、项目 .venv 和运行权限；"
            "CPU 上不启动四足 PPO 训练。inspect/stand 可用 CPU。"
        )
    print("JAX devices:", jax.devices(), flush=True)


def make_network(env, config):
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(config["value_hidden_layer_sizes"]),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )


def latest_checkpoint(path):
    """Accept a run, checkpoints parent, or concrete numeric checkpoint."""
    path = Path(path).expanduser().resolve()
    if (path / "ppo_network_config.json").is_file():
        return path
    if (path / "checkpoints").is_dir():
        path = path / "checkpoints"
    candidates = sorted(
        (
            p
            for p in path.iterdir()
            if p.is_dir()
            and p.name.isdigit()
            and (p / "ppo_network_config.json").is_file()
        ),
        key=lambda p: int(p.name),
    )
    if not candidates:
        raise ValueError(f"没有完整的数字检查点：{path}")
    return candidates[-1]


def inspect(args):
    env = make_env(args.robot, args.model_dir)
    model = env.mj_model
    print("Robot:", args.robot, " | backend:", jax.default_backend())
    print("XML:", env.xml_path)
    print("nq / nv / nu:", model.nq, model.nv, model.nu)
    print("Observation:", env.observation_size)
    print("ctrl_dt / sim_dt:", env.dt, env.sim_dt)
    print(
        "Actuator order:",
        [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ],
    )
    for name in COMMANDS:
        print(name, np.asarray(command_vector(name)))
    print("模型加载通过。模型文件不包含已经学会走路的策略。")


def stand(args):
    env = make_env(args.robot, args.model_dir)
    state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    state = env.set_command(state, command_vector("stop"))
    step = jax.jit(env.step)
    trajectory = []
    for i in range(args.steps):
        # Zero action means home joint targets, not zero actuator torque.
        state = step(state, jp.zeros(env.action_size))
        if not np.isfinite(np.asarray(state.data.qpos)).all():
            raise RuntimeError("物理状态出现 NaN/Inf")
        if i % 2 == 0:
            trajectory.append(state)
        if bool(state.done):
            print("发生任务终止；此处仅测试默认关节目标，不是已训练策略。")
            break
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    import mediapy  # noqa: PLC0415

    frames = env.render(trajectory, height=360, width=640, camera="track")
    mediapy.write_video(output, frames, fps=1.0 / env.dt / 2)
    print("视频:", output)


def training_plan(config):
    """Match Brax's sampling-update count, including evaluation-stage rounding.

    One update round collects a new batch then runs PPO's minibatch updates.
    It is neither one episode nor one individual optimizer/minibatch step.
    """
    steps_per_update = (
        config["batch_size"]
        * config["num_minibatches"]
        * config["unroll_length"]
        * config.get("action_repeat", 1)
    )
    stages = max(config["num_evals"] - 1, 1)
    resets = max(config.get("num_resets_per_eval", 0), 1)
    updates_per_stage = (
        math.ceil(config["num_timesteps"] / (stages * steps_per_update * resets))
        * resets
    )
    total_updates = stages * updates_per_stage
    return dict(
        requested_steps=config["num_timesteps"],
        steps_per_update=steps_per_update,
        updates_per_stage=updates_per_stage,
        total_updates=total_updates,
        actual_steps=total_updates * steps_per_update,
    )


def truncate_reward(value):
    """Display six decimal places, truncating toward zero rather than rounding."""
    if value is None:
        return "暂无已完成回合"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    number = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(28, number.adjusted() + 8)
        number = number.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    if number == 0:
        number = abs(number)  # Avoid a distracting -0.000000.
    return format(number, ".6f")


class TrainingProgress:
    """Write raw metrics and print actual Brax callbacks with monotone progress."""

    def __init__(self, plan, run_dir, writer, clock=time.monotonic):
        self.plan = plan
        self.run_dir = Path(run_dir)
        self.writer = writer
        self.clock = clock
        self.started = clock()
        self.latest_step = 0
        self.lock = threading.Lock()

    def __call__(self, num_steps, metrics):
        num_steps = int(num_steps)
        # JAX callbacks can execute on a host callback thread. Serialize writes
        # and retain even delayed metrics, while never printing older progress.
        with self.lock:
            raw = {key: float(value) for key, value in metrics.items()}
            for key, value in raw.items():
                self.writer.add_scalar(key, value, num_steps)
            self.writer.flush()
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({"steps": num_steps, **raw}) + "\n")
            if num_steps < self.latest_step:
                return
            self.latest_step = num_steps
            completed = min(
                num_steps // self.plan["steps_per_update"], self.plan["total_updates"]
            )
            percent = min(100.0, 100 * num_steps / self.plan["actual_steps"])
            if "eval/episode_reward" in raw:
                phase = "初始评估" if num_steps == 0 else "阶段评估"
                reward = "eval_reward=" + truncate_reward(raw["eval/episode_reward"])
            else:
                phase = "采样更新"
                reward = "train_reward=" + truncate_reward(
                    raw.get("episode/sum_reward")
                )
            print(
                f"[{phase}] 轮次={completed}/{self.plan['total_updates']} "
                f"| 进度={percent:6.2f}% "
                f"| steps={num_steps:,}/{self.plan['actual_steps']:,} "
                f"| {reward} | 已用={self.clock() - self.started:.1f}s",
                flush=True,
            )


def train(args):
    require_gpu()
    restore = None
    previous = None
    selected_model_dir = args.model_dir
    if args.restore:
        restore = latest_checkpoint(args.restore)
        previous = json.loads((restore.parent.parent / "experiment.json").read_text())
        if previous["robot"] != args.robot:
            raise ValueError("不能把其他机器人的检查点用于本实验")
        if selected_model_dir is None:
            selected_model_dir = previous["model_dir"]
    env = make_env(args.robot, selected_model_dir)
    asset_digest = model_digest(env)
    config = dict(
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        num_evals=args.num_evals,
        num_resets_per_eval=0,
        learning_rate=3e-4,
        discounting=0.97,
        entropy_cost=0.01,
        reward_scaling=1.0,
        normalize_observations=True,
        max_grad_norm=1.0,
        seed=args.seed,
        deterministic_eval=True,
        log_training_metrics=True,
    )
    if config["batch_size"] * config["num_minibatches"] % config["num_envs"]:
        raise ValueError("batch_size × num_minibatches 必须能被 num_envs 整除")
    plan = training_plan(config)
    config["training_metrics_steps"] = plan["steps_per_update"] * args.log_every_updates
    network_config = dict(
        policy_hidden_layer_sizes=[128, 128, 128],
        value_hidden_layer_sizes=[256, 256, 256],
    )
    if previous is not None:
        if (
            previous["network"] != network_config
            or previous["environment"] != env._config.to_dict()
        ):
            raise ValueError("恢复时环境或网络配置与原实验不同；先恢复原设置或从头训练")
        if previous["model_digest"] != asset_digest:
            raise ValueError("恢复时模型 XML 或资产与原实验不同；请使用原模型文件")
    run_dir = Path(args.run_dir).resolve()
    # A new directory is intentional: avoid silently overwriting an experiment.
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        robot=args.robot,
        model_dir=str(env.model_dir) if hasattr(env, "model_dir") else None,
        model_digest=asset_digest,
        environment=env._config.to_dict(),
        ppo=config,
        progress=plan,
        network=network_config,
        restore=str(restore) if restore else None,
        versions={
            name: importlib.metadata.version(name)
            for name in ("playground", "mujoco", "mujoco-mjx", "jax", "brax")
        },
    )
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    from tensorboardX import SummaryWriter  # noqa: PLC0415

    writer = SummaryWriter(str(run_dir))
    print("实验目录:", run_dir, flush=True)
    print(
        f"采样预算={plan['requested_steps']:,}步；按评估分段和整批采样取整后 "
        f"预计={plan['actual_steps']:,}步。\n"
        f"每轮新采样={plan['steps_per_update']:,}步；总采样更新轮数="
        f"{plan['total_updates']}；每{args.log_every_updates}轮打印训练状态。",
        flush=True,
    )
    print(
        "先编译并进行初始评估，再编译采样/更新循环；首次可能数分钟没有新轮次。\n"
        "train_reward是最近已完成训练回合的平均累计奖励；"
        "eval_reward是独立评估奖励，两者分开显示。",
        flush=True,
    )
    progress = TrainingProgress(plan, run_dir, writer)

    try:
        ppo.train(
            environment=env,
            eval_env=make_env(args.robot, selected_model_dir),
            wrap_env_fn=training_wrapper,
            network_factory=make_network(env, network_config),
            save_checkpoint_path=str(run_dir / "checkpoints"),
            restore_checkpoint_path=str(restore) if restore else None,
            progress_fn=progress,
            **config,
        )
    finally:
        # Drain asynchronous metric callbacks before closing their output files.
        jax.effects_barrier()
        writer.close()
    print("训练已结束。下一步使用 evaluate 检查每种速度指令，不自动生成长视频。")


def load_experiment(args):
    ckpt = latest_checkpoint(args.checkpoint)
    run_dir = ckpt.parent.parent
    manifest = json.loads((run_dir / "experiment.json").read_text())
    env = make_env(manifest["robot"], args.model_dir or manifest["model_dir"])
    if env._config.to_dict() != manifest["environment"]:
        raise ValueError("环境配置已变化，与检查点不一致；恢复原设置后再评估")
    if model_digest(env) != manifest["model_digest"]:
        raise ValueError("模型 XML 或资产已变化，与检查点不一致；请使用原模型文件")
    # Rebuild our known network; do not use Brax 0.14.2 load_policy's config reader.
    make_policy, params, _ = ppo.train(
        environment=env,
        num_timesteps=0,
        num_envs=1,
        batch_size=1,
        num_minibatches=1,
        episode_length=manifest["ppo"]["episode_length"],
        normalize_observations=manifest["ppo"]["normalize_observations"],
        network_factory=make_network(env, manifest["network"]),
        wrap_env_fn=training_wrapper,
        restore_checkpoint_path=str(ckpt),
    )
    return env, jax.jit(make_policy(params, deterministic=True)), ckpt


def evaluate(args):
    env, policy, ckpt = load_experiment(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    set_command = jax.jit(env.set_command)
    summaries = []
    for name in args.commands:
        rows, frames, errors = [], [], []
        completed = 0
        for episode in range(args.episodes):
            rng = jax.random.PRNGKey(args.seed + episode)
            state = reset(rng)
            target = command_vector(name, args.speed, args.yaw_rate)
            state = set_command(state, target)
            for t in range(args.steps):
                rng, key = jax.random.split(rng)
                action, _ = policy(state.obs, key)
                state = step(state, action)
                velocity = np.asarray(env.get_local_linvel(state.data))
                yaw = float(env.get_gyro(state.data)[2])
                if not np.isfinite(
                    np.r_[np.asarray(state.data.qpos), velocity, yaw]
                ).all():
                    raise RuntimeError("评估中出现 NaN/Inf；检查模型、控制器和训练过程")
                row = dict(
                    episode=episode,
                    time=(t + 1) * env.dt,
                    target_vx=float(target[0]),
                    target_vy=float(target[1]),
                    target_wz=float(target[2]),
                    vx=float(velocity[0]),
                    vy=float(velocity[1]),
                    wz=yaw,
                    done=float(state.done),
                )
                rows.append(row)
                # Discard initial 1 s transient from velocity error metrics.
                if (t + 1) * env.dt > 1.0:
                    errors.append(
                        np.abs(np.asarray(target) - [velocity[0], velocity[1], yaw])
                    )
                if args.video and episode == 0 and t % 2 == 0:
                    frames.append(state)
                if bool(state.done):
                    break
            completed += int(not bool(state.done) and t + 1 == args.steps)
        with (output / f"{name}.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        mean_error = np.mean(errors, axis=0).tolist() if errors else [None] * 3
        summary = dict(
            command=name,
            episodes=args.episodes,
            completed=completed,
            requested_seconds=args.steps * env.dt,
            mean_abs_error_vx_vy_wz=mean_error,
        )
        summaries.append(summary)
        print(summary, flush=True)
        if args.video and frames:
            import mediapy  # noqa: PLC0415

            images = env.render(frames, height=360, width=640, camera="track")
            mediapy.write_video(output / f"{name}.mp4", images, fps=1 / env.dt / 2)
    (output / "summary.json").write_text(
        json.dumps(dict(checkpoint=str(ckpt), results=summaries), indent=2),
        encoding="utf-8",
    )
    print("评估文件:", output)


def play(args):
    """Optional local desktop viewer; command changes apply immediately to observations."""
    env, policy, _ = load_experiment(args)
    import mujoco.viewer  # noqa: PLC0415

    selected = ["stop"]
    lock = threading.Lock()
    mapping = {87: "forward", 83: "backward", 65: "left", 68: "right", 32: "stop"}

    def key_callback(keycode):
        if keycode in mapping:
            with lock:
                selected[0] = mapping[keycode]
            print("command:", selected[0], flush=True)

    data = mujoco.MjData(env.mj_model)
    rng = jax.random.PRNGKey(args.seed)
    reset, step, set_command = (
        jax.jit(env.reset),
        jax.jit(env.step),
        jax.jit(env.set_command),
    )
    state = reset(rng)
    print(
        "点击查看器窗口：W前进 / S后退 / A左转 / D右转 / 空格停止；指令保持到下一次按键。"
    )
    with mujoco.viewer.launch_passive(
        env.mj_model, data, key_callback=key_callback
    ) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env.mj_model.camera("track").id
        while viewer.is_running():
            started = time.monotonic()
            with lock:
                name = selected[0]
            state = set_command(state, command_vector(name, args.speed, args.yaw_rate))
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            state = step(state, action)
            data.qpos[:] = np.asarray(state.data.qpos)
            data.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_forward(env.mj_model, data)
            viewer.sync()
            if bool(state.done):
                rng, key = jax.random.split(rng)
                state = reset(key)
                print("翻倒后重置。", flush=True)
            time.sleep(max(0, env.dt - (time.monotonic() - started)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Go2/Go1 速度指令跟踪教学实验")
    subs = parser.add_subparsers(dest="mode", required=True)
    descriptions = {
        "inspect": "检查模型、维度和速度指令，不进行训练",
        "stand": "默认关节目标仿真并录视频，不更新策略",
        "train": "收集仿真经验并用 PPO 更新策略、保存日志和检查点",
        "evaluate": "加载已有策略，分别测量固定指令的速度误差",
        "play": "加载策略，在本机图形窗口使用键盘切换指令",
    }
    for mode, description in descriptions.items():
        sub = subs.add_parser(
            mode,
            help=description,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub.add_argument(
            "--model-dir",
            help="Go2 模型子目录，包含 go2_mjx.xml、scene_mjx.xml 与 assets",
        )
        sub.add_argument(
            "--seed",
            type=int,
            default=1,
            help="随机种子，用于初始化和采样；做对照实验时保持一致",
        )
        if mode in ("inspect", "stand", "train"):
            sub.add_argument(
                "--robot",
                choices=["go2", "go1"],
                default="go2",
                help="本次实验使用的机器人模型",
            )
        if mode == "stand":
            sub.add_argument(
                "--steps",
                type=int,
                default=100,
                help="单个环境的最大控制步数；每步模拟0.02秒",
            )
            sub.add_argument(
                "--output",
                default="logs/tutorial_go2/stand.mp4",
                help="生成的 MP4 文件路径",
            )
        if mode == "train":
            sub.add_argument(
                "--run-dir",
                required=True,
                help="本次新实验目录，必须尚不存在，避免覆盖旧结果",
            )
            sub.add_argument(
                "--restore",
                help="加载本例已有实验/检查点参数继续训练，优化器和步数重新开始",
            )
            parameters = {
                "num_timesteps": (
                    5_000_000,
                    "所有并行环境合计的采样预算，按完整更新轮向上取整",
                ),
                "num_envs": (1024, "同时模拟的训练机器人数量；提高它通常增加显存需求"),
                "num_eval_envs": (32, "每次独立评估同时模拟的环境数，不用于梯度更新"),
                "episode_length": (
                    1000,
                    "一个回合的最大控制步数，1000步约20秒模拟时间",
                ),
                "batch_size": (
                    256,
                    "每个 PPO 小批次的轨迹段数，每段长度由 unroll-length 指定",
                ),
                "num_minibatches": (
                    4,
                    "每轮采集的新经验划为几个小批次；与batch-size共同约束并行数",
                ),
                "unroll_length": (20, "每段连续采样轨迹的控制步数"),
                "num_updates_per_batch": (
                    4,
                    "同一批新经验重复优化几遍，每遍遍历所有小批次",
                ),
                "num_evals": (
                    5,
                    "独立评估总次数，包含训练前初始评估；影响评估/保存分段",
                ),
                "log_every_updates": (
                    1,
                    "每多少个采样更新轮打印一次训练状态；调大可减少日志开销",
                ),
            }
            for key, (default, help_text) in parameters.items():
                sub.add_argument(
                    "--" + key.replace("_", "-"),
                    type=int,
                    default=default,
                    help=help_text,
                )
        if mode in ("evaluate", "play"):
            sub.add_argument(
                "--checkpoint",
                required=True,
                help="本例实验目录、checkpoints父目录或具体数字检查点",
            )
            sub.add_argument(
                "--speed",
                type=float,
                default=0.5,
                help="前进/后退目标速率的绝对值，单位米/秒",
            )
            sub.add_argument(
                "--yaw-rate",
                type=float,
                default=0.6,
                help="左转/右转目标角速率的绝对值，单位弧度/秒",
            )
        if mode == "evaluate":
            sub.add_argument(
                "--output",
                required=True,
                help="新评估输出目录，保存CSV/summary及可选视频，必须尚不存在",
            )
            sub.add_argument(
                "--commands",
                choices=list(COMMANDS),
                nargs="+",
                default=list(COMMANDS),
                help="逐项固定输入并测试的速度指令",
            )
            sub.add_argument(
                "--steps",
                type=int,
                default=250,
                help="每次测试最多控制步数；250步约5秒模拟时间",
            )
            sub.add_argument(
                "--episodes",
                type=int,
                default=3,
                help="每种指令用多少次不同初始化重复测试",
            )
            sub.add_argument(
                "--video",
                action="store_true",
                help="为每种指令第一段测试导出视频；不改变策略输入",
            )
        if mode == "play":
            sub.add_argument(
                "--viewer",
                action="store_true",
                required=True,
                help="打开本机桌面交互窗口，需要GLFW及图形会话",
            )
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "episodes",
        "num_timesteps",
        "num_envs",
        "num_eval_envs",
        "episode_length",
        "batch_size",
        "num_minibatches",
        "unroll_length",
        "num_updates_per_batch",
        "log_every_updates",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if hasattr(args, "num_evals") and args.num_evals < 2:
        parser.error("训练至少保留两次评估：--num-evals 2")
    if args.mode == "train" and args.batch_size * args.num_minibatches % args.num_envs:
        parser.error("batch-size × num-minibatches 必须能被 num-envs 整除")
    for name in ("speed", "yaw_rate"):
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                parser.error(f"--{name.replace('_', '-')} 必须为有限非负数")
    return args


def main():
    args = parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()

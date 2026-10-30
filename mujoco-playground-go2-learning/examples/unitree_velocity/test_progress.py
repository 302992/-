"""Fast checks for progress accounting and terminal-only reward formatting."""

import json
import math

import pytest

from .run import TrainingProgress, parse_args, training_plan, truncate_reward


class RecordingWriter:
    def __init__(self):
        self.values = []

    def add_scalar(self, key, value, step):
        self.values.append((key, value, step))

    def flush(self):
        pass


def test_plan_matches_brax_rounding_across_evaluation_stages():
    config = dict(
        num_timesteps=5_000_000,
        num_evals=5,
        batch_size=256,
        num_minibatches=4,
        unroll_length=20,
        num_resets_per_eval=0,
    )
    plan = training_plan(config)
    assert plan == dict(
        requested_steps=5_000_000,
        steps_per_update=20_480,
        updates_per_stage=62,
        total_updates=248,
        actual_steps=5_079_040,
    )
    # Even a small request needs a whole round in every evaluation stage.
    config.update(num_timesteps=1, num_evals=4)
    assert training_plan(config)["total_updates"] == 3
    # Match the reset repetitions supported by Brax's epoch calculation.
    config.update(num_timesteps=100_000, num_evals=3, num_resets_per_eval=2)
    assert training_plan(config)["total_updates"] == 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.123456789, "1.123456"),
        (-1.123456789, "-1.123456"),
        (1.99999999, "1.999999"),
        (1.000001, "1.000001"),
        (-0.0000001, "0.000000"),
        (0, "0.000000"),
        (None, "暂无已完成回合"),
        (math.nan, "nan"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (1e30, "1000000000000000000000000000000.000000"),
    ],
)
def test_reward_is_truncated_not_rounded(value, expected):
    assert truncate_reward(value) == expected


def test_progress_separates_sampling_and_evaluation_without_losing_raw_metrics(
    tmp_path, capsys
):
    writer = RecordingWriter()
    plan = training_plan(
        dict(
            num_timesteps=2048,
            num_evals=2,
            batch_size=32,
            num_minibatches=2,
            unroll_length=8,
        )
    )
    progress = TrainingProgress(plan, tmp_path, writer, clock=lambda: 10.0)
    progress(0, {"eval/episode_reward": 1.123456789})
    progress(512, {"episode/sps": 12345.987654321})
    progress(1024, {"episode/sum_reward": 0.0})
    progress(2048, {"eval/episode_reward": -2.123456789})
    # A delayed callback still reaches TB/JSONL but cannot rewind console progress.
    progress(512, {"episode/sum_reward": 3.987654321})
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 4
    assert "[初始评估]" in output[0] and "轮次=0/4" in output[0]
    assert "eval_reward=1.123456" in output[0]
    assert "[采样更新]" in output[1] and "轮次=1/4" in output[1]
    assert "train_reward=暂无已完成回合" in output[1]
    assert "train_reward=0.000000" in output[2]
    assert "[阶段评估]" in output[3] and "轮次=4/4" in output[3]
    assert "100.00%" in output[3] and "eval_reward=-2.123456" in output[3]
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5
    assert rows[0]["eval/episode_reward"] == 1.123456789
    assert rows[-1]["episode/sum_reward"] == 3.987654321
    assert ("episode/sps", 12345.987654321, 512) in writer.values


def test_cli_larger_default_is_valid_and_help_explains_parameters(capsys):
    args = parse_args(["train", "--run-dir", "logs/new"])
    assert (args.num_envs, args.batch_size, args.num_minibatches) == (1024, 256, 4)
    assert args.num_eval_envs == 32
    assert args.log_every_updates == 1
    with pytest.raises(SystemExit) as exc:
        parse_args(["train", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "轨迹段数" in help_text
    assert "每多少个采样更新轮" in help_text
    assert "独立评估总次数" in help_text


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--run-dir", "logs/new", "--log-every-updates", "0"],
        ["train", "--run-dir", "logs/new", "--num-envs", "2048"],
        [
            "evaluate",
            "--checkpoint",
            "logs/old",
            "--output",
            "logs/eval",
            "--speed",
            "nan",
        ],
    ],
)
def test_cli_rejects_invalid_progress_or_batch_settings(arguments):
    with pytest.raises(SystemExit) as exc:
        parse_args(arguments)
    assert exc.value.code == 2

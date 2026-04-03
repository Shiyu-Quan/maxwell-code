#!/usr/bin/env python3

"""
无 Maxwell server 场景下的离线 smoke test。

目标：
1. 编译 C++ 目标（headless）。
2. 运行 test_sync.py，验证 Python->C++ 同步链路。
3. 运行一个最小 config 启动，验证 config 解析与无 server 失败路径。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAXLAB_LIB = REPO_ROOT / "maxlab_lib"
CPP_EXECUTABLE = MAXLAB_LIB / "build" / "maxone_with_filter"
TEST_SYNC_SCRIPT = MAXLAB_LIB / "closedloop" / "test_sync.py"


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
    return result


def build_cpp_headless() -> None:
    run_cmd(["make", "USE_QT=0", "maxone_with_filter"], cwd=MAXLAB_LIB)
    if not CPP_EXECUTABLE.exists():
        raise RuntimeError(f"Missing executable after build: {CPP_EXECUTABLE}")


def run_sync_test() -> None:
    run_cmd([sys.executable, str(TEST_SYNC_SCRIPT)], cwd=REPO_ROOT)


def run_minimal_config_test() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        config_path = Path(f.name)
        payload = {
            "target_well": 0,
            "read_window_ms": 200,
            "training_window_ms": 300,
            "show_gui": False,
            "wait_for_sync": False,
            "channel_count": 1024,
            "experiment_duration_s": 1.0,
            "cycle_duration_s": 900.0,
            "rest_duration_s": 0.0,
            "encoding_scale_a": 7.0,
            "encoding_scale_b": 0.15,
            "ema_alpha": 0.2,
            "force_scale_n": 10.0,
            "sample_rate_hz": 20000.0,
            "threshold_multiplier": 3.0,
            "min_threshold": -20.0,
            "refractory_samples": 1000,
            "decoding_left_channels": [0],
            "decoding_right_channels": [1],
            "encoding_left_sequence": "encode_left_pulse",
            "encoding_right_sequence": "encode_right_pulse",
            "training_pattern_names": [],
            "log_path": "/tmp/cartpole_no_hw_smoke_episodes.jsonl",
            "random_seed": 12345,
            "mode": "continuous_adaptive",
        }
        json.dump(payload, f)

    result = run_cmd([str(CPP_EXECUTABLE), str(config_path)], cwd=MAXLAB_LIB, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    if "No connection to the mxwserver" not in output and "Cannot establish connection!" not in output:
        raise RuntimeError("Minimal config test did not hit expected no-server error path")
    print("[OK] Minimal config test observed expected no-server behavior")


def main() -> int:
    print("=" * 70)
    print("NO-HARDWARE SMOKE TEST")
    print("=" * 70)
    build_cpp_headless()
    run_sync_test()
    run_minimal_config_test()
    print("=" * 70)
    print("[DONE] All no-hardware checks passed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

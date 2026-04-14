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

import numpy as np


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
    # Avoid stale Qt-enabled object files causing link mismatch in headless build.
    run_cmd(["make", "clean"], cwd=MAXLAB_LIB, check=False)
    run_cmd(["make", "USE_QT=0", "maxone_with_filter"], cwd=MAXLAB_LIB)
    if not CPP_EXECUTABLE.exists():
        raise RuntimeError(f"Missing executable after build: {CPP_EXECUTABLE}")


def run_sync_test() -> None:
    result = run_cmd([sys.executable, str(TEST_SYNC_SCRIPT)], cwd=REPO_ROOT, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode == 0:
        print("[OK] Sync test completed with success exit code")
        return
    if "[C++] Process is ready and waiting for sync signal" in output and "[C++] Start signal sent successfully" in output:
        print("[OK] Sync test reached sync milestones (non-zero exit tolerated in no-hardware mode)")
        return
    raise RuntimeError("Sync test did not reach expected synchronization milestones")


def run_selection_config_compat_test() -> None:
    closedloop_dir = REPO_ROOT / "maxlab_lib" / "closedloop"
    if str(closedloop_dir) not in sys.path:
        sys.path.insert(0, str(closedloop_dir))

    import cartpole_selected_setup as selected_setup  # pylint: disable=import-outside-toplevel
    import cartpole_setup as cartpole_setup_module  # pylint: disable=import-outside-toplevel

    base_selection = {
        "selection_config": {
            "encoding_stim_electrodes": [3344, 3388],
            "training_stim_electrodes": [3432, 3476],
            "decoding_left_electrodes": [13215],
            "decoding_right_electrodes": [14115],
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        base_path = Path(f.name)
        json.dump(base_selection, f)
    loaded_base = selected_setup.load_selection_config(base_path)
    if loaded_base["recording_electrodes"] != list(cartpole_setup_module.RECORDING_ELECTRODES):
        raise RuntimeError("Legacy selection config should fallback to default recording pool")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        custom_path = Path(f.name)
        custom_selection = dict(base_selection)
        custom_selection["selection_config"] = dict(base_selection["selection_config"])
        custom_selection["selection_config"]["recording_electrodes"] = [10, 20, 30, 40]
        json.dump(custom_selection, f)
    loaded_custom = selected_setup.load_selection_config(custom_path)
    if loaded_custom["recording_electrodes"] != [10, 20, 30, 40]:
        raise RuntimeError("Selection config explicit recording_electrodes was not preserved")

    print("[OK] Selection config compatibility test passed")


def run_preexperiment_contract_test() -> None:
    preexp_script = REPO_ROOT / "maxlab_lib" / "closedloop" / "cartpole_preexperiment.py"
    result = run_cmd([sys.executable, str(preexp_script), "full", "--help"], cwd=REPO_ROOT, check=False)
    help_text = f"{result.stdout}\n{result.stderr}"
    if "--scan-budget" not in help_text:
        raise RuntimeError("preexperiment full help missing --scan-budget")
    if "--recording-strategy" in help_text:
        raise RuntimeError("preexperiment full help unexpectedly still exposes --recording-strategy")

    closedloop_dir = REPO_ROOT / "maxlab_lib" / "closedloop"
    if str(closedloop_dir) not in sys.path:
        sys.path.insert(0, str(closedloop_dir))
    import cartpole_preexperiment as preexp  # pylint: disable=import-outside-toplevel

    record_analysis_payload = {
        "record_analysis": {
            "downstream_recording_electrodes": [101, 102, 103],
            "locked_recording_electrodes": [201, 202],
            "eta_ranked_units_top32": [{"electrode": 101, "eta_score": 1.23}],
            "putative_units": [{"electrode": 101, "channel": 1}],
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = Path(f.name)
        json.dump(record_analysis_payload, f)

    resolved_pool = preexp._resolve_recording_pool_for_stimulate(path)  # pylint: disable=protected-access
    if resolved_pool != [101, 102, 103]:
        raise RuntimeError("Expected downstream_recording_electrodes to be prioritized")
    print("[OK] Preexperiment contract test passed")


def run_star_alignment_contract_test() -> None:
    closedloop_dir = REPO_ROOT / "maxlab_lib" / "closedloop"
    if str(closedloop_dir) not in sys.path:
        sys.path.insert(0, str(closedloop_dir))

    import cartpole_selection as selection  # pylint: disable=import-outside-toplevel

    # P0 contract: first-order window defaults to 0-10ms.
    selection_src = (closedloop_dir / "cartpole_selection.py").read_text(encoding="utf-8")
    if "first_order_window_ms: Tuple[float, float] = (0.0, 10.0)" not in selection_src:
        raise RuntimeError("Expected cartpole_selection first_order_window_ms default to be 0-10ms")

    # P0 contract: training trigger should be strict mean5 < mean20.
    training_src = (closedloop_dir / "training_controller.cpp").read_text(encoding="utf-8")
    if "if (decision.mean_5 >= decision.mean_20)" not in training_src:
        raise RuntimeError("Expected strict training trigger guard: mean_5 >= mean_20 returns no stimulation")

    # P0 contract: EMA should follow rt = 0.2 * ct + 0.8 * rt-1 (with alpha=0.2).
    loop_src = (closedloop_dir / "maxone_with_filter.cpp").read_text(encoding="utf-8")
    if "(1.0 - config.ema_alpha) * state.left_rate + config.ema_alpha * left_count" not in loop_src:
        raise RuntimeError("Expected EMA update formula aligned with STAR methods")

    # P0 contract: C1/Cm summary and burst exclusion behavior.
    trace = np.zeros(15000, dtype=np.float64)
    threshold = -1.0
    event_indices = np.asarray([100, 5200, 10300], dtype=np.int64)
    # event 0: first hit + 2 multi spikes
    trace[150] = -5.0
    trace[420] = -5.0
    trace[760] = -5.0
    # event 1: burst trial candidate (should be excluded from Cm)
    trace[5400] = -5.0
    trace[5500] = -5.0
    # event 2: first hit + no multi spike
    trace[10315] = -5.0
    burst_events = [False, True, False]
    summary = selection._summarize_target_connectivity(  # pylint: disable=protected-access
        trace=trace,
        event_sample_indices=event_indices,
        threshold=threshold,
        first_start=0,
        first_end=200,
        multi_start=200,
        multi_end=4000,
        burst_events=burst_events,
    )
    if abs(float(summary["first_order_probability"]) - (2.0 / 3.0)) > 1e-6:
        raise RuntimeError("Unexpected C1 probability in STAR alignment test")
    if abs(float(summary["multi_order_spike_count_mean"]) - 1.0) > 1e-6:
        raise RuntimeError("Unexpected Cm mean spike count (non-burst) in STAR alignment test")
    if int(summary["nonburst_event_count"]) != 2:
        raise RuntimeError("Expected exactly 2 non-burst events in STAR alignment test")

    # SALPA contract: event filtering should prioritize stimulus_probe labels.
    event_dtype = np.dtype(
        [
            ("frameno", "<i8"),
            ("eventtype", "<u4"),
            ("eventid", "<u4"),
            ("eventmessage", "O"),
        ]
    )
    events = np.asarray(
        [
            (100, 0, 1, "noise_event"),
            (200, 0, 2, "stimulus_probe"),
            (300, 0, 3, "stimulus_probe_2"),
        ],
        dtype=event_dtype,
    )
    event_frames, event_source = selection._extract_stimulus_event_frame_numbers(events)  # pylint: disable=protected-access
    if event_source != "stimulus_probe_filtered":
        raise RuntimeError("Expected stimulus_probe_filtered event source when labels are present")
    if event_frames.tolist() != [200, 300]:
        raise RuntimeError("Unexpected filtered event frames for stimulus_probe labels")

    # SALPA contract: artifact window amplitude should decrease while preserving a true negative spike.
    sample_rate_hz = 20000.0
    trace_salpa = np.random.normal(0.0, 0.2, size=5000).astype(np.float64)
    event_samples = np.asarray([1000], dtype=np.int64)
    trace_salpa[1000:1020] += 10.0  # stimulation artifact in first 1ms
    trace_salpa[1060] = -4.0  # spike at +3ms
    salpa_trace, salpa_stats = selection._apply_event_aligned_salpa(  # pylint: disable=protected-access
        trace=trace_salpa,
        event_sample_indices=event_samples,
        sample_rate_hz=sample_rate_hz,
    )
    pre_artifact = float(np.mean(np.abs(trace_salpa[1000:1020])))
    post_artifact = float(np.mean(np.abs(salpa_trace[1000:1020])))
    if not (post_artifact < pre_artifact):
        raise RuntimeError("SALPA did not reduce artifact window amplitude in synthetic test")
    if int(salpa_stats["fit_success_count"]) < 1:
        raise RuntimeError("Expected at least one successful SALPA fit in synthetic test")

    sigma = max(float(np.std(salpa_trace)), 1e-9)
    salpa_threshold = -3.0 * sigma
    recovered_spike_count = selection._count_negative_spikes_in_window(  # pylint: disable=protected-access
        trace=salpa_trace,
        start_index=1000,
        end_index=1200,
        threshold=salpa_threshold,
        refractory_samples=20,
    )
    if recovered_spike_count < 1:
        raise RuntimeError("Expected 3σ post-SALPA detection to recover synthetic spike")

    if "artifact_removal" not in selection_src:
        raise RuntimeError("Expected stimulate_analysis payload to include artifact_removal metadata")
    if "salpa_stats" not in selection_src:
        raise RuntimeError("Expected probe summaries to include salpa_stats metadata")

    print("[OK] STAR alignment contract test passed")


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
    if "No connection to the mxwserver" in output or "Cannot establish connection!" in output:
        print("[OK] Minimal config test observed expected no-server behavior")
        return
    if result.returncode == 0 and "[INFO] cartpole loop config=" in output:
        print("[OK] Minimal config test verified config parsing and runtime startup path")
        return
    raise RuntimeError("Minimal config test did not hit expected startup/no-server paths")


def main() -> int:
    print("=" * 70)
    print("NO-HARDWARE SMOKE TEST")
    print("=" * 70)
    build_cpp_headless()
    run_sync_test()
    run_preexperiment_contract_test()
    run_star_alignment_contract_test()
    run_selection_config_compat_test()
    run_minimal_config_test()
    print("=" * 70)
    print("[DONE] All no-hardware checks passed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

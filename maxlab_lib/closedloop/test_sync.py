#!/usr/bin/env python3

"""
测试 Python 启动 C++ 的同步通信功能（不涉及 MaxLab 硬件）。
适配当前 maxone_with_filter 的 config.json 接口。
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

CPP_EXECUTABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/maxone_with_filter",
)


def build_sync_test_config(log_path: str) -> dict:
    return {
        "target_well": 0,
        "read_window_ms": 200,
        "training_window_ms": 300,
        "show_gui": False,
        "wait_for_sync": True,
        "channel_count": 1024,
        "experiment_duration_s": 3600.0,
        "num_cycles": 1,
        "active_cycle_duration_s": 900.0,
        "rest_duration_s": 2700.0,
        "cycle_condition_schedule": ["adaptive"],
        "first_cycle_pass_threshold_s": 10.0,
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
        "log_path": log_path,
        "random_seed": 12345,
        "mode": "continuous_adaptive",
    }


class CPPProcessManager:
    """管理 C++ 进程并观测同步日志。"""

    def __init__(self, executable: str, config_path: str):
        self.executable = executable
        self.config_path = config_path
        self.process = None
        self.output_thread = None
        self.running = False
        self.ready_event = threading.Event()
        self.start_ack_event = threading.Event()
        self.lines = []

    def start(self):
        print(f"[C++] Starting: {self.executable}")
        print(f"[C++] Config: {self.config_path}")
        self.process = subprocess.Popen(
            [self.executable, self.config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.running = True
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()
        self._wait_for_ready()

    def _read_output(self):
        ready_marker = "[SYNC] Waiting for start signal"
        start_ack_marker = "[SYNC] Start signal received"
        while self.running and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    stripped = line.strip()
                    self.lines.append(stripped)
                    print(f"[C++ OUT] {stripped}")
                    if ready_marker in line and not self.ready_event.is_set():
                        self.ready_event.set()
                    if start_ack_marker in line and not self.start_ack_event.is_set():
                        self.start_ack_event.set()
            except Exception:
                break

    def _wait_for_ready(self):
        print("[C++] Waiting for process to be ready...")
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout and not self.ready_event.is_set():
            time.sleep(0.1)

        if self.ready_event.is_set():
            print("[C++] Process is ready and waiting for sync signal")
            return

        print(f"[C++] Warning: Did not see ready marker within {timeout}s")
        print("[C++] Captured output so far:")
        for line in self.lines[-8:]:
            print(f"  {line}")

    def send_start_signal(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("C++ process is not running")

        print("[C++] Sending 'start' signal...")
        self.process.stdin.write("start\n")
        self.process.stdin.flush()
        print("[C++] Start signal sent successfully")

    def wait_for_exit(self, timeout=10):
        if self.process is None:
            raise RuntimeError("C++ process is not running")
        return self.process.wait(timeout=timeout)

    def stop(self):
        if self.process is None:
            return

        print("[C++] Stopping process...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=5)
            print(f"[C++] Process exited with code {self.process.returncode}")
        except subprocess.TimeoutExpired:
            print("[C++] Process did not exit gracefully, force killing...")
            self.process.kill()
            self.process.wait()
            print("[C++] Process was killed")

        self.running = False
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1)

        self.process = None


def write_temp_config() -> str:
    fd, path = tempfile.mkstemp(prefix="cartpole_sync_test_", suffix=".json")
    os.close(fd)
    payload = build_sync_test_config(log_path="/tmp/cartpole_sync_test_episodes.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def run_test():
    print("=" * 60)
    print("PYTHON-C++ SYNC TEST (NO HARDWARE)")
    print("=" * 60)

    cpp_manager = None
    config_path = None
    try:
        config_path = write_temp_config()
        print(f"[TEST] Generated config: {config_path}")

        print("\n=== STEP 1: Starting C++ Process ===")
        cpp_manager = CPPProcessManager(CPP_EXECUTABLE, config_path)
        cpp_manager.start()
        if not cpp_manager.ready_event.is_set():
            raise RuntimeError("Did not observe sync ready marker")

        print("\n=== STEP 2: Simulating Hardware Configuration ===")
        print("[TEST] Waiting 2 seconds...")
        time.sleep(2)

        print("\n=== STEP 3: Sending Start Signal ===")
        cpp_manager.send_start_signal()
        if not cpp_manager.start_ack_event.wait(timeout=2):
            raise RuntimeError("Did not observe start-ack marker from C++")

        print("\n=== STEP 4: Waiting for expected no-server failure ===")
        exit_code = cpp_manager.wait_for_exit(timeout=8)
        print(f"[TEST] C++ exited with code {exit_code}")

        combined_output = "\n".join(cpp_manager.lines)
        expected_markers = [
            "No connection to the mxwserver",
            "Cannot establish connection!",
            "Exception:",
        ]
        if not any(marker in combined_output for marker in expected_markers):
            raise RuntimeError("C++ did not report expected no-server failure marker")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY (expected no-server failure observed)")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n\n[TEST] Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\n[TEST] Cleaning up...")
        if cpp_manager is not None:
            cpp_manager.stop()
        if config_path and os.path.exists(config_path):
            os.remove(config_path)
        print("[TEST] Cleanup complete")


if __name__ == "__main__":
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"ERROR: C++ executable not found: {CPP_EXECUTABLE}")
        print("Please build the C++ program first:")
        print("  cd maxlab_lib && make USE_QT=0 maxone_with_filter")
        sys.exit(1)

    exit_code = run_test()
    sys.exit(exit_code)

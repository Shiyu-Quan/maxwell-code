#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import random
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import maxlab as mx


ALL_STIM_ELECTRODES = [3344, 3388, 3432, 3476, 6644, 6688, 6732, 6776]
ENCODING_STIM_ELECTRODES = ALL_STIM_ELECTRODES[:2]
TRAINING_STIM_ELECTRODES = ALL_STIM_ELECTRODES[2:]

CPP_EXECUTABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/maxone_with_filter",
)
RECORDING_DIR = Path.home() / "cartpole_experiments"
RECORDING_DIR.mkdir(parents=True, exist_ok=True)

READ_WINDOW_MS = 200
TRAINING_WINDOW_MS = 300
CYCLE_DURATION_S = 15 * 60
REST_DURATION_S = 45 * 60
FIRST_CYCLE_PASS_THRESHOLD_S = 10.0
RANDOM_TRAINING_WINDOW_MS = 200
ADAPTIVE_TRAINING_WINDOW_MS = 300
PAPER_MODE_CHOICES = ("cycled", "continuous_adaptive", "null", "random", "adaptive")

STIM_PARAMS = {
    # STAR methods aligned: 400 uV peak-to-peak biphasic pulse,
    # i.e. +/-200 uV per phase.
    "pulse_amplitude_mV": 0.2,
    "phase_us": 200,
    "inter_pulse_interval_ms": 10,
    "training_frequency_hz": 10,
}

event_counter = 0


def generate_vertical_electrodes(base_electrode: int, num_rows: int = 10, row_offset: int = 220) -> List[int]:
    return [base_electrode + i * row_offset for i in range(num_rows)]


def generate_electrode_pool() -> List[int]:
    base = 13200
    ranges = [
        list(range(base + 15, base + 55, 2)),
        list(range(base + 56, base + 95, 2)),
        list(range(base + 125, base + 165, 2)),
        list(range(base + 166, base + 205, 2)),
    ]
    unique_base = list(dict.fromkeys([electrode for chunk in ranges for electrode in chunk]))
    recording = []
    for base_electrode in unique_base:
        recording.extend(generate_vertical_electrodes(base_electrode))
    return recording


RECORDING_ELECTRODES = generate_electrode_pool()
DECODING_LEFT_ELECTRODES = [RECORDING_ELECTRODES[0]]
DECODING_RIGHT_ELECTRODES = [RECORDING_ELECTRODES[400]]


def print_step(message: str) -> None:
    print(f"[STEP] {message}")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_success(message: str) -> None:
    print(f"[OK] {message}")


class CPPProcessManager:
    def __init__(
        self,
        executable: str,
        config_path: Path,
        line_handler: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.executable = executable
        self.config_path = config_path
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()
        self.running = False
        self.line_handler = line_handler
        self.stdin_lock = threading.Lock()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [self.executable, str(self.config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.running = True
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()
        if not self.ready_event.wait(timeout=10):
            print_info("C++ ready marker not observed within 10 seconds; continuing")

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        ready_marker = "[SYNC] Waiting for start signal"
        while self.running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            print(f"[C++] {line}")
            if ready_marker in line:
                self.ready_event.set()
            if self.line_handler is not None:
                response = self.line_handler(line)
                if response:
                    self._write_stdin_line(response)

    def send_start_signal(self) -> None:
        self._write_stdin_line("start")

    def _write_stdin_line(self, line: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("C++ process is not running")
        with self.stdin_lock:
            self.process.stdin.write(f"{line}\n")
            self.process.stdin.flush()

    def wait(self, timeout: Optional[float] = None) -> int:
        if self.process is None:
            raise RuntimeError("C++ process is not running")
        return self.process.wait(timeout=timeout)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.running = False
        if self.output_thread is not None and self.output_thread.is_alive():
            self.output_thread.join(timeout=1)


def initialize_system() -> None:
    print_step("Initializing MaxLab")
    mx.initialize()
    result = mx.send(mx.Core().enable_stimulation_power(True))
    if result != "Ok":
        raise RuntimeError(f"Failed to enable stimulation power: {result}")
    time.sleep(mx.Timing.waitInit)
    print_success("MaxLab initialized")


def configure_array(recording_electrodes: Sequence[int], stimulation_electrodes: Sequence[int]) -> mx.Array:
    print_step("Configuring electrode array")
    array = mx.Array("cartpole_experiment")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(list(recording_electrodes))
    array.select_stimulation_electrodes(list(stimulation_electrodes))
    array.route()
    return array


def build_stim_candidate_electrodes(stim_electrodes: Sequence[int], radius: int = 2) -> List[int]:
    candidates: List[int] = []
    seen = set()
    for electrode in stim_electrodes:
        if electrode not in seen:
            candidates.append(electrode)
            seen.add(electrode)
    for electrode in stim_electrodes:
        for search_radius in range(1, radius + 1):
            for neighbor in mx.electrode_neighbors(electrode, search_radius):
                if neighbor not in seen:
                    candidates.append(neighbor)
                    seen.add(neighbor)
    return candidates


def connect_stim_units_to_stim_electrodes(
    requested_electrodes: Sequence[int],
    array: mx.Array,
    candidate_electrodes: Sequence[int],
    radius: int = 2,
) -> Tuple[Dict[int, int], List[int]]:
    print_step("Connecting stimulation units")
    electrode_to_unit: Dict[int, int] = {}
    resolved: List[int] = []
    used_units = set()
    candidate_set = set(candidate_electrodes)

    def disconnect(electrode: int) -> None:
        try:
            array.disconnect_electrode_from_stimulation(electrode)
        except Exception:
            pass

    def try_connect(electrode: int) -> Optional[int]:
        array.connect_electrode_to_stimulation(electrode)
        stim = array.query_stimulation_at_electrode(electrode)
        if len(stim) == 0:
            disconnect(electrode)
            return None
        unit = int(stim)
        if unit in used_units:
            disconnect(electrode)
            return None
        return unit

    for requested in requested_electrodes:
        candidates = [requested]
        for search_radius in range(1, radius + 1):
            candidates.extend(
                neighbor
                for neighbor in mx.electrode_neighbors(requested, search_radius)
                if neighbor in candidate_set
            )
        selected_electrode = None
        selected_unit = None
        for candidate in candidates:
            selected_unit = try_connect(candidate)
            if selected_unit is not None:
                selected_electrode = candidate
                break
        if selected_electrode is None or selected_unit is None:
            raise RuntimeError(f"Unable to connect stimulation electrode near {requested}")
        electrode_to_unit[selected_electrode] = selected_unit
        resolved.append(selected_electrode)
        used_units.add(selected_unit)

    mapping_str = ", ".join(f"E{e}:U{u}" for e, u in electrode_to_unit.items())
    print_success(f"Connected units: {mapping_str}")
    return electrode_to_unit, resolved


def configure_and_powerup_stim_units(stim_units: Sequence[int]) -> None:
    print_step("Powering stimulation units")
    for stim_unit in stim_units:
        mx.send(
            mx.StimulationUnit(stim_unit)
            .power_up(True)
            .connect(True)
            .set_voltage_mode()
            .dac_source(0)
        )
    print_success(f"Powered {len(stim_units)} stimulation units")


def amplitude_mV_to_dac_bits(amplitude_mV: float) -> int:
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    return int(512 - (amplitude_mV / dac_lsb_mV))


def create_biphasic_pulse(seq: mx.Sequence, amplitude_mV: float, phase_us: int, event_label: str) -> None:
    global event_counter
    phase_samples = int(phase_us / 50)
    positive_bits = amplitude_mV_to_dac_bits(amplitude_mV)
    negative_bits = amplitude_mV_to_dac_bits(-amplitude_mV)
    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, event_label))
    seq.append(mx.DAC(0, positive_bits))
    seq.append(mx.DelaySamples(phase_samples))
    seq.append(mx.DAC(0, negative_bits))
    seq.append(mx.DelaySamples(phase_samples))
    seq.append(mx.DAC(0, 512))


def create_unit_configuration_commands(target_units: Sequence[int], all_units: Sequence[int]) -> List:
    commands = []
    target_unit_set = set(target_units)
    for unit_id in all_units:
        commands.append(
            mx.StimulationUnit(unit_id)
            .connect(unit_id in target_unit_set)
            .power_up(True)
            .set_voltage_mode()
            .dac_source(0)
        )
    return commands


def append_pulse_for_unit(
    seq: mx.Sequence,
    target_unit: int,
    all_units: Sequence[int],
    event_label: str,
) -> None:
    for cmd in create_unit_configuration_commands([target_unit], all_units):
        seq.append(cmd)
    create_biphasic_pulse(
        seq,
        amplitude_mV=STIM_PARAMS["pulse_amplitude_mV"],
        phase_us=STIM_PARAMS["phase_us"],
        event_label=event_label,
    )


def prepare_encoding_sequences(unit_ids: Sequence[int], all_unit_ids: Sequence[int]) -> Dict[str, mx.Sequence]:
    if len(unit_ids) != 2:
        raise RuntimeError("Expected exactly two encoding stimulation units")
    sequences: Dict[str, mx.Sequence] = {}
    names = ["encode_left_pulse", "encode_right_pulse"]
    for name, unit_id in zip(names, unit_ids):
        seq = mx.Sequence(name=name, persistent=True)
        append_pulse_for_unit(seq, unit_id, all_unit_ids, f"{name}_pulse")
        seq.send()
        sequences[name] = seq
    return sequences


def resolve_mode_schedule(mode: str, num_cycles: int) -> List[Dict[str, object]]:
    if mode not in PAPER_MODE_CHOICES:
        raise ValueError(f"Unsupported mode: {mode}")
    if num_cycles <= 0:
        raise ValueError(f"num_cycles must be > 0, got {num_cycles}")

    if mode == "cycled":
        conditions = ["null", "random", "adaptive"]
        return [
            {"cycle_index": idx, "condition": conditions[idx % len(conditions)]}
            for idx in range(num_cycles)
        ]

    condition = "adaptive" if mode == "continuous_adaptive" else mode
    return [{"cycle_index": idx, "condition": condition} for idx in range(num_cycles)]


def requires_adaptive_patterns(mode: str, num_cycles: int) -> bool:
    return any(cycle["condition"] == "adaptive" for cycle in resolve_mode_schedule(mode, num_cycles))


def requires_random_bridge(mode: str, num_cycles: int) -> bool:
    return any(cycle["condition"] == "random" for cycle in resolve_mode_schedule(mode, num_cycles))


def validate_mode_requirements(mode: str, training_stim_electrodes: Sequence[int]) -> None:
    training_count = len(training_stim_electrodes)
    if mode in {"random", "cycled"} and training_count < 5:
        raise ValueError(
            f"Mode '{mode}' requires at least 5 training electrodes, got {training_count}"
        )
    if mode in {"adaptive", "continuous_adaptive"} and training_count < 2:
        raise ValueError(
            f"Mode '{mode}' requires at least 2 training electrodes, got {training_count}"
        )


def _period_samples() -> int:
    return int(20000 / STIM_PARAMS["training_frequency_hz"])


def _inter_pulse_samples() -> int:
    return int(STIM_PARAMS["inter_pulse_interval_ms"] * 1000 / 50)


def _append_training_epoch(
    seq: mx.Sequence,
    unit_ids: Sequence[int],
    all_unit_ids: Sequence[int],
    event_prefix: str,
) -> None:
    inter_pulse_samples = _inter_pulse_samples()
    for pulse_index, unit_id in enumerate(unit_ids):
        append_pulse_for_unit(seq, unit_id, all_unit_ids, f"{event_prefix}_{pulse_index}")
        if pulse_index < len(unit_ids) - 1:
            seq.append(mx.DelaySamples(inter_pulse_samples))
    trailing_gap = max(0, _period_samples() - inter_pulse_samples * max(0, len(unit_ids) - 1))
    seq.append(mx.DelaySamples(trailing_gap))


def prepare_adaptive_training_sequences(training_unit_ids: Sequence[int], all_unit_ids: Sequence[int]) -> List[str]:
    if len(training_unit_ids) < 2:
        raise RuntimeError("At least two training units are required")
    pattern_names: List[str] = []
    repetitions = int((ADAPTIVE_TRAINING_WINDOW_MS / 1000.0) * STIM_PARAMS["training_frequency_hz"])

    for first_idx, second_idx in itertools.combinations(range(len(training_unit_ids)), 2):
        name = f"train_pair_{first_idx}_{second_idx}"
        seq = mx.Sequence(name=name, persistent=True)
        first_unit = training_unit_ids[first_idx]
        second_unit = training_unit_ids[second_idx]
        for repetition in range(repetitions):
            _append_training_epoch(
                seq,
                [first_unit, second_unit],
                all_unit_ids,
                f"{name}_{repetition}",
            )
        seq.send()
        pattern_names.append(name)
    return pattern_names


def prepare_training_sequences(training_unit_ids: Sequence[int], all_unit_ids: Sequence[int]) -> List[str]:
    return prepare_adaptive_training_sequences(training_unit_ids, all_unit_ids)


class RandomTrainingSequenceManager:
    def __init__(
        self,
        training_electrodes: Sequence[int],
        training_unit_ids: Sequence[int],
        all_unit_ids: Sequence[int],
        random_seed: int = 12345,
    ) -> None:
        if len(training_electrodes) != len(training_unit_ids):
            raise ValueError("training_electrodes and training_unit_ids must align one-to-one")
        if len(training_unit_ids) < 5:
            raise ValueError("Random training requires at least 5 training units")
        self.training_pairs = list(zip(training_electrodes, training_unit_ids))
        self.all_unit_ids = list(all_unit_ids)
        self.rng = random.Random(random_seed)
        self.sequence_counter = 0

    def generate_sequence(self) -> Dict[str, object]:
        selected_pairs = self.rng.sample(self.training_pairs, 5)
        selected_electrodes = [int(electrode) for electrode, _ in selected_pairs]
        selected_units = [int(unit_id) for _, unit_id in selected_pairs]
        repetitions = int((RANDOM_TRAINING_WINDOW_MS / 1000.0) * STIM_PARAMS["training_frequency_hz"])
        sequence_name = f"train_random5_{self.sequence_counter}"
        self.sequence_counter += 1
        seq = mx.Sequence(name=sequence_name, persistent=True)
        for repetition in range(repetitions):
            _append_training_epoch(
                seq,
                selected_units,
                self.all_unit_ids,
                f"{sequence_name}_{repetition}",
            )
        seq.send()
        return {
            "sequence_name": sequence_name,
            "sequence_type": "random5",
            "training_electrodes": selected_electrodes,
        }


def make_training_request_handler(
    random_sequence_manager: Optional[RandomTrainingSequenceManager],
) -> Callable[[str], Optional[str]]:
    def handle_line(line: str) -> Optional[str]:
        if not line.startswith("[TRAINING_REQUEST]"):
            return None
        request_type = line.partition("]")[2].strip()
        if request_type != "random5":
            return "training_sequence_error|unsupported_request"
        if random_sequence_manager is None:
            return "training_sequence_error|random_manager_unavailable"
        payload = random_sequence_manager.generate_sequence()
        electrodes = ",".join(str(value) for value in payload["training_electrodes"])
        return f"training_sequence|{payload['sequence_name']}|{payload['sequence_type']}|{electrodes}"

    return handle_line


def start_recording(recording_name: str, wells: Sequence[int]) -> mx.Saving:
    saving = mx.Saving()
    saving.open_directory(str(RECORDING_DIR))
    saving.start_file(recording_name)
    saving.group_define(0, "all_channels", list(range(1024)))
    saving.start_recording(list(wells))
    return saving


def stop_recording(saving: mx.Saving) -> None:
    saving.stop_recording()
    saving.stop_file()
    saving.group_delete_all()
    time.sleep(mx.Timing.waitAfterRecording)


def build_runtime_config_payload(
    target_well: int,
    decoding_left_channels: Sequence[int],
    decoding_right_channels: Sequence[int],
    encoding_stim_electrodes: Optional[Sequence[int]],
    training_stim_electrodes: Optional[Sequence[int]],
    decoding_left_electrodes: Optional[Sequence[int]],
    decoding_right_electrodes: Optional[Sequence[int]],
    adaptive_pattern_names: Sequence[str],
    log_path: Path,
    num_cycles: int,
    mode: str,
    show_gui: bool,
) -> Dict[str, object]:
    cycle_schedule = resolve_mode_schedule(mode, num_cycles)
    cycle_conditions = [str(cycle["condition"]) for cycle in cycle_schedule]
    experiment_duration_s = num_cycles * (CYCLE_DURATION_S + REST_DURATION_S)
    return {
        "target_well": target_well,
        "read_window_ms": READ_WINDOW_MS,
        "training_window_ms": TRAINING_WINDOW_MS,
        "show_gui": bool(show_gui),
        "wait_for_sync": True,
        "channel_count": 1024,
        "experiment_duration_s": experiment_duration_s,
        "num_cycles": int(num_cycles),
        "active_cycle_duration_s": CYCLE_DURATION_S,
        "rest_duration_s": REST_DURATION_S,
        "cycle_condition_schedule": cycle_conditions,
        "first_cycle_pass_threshold_s": FIRST_CYCLE_PASS_THRESHOLD_S,
        "encoding_scale_a": 7.0,
        "encoding_scale_b": 0.15,
        "ema_alpha": 0.2,
        "force_scale_n": 10.0,
        "sample_rate_hz": 20000.0,
        "threshold_multiplier": 3.0,
        "min_threshold": -20.0,
        "refractory_samples": 1000,
        "decoding_left_channels": list(decoding_left_channels),
        "decoding_right_channels": list(decoding_right_channels),
        "encoding_stim_electrodes": list(encoding_stim_electrodes or []),
        "training_stim_electrodes": list(training_stim_electrodes or []),
        "decoding_left_electrodes": list(decoding_left_electrodes or []),
        "decoding_right_electrodes": list(decoding_right_electrodes or []),
        "stim_pulse_amplitude_mV": STIM_PARAMS["pulse_amplitude_mV"],
        "stim_phase_us": STIM_PARAMS["phase_us"],
        "stim_inter_pulse_interval_ms": STIM_PARAMS["inter_pulse_interval_ms"],
        "stim_training_frequency_hz": STIM_PARAMS["training_frequency_hz"],
        "artifact_filter_mode": "iir",
        "encoding_left_sequence": "encode_left_pulse",
        "encoding_right_sequence": "encode_right_pulse",
        "training_pattern_names": list(adaptive_pattern_names),
        "adaptive_training_pattern_names": list(adaptive_pattern_names),
        "log_path": str(log_path),
        "random_seed": 12345,
        "mode": mode,
    }


def export_runtime_config(
    config_path: Path,
    target_well: int,
    decoding_left_channels: Sequence[int],
    decoding_right_channels: Sequence[int],
    encoding_stim_electrodes: Optional[Sequence[int]],
    training_stim_electrodes: Optional[Sequence[int]],
    decoding_left_electrodes: Optional[Sequence[int]],
    decoding_right_electrodes: Optional[Sequence[int]],
    adaptive_pattern_names: Sequence[str],
    log_path: Path,
    num_cycles: int,
    mode: str,
    show_gui: bool,
) -> None:
    runtime_config = build_runtime_config_payload(
        target_well=target_well,
        decoding_left_channels=decoding_left_channels,
        decoding_right_channels=decoding_right_channels,
        encoding_stim_electrodes=encoding_stim_electrodes,
        training_stim_electrodes=training_stim_electrodes,
        decoding_left_electrodes=decoding_left_electrodes,
        decoding_right_electrodes=decoding_right_electrodes,
        adaptive_pattern_names=adaptive_pattern_names,
        log_path=log_path,
        num_cycles=num_cycles,
        mode=mode,
        show_gui=show_gui,
    )
    config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")


def run_cartpole_experiment(num_cycles: int, mode: str, wells: Sequence[int], show_gui: bool) -> None:
    if mode not in PAPER_MODE_CHOICES:
        raise ValueError(f"Unsupported mode: {mode}")
    if not os.path.exists(CPP_EXECUTABLE):
        raise RuntimeError(f"C++ executable not found: {CPP_EXECUTABLE}")
    if num_cycles <= 0:
        raise ValueError(f"num_cycles must be > 0, got {num_cycles}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"cartpole_{mode}_{timestamp}"
    config_path = RECORDING_DIR / f"{session_name}_config.json"
    log_path = RECORDING_DIR / f"{session_name}_episodes.jsonl"

    initialize_system()
    mx.activate(list(wells))

    stimulation_electrodes = list(ENCODING_STIM_ELECTRODES) + list(TRAINING_STIM_ELECTRODES)
    stimulation_candidates = build_stim_candidate_electrodes(stimulation_electrodes)
    array = configure_array(RECORDING_ELECTRODES, stimulation_candidates)
    electrode_to_unit, resolved_stim = connect_stim_units_to_stim_electrodes(
        stimulation_electrodes,
        array,
        stimulation_candidates,
    )

    array.download(list(wells))
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)
    mx.clear_events()

    all_units = list(electrode_to_unit.values())
    configure_and_powerup_stim_units(all_units)

    resolved_encoding = resolved_stim[: len(ENCODING_STIM_ELECTRODES)]
    resolved_training = resolved_stim[len(ENCODING_STIM_ELECTRODES) :]
    validate_mode_requirements(mode, resolved_training)
    encoding_units = [electrode_to_unit[electrode] for electrode in resolved_encoding]
    training_units = [electrode_to_unit[electrode] for electrode in resolved_training]

    prepare_encoding_sequences(encoding_units, all_units)
    adaptive_pattern_names = (
        prepare_adaptive_training_sequences(training_units, all_units)
        if requires_adaptive_patterns(mode, num_cycles)
        else []
    )
    random_sequence_manager = (
        RandomTrainingSequenceManager(resolved_training, training_units, all_units)
        if requires_random_bridge(mode, num_cycles)
        else None
    )

    config = array.get_config()
    decoding_left_channels = config.get_channels_for_electrodes(DECODING_LEFT_ELECTRODES)
    decoding_right_channels = config.get_channels_for_electrodes(DECODING_RIGHT_ELECTRODES)
    if not decoding_left_channels or not decoding_right_channels:
        raise RuntimeError("Failed to map manual decoding electrodes to recording channels")

    export_runtime_config(
        config_path=config_path,
        target_well=wells[0],
        decoding_left_channels=decoding_left_channels,
        decoding_right_channels=decoding_right_channels,
        encoding_stim_electrodes=resolved_encoding,
        training_stim_electrodes=resolved_training,
        decoding_left_electrodes=DECODING_LEFT_ELECTRODES,
        decoding_right_electrodes=DECODING_RIGHT_ELECTRODES,
        adaptive_pattern_names=adaptive_pattern_names,
        log_path=log_path,
        num_cycles=num_cycles,
        mode=mode,
        show_gui=show_gui,
    )
    print_success(f"Runtime config written to {config_path}")

    cpp = CPPProcessManager(
        CPP_EXECUTABLE,
        config_path,
        line_handler=make_training_request_handler(random_sequence_manager),
    )
    cpp.start()
    saving = start_recording(session_name, wells)

    try:
        cpp.send_start_signal()
        print_success("C++ cartpole loop started")
        if cpp.process is None:
            raise RuntimeError("Missing C++ process handle")
        while cpp.process.poll() is None:
            time.sleep(1)
        if cpp.process.returncode != 0:
            raise RuntimeError(f"C++ process exited with code {cpp.process.returncode}")
    finally:
        stop_recording(saving)
        cpp.stop()

    print_success(f"Experiment complete. Episode log: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cartpole closed-loop experiment")
    parser.add_argument(
        "--num-cycles",
        type=int,
        required=True,
        help="Number of experiment cycles. One cycle = 15 min training + 45 min rest.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cycled",
        choices=list(PAPER_MODE_CHOICES),
    )
    parser.add_argument("--wells", type=int, nargs="+", default=[0])
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()

    run_cartpole_experiment(
        num_cycles=args.num_cycles,
        mode=args.mode,
        wells=args.wells,
        show_gui=args.show_gui,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

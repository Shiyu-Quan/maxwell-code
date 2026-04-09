#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import maxlab as mx

from cartpole_setup import (
    CPP_EXECUTABLE,
    RECORDING_DIR,
    RECORDING_ELECTRODES,
    CPPProcessManager,
    build_stim_candidate_electrodes,
    configure_and_powerup_stim_units,
    configure_array,
    connect_stim_units_to_stim_electrodes,
    export_runtime_config,
    initialize_system,
    prepare_encoding_sequences,
    prepare_training_sequences,
    print_success,
    start_recording,
    stop_recording,
)


def load_selection_config(selection_config_path: Path) -> Dict[str, List[int]]:
    payload = json.loads(selection_config_path.read_text(encoding="utf-8"))
    selection = payload.get("selection_config", payload)
    required_keys = (
        "encoding_stim_electrodes",
        "training_stim_electrodes",
        "decoding_left_electrodes",
        "decoding_right_electrodes",
    )
    missing = [key for key in required_keys if key not in selection]
    if missing:
        raise RuntimeError(
            f"Selection config {selection_config_path} is missing keys: {', '.join(missing)}"
        )
    return {key: [int(value) for value in selection[key]] for key in required_keys}


def run_selected_cartpole_experiment(
    duration_minutes: int,
    mode: str,
    wells: Sequence[int],
    show_gui: bool,
    selection_config_path: Path,
) -> None:
    if mode not in {"cycled_adaptive", "continuous_adaptive"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not os.path.exists(CPP_EXECUTABLE):
        raise RuntimeError(f"C++ executable not found: {CPP_EXECUTABLE}")

    selection = load_selection_config(selection_config_path)
    encoding_stim_electrodes = selection["encoding_stim_electrodes"]
    training_stim_electrodes = selection["training_stim_electrodes"]
    decoding_left_electrodes = selection["decoding_left_electrodes"]
    decoding_right_electrodes = selection["decoding_right_electrodes"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"cartpole_{mode}_{timestamp}"
    config_path = RECORDING_DIR / f"{session_name}_config.json"
    log_path = RECORDING_DIR / f"{session_name}_episodes.jsonl"

    initialize_system()
    mx.activate(list(wells))

    stimulation_electrodes = list(encoding_stim_electrodes) + list(training_stim_electrodes)
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

    resolved_encoding = resolved_stim[: len(encoding_stim_electrodes)]
    resolved_training = resolved_stim[len(encoding_stim_electrodes) :]
    encoding_units = [electrode_to_unit[electrode] for electrode in resolved_encoding]
    training_units = [electrode_to_unit[electrode] for electrode in resolved_training]

    prepare_encoding_sequences(encoding_units, all_units)
    training_pattern_names = prepare_training_sequences(training_units, all_units)

    config = array.get_config()
    decoding_left_channels = config.get_channels_for_electrodes(decoding_left_electrodes)
    decoding_right_channels = config.get_channels_for_electrodes(decoding_right_electrodes)
    if not decoding_left_channels or not decoding_right_channels:
        raise RuntimeError("Failed to map selected decoding electrodes to recording channels")

    export_runtime_config(
        config_path=config_path,
        target_well=wells[0],
        decoding_left_channels=decoding_left_channels,
        decoding_right_channels=decoding_right_channels,
        encoding_stim_electrodes=resolved_encoding,
        training_stim_electrodes=resolved_training,
        decoding_left_electrodes=decoding_left_electrodes,
        decoding_right_electrodes=decoding_right_electrodes,
        training_pattern_names=training_pattern_names,
        log_path=log_path,
        duration_minutes=duration_minutes,
        mode=mode,
        show_gui=show_gui,
    )
    print_success(f"Runtime config written to {config_path}")

    cpp = CPPProcessManager(CPP_EXECUTABLE, config_path)
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
    parser = argparse.ArgumentParser(description="Cartpole closed-loop experiment using selection_config")
    parser.add_argument("--duration", type=int, default=15, help="Experiment duration in minutes")
    parser.add_argument(
        "--mode",
        type=str,
        default="cycled_adaptive",
        choices=["cycled_adaptive", "continuous_adaptive"],
    )
    parser.add_argument("--wells", type=int, nargs="+", default=[0])
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--selection-config", type=Path, required=True)
    args = parser.parse_args()

    run_selected_cartpole_experiment(
        duration_minutes=args.duration,
        mode=args.mode,
        wells=args.wells,
        show_gui=args.show_gui,
        selection_config_path=args.selection_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

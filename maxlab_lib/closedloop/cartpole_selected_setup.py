#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import maxlab as mx

from cartpole_setup import (
    CPP_EXECUTABLE,
    PAPER_MODE_CHOICES,
    RECORDING_DIR,
    RECORDING_ELECTRODES,
    CPPProcessManager,
    build_stim_candidate_electrodes,
    configure_and_powerup_stim_units,
    configure_array,
    connect_stim_units_to_stim_electrodes,
    export_runtime_config,
    initialize_system,
    make_training_request_handler,
    prepare_encoding_sequences,
    prepare_adaptive_training_sequences,
    print_info,
    print_success,
    requires_adaptive_patterns,
    requires_random_bridge,
    start_recording,
    stop_recording,
    validate_mode_requirements as validate_mode_requirements_common,
    RandomTrainingSequenceManager,
)


def load_selection_config(selection_config_path: Path) -> Dict[str, Any]:
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

    parsed = {key: [int(value) for value in selection[key]] for key in required_keys}
    if "recording_electrodes" in selection and list(selection["recording_electrodes"]):
        parsed["recording_electrodes"] = [int(value) for value in selection["recording_electrodes"]]
        parsed["recording_electrodes_source"] = "selection_config"
    else:
        parsed["recording_electrodes"] = [int(value) for value in RECORDING_ELECTRODES]
        parsed["recording_electrodes_source"] = "default_fallback_pool"
    return parsed


def validate_mode_requirements(mode: str, training_stim_electrodes: Sequence[int]) -> None:
    validate_mode_requirements_common(mode, training_stim_electrodes)


def run_selected_cartpole_experiment(
    mode: str,
    wells: Sequence[int],
    show_gui: bool,
    selection_config_path: Path,
    num_cycles: int,
) -> None:
    if mode not in PAPER_MODE_CHOICES:
        raise ValueError(f"Unsupported mode: {mode}")
    if not os.path.exists(CPP_EXECUTABLE):
        raise RuntimeError(f"C++ executable not found: {CPP_EXECUTABLE}")
    if num_cycles <= 0:
        raise ValueError(f"num_cycles must be > 0, got {num_cycles}")
    print_info(
        f"Using paper-aligned cycle count: cycles={num_cycles}, "
        "one_cycle=60 min (15 train + 45 rest)"
    )

    selection = load_selection_config(selection_config_path)
    encoding_stim_electrodes = selection["encoding_stim_electrodes"]
    training_stim_electrodes = selection["training_stim_electrodes"]
    decoding_left_electrodes = selection["decoding_left_electrodes"]
    decoding_right_electrodes = selection["decoding_right_electrodes"]
    recording_electrodes = selection["recording_electrodes"]
    validate_mode_requirements(mode, training_stim_electrodes)
    print_info(
        "Using recording electrodes from "
        f"{selection['recording_electrodes_source']}: count={len(recording_electrodes)}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"cartpole_{mode}_{timestamp}"
    config_path = RECORDING_DIR / f"{session_name}_config.json"
    log_path = RECORDING_DIR / f"{session_name}_episodes.jsonl"

    initialize_system()
    mx.activate(list(wells))

    stimulation_electrodes = list(encoding_stim_electrodes) + list(training_stim_electrodes)
    stimulation_candidates = build_stim_candidate_electrodes(stimulation_electrodes)
    array = configure_array(recording_electrodes, stimulation_candidates)
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
    parser = argparse.ArgumentParser(description="Cartpole closed-loop experiment using selection_config")
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
    parser.add_argument("--selection-config", type=Path, required=True)
    args = parser.parse_args()

    run_selected_cartpole_experiment(
        mode=args.mode,
        wells=args.wells,
        show_gui=args.show_gui,
        selection_config_path=args.selection_config,
        num_cycles=args.num_cycles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

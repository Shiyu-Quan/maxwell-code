#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import maxlab as mx

from cartpole_selection import analyze_spontaneous_recording, analyze_stimulation_manifest
from cartpole_setup import (
    RECORDING_DIR,
    RECORDING_ELECTRODES,
    STIM_PARAMS,
    append_pulse_for_unit,
    build_stim_candidate_electrodes,
    configure_and_powerup_stim_units,
    configure_array,
    connect_stim_units_to_stim_electrodes,
    initialize_system,
    print_info,
    print_step,
    print_success,
    start_recording,
    stop_recording,
)


SAMPLE_RATE_HZ = 20000.0
SPONTANEOUS_THRESHOLD_MULTIPLIER = 5.0
REALTIME_DETECTION_MULTIPLIER = 3.0
RECORD_REFRACTORY_SAMPLES = 20
FIRST_ORDER_WINDOW_MS = (10.0, 18.0)
MULTI_ORDER_WINDOW_MS = (10.0, 200.0)
BURST_MAD_MULTIPLIER = 3.0
STIM_REPETITIONS = 50
STIM_FREQUENCY_HZ = 2.0
MIN_PUTATIVE_UNITS = 8


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _recording_path(recording_name: str) -> Path:
    return RECORDING_DIR / f"{recording_name}.raw.h5"


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_recording_metadata(
    metadata_path: Path,
    array: mx.Array,
    recording_electrodes: Sequence[int],
    target_well: int,
    duration_s: float,
) -> Dict[str, Any]:
    config = array.get_config()
    mapped_channels = config.get_channels_for_electrodes(list(recording_electrodes))
    recording_channels = [
        {"electrode": int(electrode), "channel": int(channel)}
        for electrode, channel in zip(recording_electrodes, mapped_channels)
    ]
    payload = {
        "well": int(target_well),
        "duration_s": float(duration_s),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "recording_electrodes": [int(electrode) for electrode in recording_electrodes],
        "recording_channels": recording_channels,
    }
    _json_dump(metadata_path, payload)
    return payload


def _power_down_all_stim_units() -> None:
    for stimulation_unit in range(32):
        mx.send(mx.StimulationUnit(stimulation_unit).power_up(False).connect(False))


def run_record_stage(
    duration_s: float,
    wells: Sequence[int],
    recording_electrodes: Sequence[int],
    analyze: bool,
) -> Dict[str, Path]:
    timestamp = _timestamp()
    recording_name = f"cartpole_record_{timestamp}"
    recording_path = _recording_path(recording_name)
    metadata_path = RECORDING_DIR / f"{recording_name}_meta.json"
    analysis_path = RECORDING_DIR / f"{recording_name}_putative_units.json"

    initialize_system()
    mx.activate(list(wells))

    print_step("Routing recording electrodes for spontaneous Record stage")
    array = configure_array(recording_electrodes, [])
    array.download(list(wells))
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)
    mx.clear_events()

    _export_recording_metadata(
        metadata_path=metadata_path,
        array=array,
        recording_electrodes=recording_electrodes,
        target_well=int(wells[0]),
        duration_s=duration_s,
    )
    saving = start_recording(recording_name, list(wells))
    try:
        print_info(f"Recording spontaneous activity for {duration_s:.1f} seconds")
        time.sleep(duration_s)
    finally:
        stop_recording(saving)

    print_success(f"Record stage raw file written to {recording_path}")
    print_success(f"Record stage metadata written to {metadata_path}")

    if analyze:
        analyze_spontaneous_recording(
            recording_path=recording_path,
            metadata_path=metadata_path,
            output_path=analysis_path,
            threshold_multiplier=SPONTANEOUS_THRESHOLD_MULTIPLIER,
            refractory_samples=RECORD_REFRACTORY_SAMPLES,
            min_spike_count=20,
            top_k=64,
        )
        print_success(f"Putative units written to {analysis_path}")

    return {
        "recording_path": recording_path,
        "metadata_path": metadata_path,
        "analysis_path": analysis_path,
    }


def _load_putative_units(record_analysis_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = json.loads(record_analysis_path.read_text(encoding="utf-8"))["record_analysis"]
    units = list(payload["putative_units"])
    if len(units) < MIN_PUTATIVE_UNITS:
        raise RuntimeError(
            f"Record analysis only found {len(units)} putative units; need at least {MIN_PUTATIVE_UNITS}"
        )
    return payload, units


def _synthesize_putative_units(record_analysis: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    channel_metrics = list(record_analysis.get("channel_metrics", []))
    synthesized: List[Dict[str, Any]] = []
    for row in channel_metrics:
        electrode = int(row.get("electrode", -1))
        if electrode < 0:
            continue
        synthesized.append(
            {
                "channel": int(row.get("channel", -1)),
                "electrode": electrode,
                "spike_count": int(row.get("spike_count", 0)),
                "firing_rate_hz": float(row.get("firing_rate_hz", 0.0)),
                "rms": float(row.get("rms", 0.0) if row.get("rms") is not None else 0.0),
                "threshold": float(row.get("threshold", 0.0) if row.get("threshold") is not None else 0.0),
                "median_negative_peak": float(
                    row.get("median_negative_peak", 0.0) if row.get("median_negative_peak") is not None else 0.0
                ),
                "score": float(row.get("score", 0.0) if row.get("score") is not None else 0.0),
            }
        )
        if len(synthesized) >= count:
            break
    return synthesized


def _unique_electrodes(items: Sequence[int]) -> List[int]:
    seen = set()
    output: List[int] = []
    for item in items:
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _write_debug_selection_from_manifest(
    manifest: Dict[str, Any],
    record_analysis: Dict[str, Any],
    selection_path: Path,
    analysis_path: Path,
    reason: str,
) -> None:
    source_electrodes = _unique_electrodes(
        int(item["source_electrode"]) for item in manifest.get("stimulus_recordings", [])
    )
    if len(source_electrodes) < 4:
        raise RuntimeError(
            "Debug mode fallback requires at least 4 probed source electrodes to synthesize selection config"
        )

    encoding = source_electrodes[:2]
    training_pool = source_electrodes[2:]
    if len(training_pool) < 2:
        training_pool = source_electrodes[-2:]
    training = training_pool[: max(2, min(6, len(training_pool)))]

    record_units = list(record_analysis.get("putative_units", []))
    candidate_decoding = _unique_electrodes(
        int(unit.get("electrode", -1)) for unit in record_units if int(unit.get("electrode", -1)) >= 0
    )
    non_stim_decoding = [electrode for electrode in candidate_decoding if electrode not in set(encoding + training)]
    if len(non_stim_decoding) >= 2:
        decoding = non_stim_decoding[:2]
    else:
        fallback = [electrode for electrode in source_electrodes if electrode not in set(encoding)]
        if len(fallback) < 2:
            raise RuntimeError("Debug mode fallback could not synthesize two decoding electrodes")
        decoding = fallback[:2]

    selection_payload = {
        "selection_config": {
            "encoding_stim_electrodes": encoding,
            "decoding_left_electrodes": [int(decoding[0])],
            "decoding_right_electrodes": [int(decoding[1])],
            "training_stim_electrodes": training,
            "source_record_analysis": str(manifest.get("record_analysis_path", "")),
            "source_stim_analysis": str(analysis_path),
            "debug_mode": True,
            "debug_reason": reason,
        }
    }
    analysis_payload = {
        "stimulate_analysis": {
            "manifest_path": str(manifest.get("manifest_path", "")),
            "record_analysis_path": str(manifest.get("record_analysis_path", "")),
            "debug_mode": True,
            "debug_reason": reason,
            "probe_count": len(manifest.get("stimulus_recordings", [])),
        }
    }
    _json_dump(analysis_path, analysis_payload)
    _json_dump(selection_path, selection_payload)


def _prepare_single_probe_sequence(stim_unit: int) -> mx.Sequence:
    seq = mx.Sequence()
    append_pulse_for_unit(
        seq,
        target_unit=stim_unit,
        all_units=[stim_unit],
        event_label="stimulus_probe",
    )
    return seq


def _probe_single_unit(
    source_unit: Dict[str, Any],
    recording_electrodes: Sequence[int],
    wells: Sequence[int],
    repetitions: int,
    stim_frequency_hz: float,
    stim_neighbor_radius: int,
) -> Dict[str, Any]:
    source_electrode = int(source_unit["electrode"])
    source_channel = int(source_unit["channel"])
    candidate_electrodes = build_stim_candidate_electrodes([source_electrode], radius=stim_neighbor_radius)
    array = configure_array(recording_electrodes, candidate_electrodes)
    electrode_to_unit, resolved = connect_stim_units_to_stim_electrodes(
        [source_electrode],
        array,
        candidate_electrodes,
        radius=stim_neighbor_radius,
    )
    resolved_stim_electrode = int(resolved[0])
    stim_unit = int(electrode_to_unit[resolved_stim_electrode])

    array.download(list(wells))
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)
    mx.clear_events()
    configure_and_powerup_stim_units([stim_unit])

    recording_name = f"cartpole_stim_probe_ch{source_channel}_{_timestamp()}"
    saving = start_recording(recording_name, list(wells))
    try:
        sequence = _prepare_single_probe_sequence(stim_unit)
        inter_train_delay_s = 1.0 / stim_frequency_hz
        for _ in range(repetitions):
            sequence.send()
            time.sleep(inter_train_delay_s)
    finally:
        stop_recording(saving)
        _power_down_all_stim_units()

    return {
        "source_channel": source_channel,
        "source_electrode": source_electrode,
        "resolved_stim_electrode": resolved_stim_electrode,
        "stim_unit": stim_unit,
        "repetitions": repetitions,
        "stim_frequency_hz": stim_frequency_hz,
        "recording_path": str(_recording_path(recording_name)),
    }


def run_stimulate_stage(
    record_analysis_path: Path,
    wells: Sequence[int],
    recording_electrodes: Sequence[int],
    repetitions: int,
    stim_frequency_hz: float,
    stim_neighbor_radius: int,
    max_probe_units: int,
    allow_empty_putative: bool = False,
    mock_putative_count: int = MIN_PUTATIVE_UNITS,
) -> Dict[str, Path]:
    try:
        record_analysis, putative_units = _load_putative_units(record_analysis_path)
    except RuntimeError as exc:
        if not allow_empty_putative:
            raise
        payload = json.loads(record_analysis_path.read_text(encoding="utf-8"))["record_analysis"]
        putative_units = _synthesize_putative_units(payload, max(mock_putative_count, MIN_PUTATIVE_UNITS))
        if len(putative_units) < MIN_PUTATIVE_UNITS:
            raise RuntimeError(
                "Debug mode enabled but unable to synthesize enough putative units from recording metadata"
            ) from exc
        record_analysis = payload
        print_info(
            f"Using synthesized putative units for debug mode: {len(putative_units)} units "
            f"(original analysis error: {exc})"
        )

    timestamp = _timestamp()
    manifest_path = RECORDING_DIR / f"cartpole_stimulate_{timestamp}_manifest.json"
    analysis_path = RECORDING_DIR / f"cartpole_stimulate_{timestamp}_analysis.json"
    selection_path = RECORDING_DIR / f"cartpole_selection_{timestamp}.json"

    initialize_system()
    mx.activate(list(wells))

    stimulus_recordings: List[Dict[str, Any]] = []
    probed_units = putative_units[:max_probe_units]
    for index, unit in enumerate(probed_units, start=1):
        print_step(
            f"Stimulate phase probe {index}/{len(probed_units)} on electrode {int(unit['electrode'])}"
        )
        try:
            stimulus_recordings.append(
                _probe_single_unit(
                    source_unit=unit,
                    recording_electrodes=recording_electrodes,
                    wells=wells,
                    repetitions=repetitions,
                    stim_frequency_hz=stim_frequency_hz,
                    stim_neighbor_radius=stim_neighbor_radius,
                )
            )
        except Exception as exc:
            print_info(
                f"Skipping electrode {int(unit['electrode'])}: failed to route/probe ({exc})"
            )

    manifest = {
        "record_analysis_path": str(record_analysis_path),
        "manifest_path": str(manifest_path),
        "record_analysis_summary": {
            "sample_rate_hz": record_analysis["sample_rate_hz"],
            "putative_unit_count": len(putative_units),
        },
        "stimulus_recordings": stimulus_recordings,
    }
    _json_dump(manifest_path, manifest)
    if len(stimulus_recordings) < 4:
        raise RuntimeError(
            f"Only {len(stimulus_recordings)} stimulation probes succeeded; need at least 4 to configure roles"
        )
    try:
        analyze_stimulation_manifest(
            manifest_path=manifest_path,
            output_path=analysis_path,
            selection_output_path=selection_path,
            first_order_window_ms=FIRST_ORDER_WINDOW_MS,
            multi_order_window_ms=MULTI_ORDER_WINDOW_MS,
            detection_multiplier=REALTIME_DETECTION_MULTIPLIER,
            burst_mad_multiplier=BURST_MAD_MULTIPLIER,
        )
    except RuntimeError as exc:
        if not allow_empty_putative:
            raise
        debug_reason = f"analysis_fallback: {exc}"
        print_info(f"Falling back to synthesized debug selection config ({debug_reason})")
        _write_debug_selection_from_manifest(
            manifest=manifest,
            record_analysis=record_analysis,
            selection_path=selection_path,
            analysis_path=analysis_path,
            reason=debug_reason,
        )
    print_success(f"Stimulate manifest written to {manifest_path}")
    print_success(f"Stimulate analysis written to {analysis_path}")
    print_success(f"Selection config written to {selection_path}")

    return {
        "manifest_path": manifest_path,
        "analysis_path": analysis_path,
        "selection_path": selection_path,
    }


def run_full_preexperiment(
    duration_s: float,
    wells: Sequence[int],
    recording_electrodes: Sequence[int],
    repetitions: int,
    stim_frequency_hz: float,
    stim_neighbor_radius: int,
    max_probe_units: int,
    allow_empty_putative: bool = False,
    mock_putative_count: int = MIN_PUTATIVE_UNITS,
) -> Dict[str, Path]:
    record_outputs = run_record_stage(
        duration_s=duration_s,
        wells=wells,
        recording_electrodes=recording_electrodes,
        analyze=True,
    )
    stimulate_outputs = run_stimulate_stage(
        record_analysis_path=record_outputs["analysis_path"],
        wells=wells,
        recording_electrodes=recording_electrodes,
        repetitions=repetitions,
        stim_frequency_hz=stim_frequency_hz,
        stim_neighbor_radius=stim_neighbor_radius,
        max_probe_units=max_probe_units,
        allow_empty_putative=allow_empty_putative,
        mock_putative_count=mock_putative_count,
    )
    outputs = dict(record_outputs)
    outputs.update(stimulate_outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Cartpole preexperiment: Record and Stimulate stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--duration-s", type=float, default=300.0)
    record_parser.add_argument("--wells", type=int, nargs="+", default=[0])
    record_parser.add_argument("--no-analysis", action="store_true")

    stimulate_parser = subparsers.add_parser("stimulate")
    stimulate_parser.add_argument("--record-analysis", type=Path, required=True)
    stimulate_parser.add_argument("--wells", type=int, nargs="+", default=[0])
    stimulate_parser.add_argument("--repetitions", type=int, default=STIM_REPETITIONS)
    stimulate_parser.add_argument("--stim-frequency-hz", type=float, default=STIM_FREQUENCY_HZ)
    stimulate_parser.add_argument("--stim-neighbor-radius", type=int, default=2)
    stimulate_parser.add_argument("--max-probe-units", type=int, default=16)
    stimulate_parser.add_argument("--allow-empty-putative", action="store_true")
    stimulate_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)

    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--duration-s", type=float, default=300.0)
    full_parser.add_argument("--wells", type=int, nargs="+", default=[0])
    full_parser.add_argument("--repetitions", type=int, default=STIM_REPETITIONS)
    full_parser.add_argument("--stim-frequency-hz", type=float, default=STIM_FREQUENCY_HZ)
    full_parser.add_argument("--stim-neighbor-radius", type=int, default=2)
    full_parser.add_argument("--max-probe-units", type=int, default=16)
    full_parser.add_argument("--allow-empty-putative", action="store_true")
    full_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)

    args = parser.parse_args()

    if args.command == "record":
        run_record_stage(
            duration_s=args.duration_s,
            wells=args.wells,
            recording_electrodes=RECORDING_ELECTRODES,
            analyze=not args.no_analysis,
        )
    elif args.command == "stimulate":
        run_stimulate_stage(
            record_analysis_path=args.record_analysis,
            wells=args.wells,
            recording_electrodes=RECORDING_ELECTRODES,
            repetitions=args.repetitions,
            stim_frequency_hz=args.stim_frequency_hz,
            stim_neighbor_radius=args.stim_neighbor_radius,
            max_probe_units=args.max_probe_units,
            allow_empty_putative=args.allow_empty_putative,
            mock_putative_count=args.mock_putative_count,
        )
    elif args.command == "full":
        run_full_preexperiment(
            duration_s=args.duration_s,
            wells=args.wells,
            recording_electrodes=RECORDING_ELECTRODES,
            repetitions=args.repetitions,
            stim_frequency_hz=args.stim_frequency_hz,
            stim_neighbor_radius=args.stim_neighbor_radius,
            max_probe_units=args.max_probe_units,
            allow_empty_putative=args.allow_empty_putative,
            mock_putative_count=args.mock_putative_count,
        )
    else:  # pragma: no cover - argparse guards this
        raise RuntimeError(f"Unsupported command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

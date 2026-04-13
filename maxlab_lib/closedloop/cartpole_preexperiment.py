#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import maxlab as mx

from cartpole_selection import analyze_spontaneous_recording, analyze_stimulation_manifest
from cartpole_setup import (
    RECORDING_DIR,
    RECORDING_ELECTRODES,
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
DEFAULT_MAX_PROBE_UNITS = 32
MIN_PUTATIVE_UNITS = 8
MAX_RECORDING_CHANNELS = 1024
SCAN_BATCH_DURATION_S = {
    "speed": 8.0,
    "balanced": 15.0,
    "coverage": 30.0,
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _recording_path(recording_name: str) -> Path:
    return RECORDING_DIR / f"{recording_name}.raw.h5"


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chunked(items: Sequence[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


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


def _discover_scan_candidates() -> List[int]:
    try:
        candidates = list(mx.electrode_rectangle_indices(0, 0, 219, 119))
        candidates = _unique_electrodes(candidates)
        if candidates:
            return candidates
    except Exception as exc:
        print_info(f"Failed to query rectangle electrode indices, fallback to range(26400): {exc}")
    return list(range(26400))


def _activity_score(row: Dict[str, Any]) -> float:
    firing_rate = float(row.get("firing_rate_hz", 0.0) or 0.0)
    p2p = float(row.get("peak_to_peak_amplitude", 0.0) or 0.0)
    median_peak = float(row.get("median_negative_peak", 0.0) or 0.0)
    amplitude = max(p2p, median_peak)
    return float((1.0 + firing_rate) * (1.0 + amplitude))


def _select_locked_recording_electrodes(
    activity_map_rows: Sequence[Dict[str, Any]],
    candidate_electrodes: Sequence[int],
) -> List[int]:
    ranked = sorted(activity_map_rows, key=_activity_score, reverse=True)
    seed_count = min(256, len(ranked))
    seeds = [int(item["electrode"]) for item in ranked[:seed_count] if int(item.get("electrode", -1)) >= 0]
    locked: List[int] = []
    used = set()

    for electrode in seeds:
        if len(locked) >= MAX_RECORDING_CHANNELS:
            break
        if electrode in used:
            continue
        locked.append(electrode)
        used.add(electrode)

    for electrode in seeds:
        if len(locked) >= MAX_RECORDING_CHANNELS:
            break
        for neighbor in mx.electrode_neighbors(electrode, 1):
            neighbor = int(neighbor)
            if neighbor in used:
                continue
            locked.append(neighbor)
            used.add(neighbor)
            if len(locked) >= MAX_RECORDING_CHANNELS:
                break

    for row in ranked:
        if len(locked) >= MAX_RECORDING_CHANNELS:
            break
        electrode = int(row.get("electrode", -1))
        if electrode < 0 or electrode in used:
            continue
        locked.append(electrode)
        used.add(electrode)

    for electrode in candidate_electrodes:
        if len(locked) >= MAX_RECORDING_CHANNELS:
            break
        electrode = int(electrode)
        if electrode in used:
            continue
        locked.append(electrode)
        used.add(electrode)

    return locked


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


def _run_recording_batch(
    recording_name: str,
    recording_electrodes: Sequence[int],
    wells: Sequence[int],
    duration_s: float,
    target_well: int,
) -> Tuple[Path, Path]:
    recording_path = _recording_path(recording_name)
    metadata_path = RECORDING_DIR / f"{recording_name}_meta.json"
    print_step(f"Routing recording electrodes ({len(recording_electrodes)} electrodes)")
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
        target_well=target_well,
        duration_s=duration_s,
    )
    saving = start_recording(recording_name, list(wells))
    try:
        print_info(f"Recording for {duration_s:.1f} seconds")
        time.sleep(duration_s)
    finally:
        stop_recording(saving)
    return recording_path, metadata_path


def _dedupe_activity_map_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_electrode: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        electrode = int(row.get("electrode", -1))
        if electrode < 0:
            continue
        existing = best_by_electrode.get(electrode)
        if existing is None or _activity_score(row) > _activity_score(existing):
            best_by_electrode[electrode] = dict(row)
    deduped = list(best_by_electrode.values())
    deduped.sort(key=_activity_score, reverse=True)
    return deduped


def run_record_stage(
    duration_s: float,
    wells: Sequence[int],
    analyze: bool,
    scan_budget: str = "balanced",
) -> Dict[str, Any]:
    if scan_budget not in SCAN_BATCH_DURATION_S:
        raise RuntimeError(f"Unsupported scan budget: {scan_budget}")

    timestamp = _timestamp()
    target_well = int(wells[0])
    scan_manifest_path = RECORDING_DIR / f"cartpole_record_scan_{timestamp}_manifest.json"
    activity_map_path = RECORDING_DIR / f"cartpole_record_scan_{timestamp}_activity_map.json"
    analysis_path = RECORDING_DIR / f"cartpole_record_{timestamp}_putative_units.json"
    recording_name = f"cartpole_record_{timestamp}"
    recording_path = _recording_path(recording_name)
    metadata_path = RECORDING_DIR / f"{recording_name}_meta.json"

    candidate_electrodes = _discover_scan_candidates()
    batches = list(_chunked(candidate_electrodes, MAX_RECORDING_CHANNELS))
    per_batch_duration_s = float(SCAN_BATCH_DURATION_S[scan_budget])
    print_info(
        "Two-phase chip scan enabled: "
        f"candidates={len(candidate_electrodes)}, batches={len(batches)}, "
        f"batch_size<={MAX_RECORDING_CHANNELS}, per_batch_duration_s={per_batch_duration_s:.1f}"
    )

    initialize_system()
    mx.activate(list(wells))

    batch_entries: List[Dict[str, Any]] = []
    activity_map_rows: List[Dict[str, Any]] = []

    for batch_index, batch_electrodes in enumerate(batches, start=1):
        batch_recording_name = f"cartpole_record_scan_{timestamp}_b{batch_index:03d}"
        print_step(f"Phase-1 scan batch {batch_index}/{len(batches)}")
        batch_recording_path, batch_metadata_path = _run_recording_batch(
            recording_name=batch_recording_name,
            recording_electrodes=batch_electrodes,
            wells=wells,
            duration_s=per_batch_duration_s,
            target_well=target_well,
        )
        batch_entry: Dict[str, Any] = {
            "batch_index": batch_index,
            "recording_path": str(batch_recording_path),
            "metadata_path": str(batch_metadata_path),
            "electrode_count": len(batch_electrodes),
        }
        if analyze:
            batch_analysis_path = RECORDING_DIR / f"{batch_recording_name}_putative_units.json"
            batch_record_analysis = analyze_spontaneous_recording(
                recording_path=batch_recording_path,
                metadata_path=batch_metadata_path,
                output_path=batch_analysis_path,
                threshold_multiplier=SPONTANEOUS_THRESHOLD_MULTIPLIER,
                refractory_samples=RECORD_REFRACTORY_SAMPLES,
                min_spike_count=20,
                top_k=MAX_RECORDING_CHANNELS,
            )["record_analysis"]
            batch_entry["analysis_path"] = str(batch_analysis_path)
            batch_entry["putative_unit_count"] = len(batch_record_analysis.get("putative_units", []))
            for row in batch_record_analysis.get("channel_metrics", []):
                electrode = int(row.get("electrode", -1))
                if electrode < 0:
                    continue
                enriched = dict(row)
                enriched["batch_index"] = batch_index
                activity_map_rows.append(enriched)
        batch_entries.append(batch_entry)

    deduped_activity_rows = _dedupe_activity_map_rows(activity_map_rows)
    if not deduped_activity_rows:
        deduped_activity_rows = [{"electrode": int(e), "firing_rate_hz": 0.0, "peak_to_peak_amplitude": 0.0} for e in candidate_electrodes[:MAX_RECORDING_CHANNELS]]
    locked_1024 = _select_locked_recording_electrodes(deduped_activity_rows, candidate_electrodes)

    activity_payload = {
        "scan_budget": scan_budget,
        "candidate_electrode_count": len(candidate_electrodes),
        "batch_count": len(batches),
        "activity_map": deduped_activity_rows,
        "locked_recording_electrodes": locked_1024,
    }
    _json_dump(activity_map_path, activity_payload)
    _json_dump(
        scan_manifest_path,
        {
            "record_strategy": "chip_scan",
            "scan_budget": scan_budget,
            "scan_batch_duration_s": per_batch_duration_s,
            "batch_size_limit": MAX_RECORDING_CHANNELS,
            "candidate_electrode_count": len(candidate_electrodes),
            "batch_count": len(batches),
            "activity_map_path": str(activity_map_path),
            "locked_recording_electrodes": locked_1024,
            "batches": batch_entries,
        },
    )
    print_success(f"Phase-1 scan manifest written to {scan_manifest_path}")
    print_success(f"Activity map written to {activity_map_path}")

    print_step("Phase-2 locked recording on selected 1024 electrodes")
    phase2_recording_path, phase2_metadata_path = _run_recording_batch(
        recording_name=recording_name,
        recording_electrodes=locked_1024,
        wells=wells,
        duration_s=duration_s,
        target_well=target_well,
    )
    print_success(f"Record stage raw file written to {phase2_recording_path}")
    print_success(f"Record stage metadata written to {phase2_metadata_path}")

    if analyze:
        record_analysis = analyze_spontaneous_recording(
            recording_path=phase2_recording_path,
            metadata_path=phase2_metadata_path,
            output_path=analysis_path,
            threshold_multiplier=SPONTANEOUS_THRESHOLD_MULTIPLIER,
            refractory_samples=RECORD_REFRACTORY_SAMPLES,
            min_spike_count=20,
            top_k=MAX_RECORDING_CHANNELS,
        )["record_analysis"]
        record_analysis["downstream_recording_electrodes"] = [int(item) for item in locked_1024]
        record_analysis["locked_recording_electrodes"] = [int(item) for item in locked_1024]
        record_analysis["activity_map_path"] = str(activity_map_path)
        record_analysis["scan_manifest_path"] = str(scan_manifest_path)
        record_analysis["scan_metadata"] = {
            "strategy": "chip_scan",
            "budget": scan_budget,
            "per_batch_duration_s": per_batch_duration_s,
            "candidate_electrode_count": len(candidate_electrodes),
            "batch_size_limit": MAX_RECORDING_CHANNELS,
            "batch_count": len(batches),
        }
        _json_dump(analysis_path, {"record_analysis": record_analysis})
        print_success(f"Record analysis written to {analysis_path}")

    return {
        "recording_path": phase2_recording_path,
        "metadata_path": phase2_metadata_path,
        "analysis_path": analysis_path,
        "recording_electrodes": locked_1024,
        "activity_map_path": activity_map_path,
        "scan_manifest_path": scan_manifest_path,
    }


def _load_record_analysis_payload(record_analysis_path: Path) -> Dict[str, Any]:
    payload = json.loads(record_analysis_path.read_text(encoding="utf-8"))
    if "record_analysis" not in payload:
        raise RuntimeError(f"Record analysis file is missing 'record_analysis': {record_analysis_path}")
    return payload["record_analysis"]


def _load_putative_units(record_analysis_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = _load_record_analysis_payload(record_analysis_path)
    units = list(payload.get("putative_units", []))
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
                "peak_to_peak_amplitude": float(
                    row.get("peak_to_peak_amplitude", 0.0)
                    if row.get("peak_to_peak_amplitude") is not None
                    else 0.0
                ),
                "score": float(row.get("score", 0.0) if row.get("score") is not None else 0.0),
            }
        )
        if len(synthesized) >= count:
            break
    return synthesized


def _resolve_recording_pool_for_stimulate(record_analysis_path: Path) -> List[int]:
    try:
        record_analysis = _load_record_analysis_payload(record_analysis_path)
    except Exception as exc:
        print_info(f"Falling back to default recording pool ({exc})")
        return list(RECORDING_ELECTRODES)

    for key in ("downstream_recording_electrodes", "locked_recording_electrodes", "recording_electrodes"):
        pool = record_analysis.get(key, [])
        if isinstance(pool, list) and pool:
            return _unique_electrodes([int(item) for item in pool])[:MAX_RECORDING_CHANNELS]

    return list(RECORDING_ELECTRODES)


def _inject_selection_metadata(
    selection_path: Path,
    recording_electrodes: Sequence[int],
    eta_ranked_units_top32: Sequence[Dict[str, Any]],
) -> None:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if "selection_config" not in payload or not isinstance(payload["selection_config"], dict):
        payload = {"selection_config": payload}
    payload["selection_config"]["recording_electrodes"] = [int(item) for item in recording_electrodes]
    payload["selection_config"]["eta_ranked_units_top32"] = list(eta_ranked_units_top32)
    _json_dump(selection_path, payload)


def _write_debug_selection_from_manifest(
    manifest: Dict[str, Any],
    record_analysis: Dict[str, Any],
    selection_path: Path,
    analysis_path: Path,
    reason: str,
    recording_electrodes: Sequence[int],
    eta_ranked_units_top32: Sequence[Dict[str, Any]],
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
            "recording_electrodes": [int(item) for item in recording_electrodes],
            "eta_ranked_units_top32": list(eta_ranked_units_top32),
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


def _power_down_all_stim_units() -> None:
    for stimulation_unit in range(32):
        mx.send(mx.StimulationUnit(stimulation_unit).power_up(False).connect(False))


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
    canonical_recording_electrodes = _unique_electrodes(list(recording_electrodes))[:MAX_RECORDING_CHANNELS]
    if not canonical_recording_electrodes:
        canonical_recording_electrodes = list(RECORDING_ELECTRODES)

    try:
        record_analysis, putative_units = _load_putative_units(record_analysis_path)
    except RuntimeError as exc:
        if not allow_empty_putative:
            raise
        payload = _load_record_analysis_payload(record_analysis_path)
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

    eta_ranked_units_top32 = list(record_analysis.get("eta_ranked_units_top32", []))
    probe_candidates = eta_ranked_units_top32 if eta_ranked_units_top32 else putative_units
    probe_source = "eta_top32" if eta_ranked_units_top32 else "putative_units"

    timestamp = _timestamp()
    manifest_path = RECORDING_DIR / f"cartpole_stimulate_{timestamp}_manifest.json"
    analysis_path = RECORDING_DIR / f"cartpole_stimulate_{timestamp}_analysis.json"
    selection_path = RECORDING_DIR / f"cartpole_selection_{timestamp}.json"

    initialize_system()
    mx.activate(list(wells))

    stimulus_recordings: List[Dict[str, Any]] = []
    probed_units = list(probe_candidates[:max_probe_units])
    print_info(
        f"Stimulate stage using probe_source={probe_source}, "
        f"recording_pool={len(canonical_recording_electrodes)}, probes={len(probed_units)}"
    )
    for index, unit in enumerate(probed_units, start=1):
        print_step(
            f"Stimulate phase probe {index}/{len(probed_units)} on electrode {int(unit['electrode'])}"
        )
        try:
            stimulus_recordings.append(
                _probe_single_unit(
                    source_unit=unit,
                    recording_electrodes=canonical_recording_electrodes,
                    wells=wells,
                    repetitions=repetitions,
                    stim_frequency_hz=stim_frequency_hz,
                    stim_neighbor_radius=stim_neighbor_radius,
                )
            )
        except Exception as exc:
            print_info(f"Skipping electrode {int(unit['electrode'])}: failed to route/probe ({exc})")

    manifest = {
        "record_analysis_path": str(record_analysis_path),
        "manifest_path": str(manifest_path),
        "record_analysis_summary": {
            "sample_rate_hz": record_analysis["sample_rate_hz"],
            "putative_unit_count": len(putative_units),
        },
        "recording_electrodes": [int(item) for item in canonical_recording_electrodes],
        "probe_source": probe_source,
        "probed_unit_count": len(probed_units),
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
        _inject_selection_metadata(
            selection_path=selection_path,
            recording_electrodes=canonical_recording_electrodes,
            eta_ranked_units_top32=eta_ranked_units_top32,
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
            recording_electrodes=canonical_recording_electrodes,
            eta_ranked_units_top32=eta_ranked_units_top32,
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
    repetitions: int,
    stim_frequency_hz: float,
    stim_neighbor_radius: int,
    max_probe_units: int,
    allow_empty_putative: bool = False,
    mock_putative_count: int = MIN_PUTATIVE_UNITS,
    scan_budget: str = "balanced",
) -> Dict[str, Any]:
    record_outputs = run_record_stage(
        duration_s=duration_s,
        wells=wells,
        analyze=True,
        scan_budget=scan_budget,
    )
    stimulate_outputs = run_stimulate_stage(
        record_analysis_path=record_outputs["analysis_path"],
        wells=wells,
        recording_electrodes=record_outputs["recording_electrodes"],
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
    record_parser.add_argument(
        "--scan-budget",
        type=str,
        choices=tuple(SCAN_BATCH_DURATION_S.keys()),
        default="balanced",
    )

    stimulate_parser = subparsers.add_parser("stimulate")
    stimulate_parser.add_argument("--record-analysis", type=Path, required=True)
    stimulate_parser.add_argument("--wells", type=int, nargs="+", default=[0])
    stimulate_parser.add_argument("--repetitions", type=int, default=STIM_REPETITIONS)
    stimulate_parser.add_argument("--stim-frequency-hz", type=float, default=STIM_FREQUENCY_HZ)
    stimulate_parser.add_argument("--stim-neighbor-radius", type=int, default=2)
    stimulate_parser.add_argument("--max-probe-units", type=int, default=DEFAULT_MAX_PROBE_UNITS)
    stimulate_parser.add_argument("--allow-empty-putative", action="store_true")
    stimulate_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)

    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--duration-s", type=float, default=300.0)
    full_parser.add_argument("--wells", type=int, nargs="+", default=[0])
    full_parser.add_argument("--repetitions", type=int, default=STIM_REPETITIONS)
    full_parser.add_argument("--stim-frequency-hz", type=float, default=STIM_FREQUENCY_HZ)
    full_parser.add_argument("--stim-neighbor-radius", type=int, default=2)
    full_parser.add_argument("--max-probe-units", type=int, default=DEFAULT_MAX_PROBE_UNITS)
    full_parser.add_argument("--allow-empty-putative", action="store_true")
    full_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)
    full_parser.add_argument(
        "--scan-budget",
        type=str,
        choices=tuple(SCAN_BATCH_DURATION_S.keys()),
        default="balanced",
    )

    args = parser.parse_args()

    if args.command == "record":
        run_record_stage(
            duration_s=args.duration_s,
            wells=args.wells,
            analyze=not args.no_analysis,
            scan_budget=args.scan_budget,
        )
    elif args.command == "stimulate":
        recording_pool = _resolve_recording_pool_for_stimulate(args.record_analysis)
        run_stimulate_stage(
            record_analysis_path=args.record_analysis,
            wells=args.wells,
            recording_electrodes=recording_pool,
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
            repetitions=args.repetitions,
            stim_frequency_hz=args.stim_frequency_hz,
            stim_neighbor_radius=args.stim_neighbor_radius,
            max_probe_units=args.max_probe_units,
            allow_empty_putative=args.allow_empty_putative,
            mock_putative_count=args.mock_putative_count,
            scan_budget=args.scan_budget,
        )
    else:  # pragma: no cover - argparse guards this
        raise RuntimeError(f"Unsupported command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None


def _require_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "This analysis stage requires 'h5py'. Install it in the experiment "
            "environment before running record/stimulate analysis."
        )


@dataclass
class PutativeUnit:
    channel: int
    electrode: int
    spike_count: int
    firing_rate_hz: float
    rms: float
    threshold: float
    median_negative_peak: float
    score: float


@dataclass
class RecordAnalysisResult:
    recording_path: str
    metadata_path: str
    sample_rate_hz: float
    duration_s: float
    threshold_multiplier: float
    refractory_samples: int
    min_spike_count: int
    putative_units: List[PutativeUnit]
    channel_metrics: List[Dict[str, float]]


@dataclass
class PairConnectivity:
    source_channel: int
    source_electrode: int
    target_channel: int
    target_electrode: int
    first_order_probability: float
    multi_order_probability: float


@dataclass
class StimulusProbeSummary:
    source_channel: int
    source_electrode: int
    resolved_stim_electrode: int
    repetitions: int
    burst_probability: float
    target_probabilities: List[PairConnectivity]


@dataclass
class SelectionConfig:
    encoding_stim_electrodes: List[int]
    decoding_left_electrodes: List[int]
    decoding_right_electrodes: List[int]
    training_stim_electrodes: List[int]
    source_record_analysis: str
    source_stim_analysis: str


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_recording_metadata(metadata_path: Path) -> Dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _recording_group(file_handle: "h5py.File", well: int, group_name: str) -> "h5py.Group":
    return file_handle[f"wells/well{well:03d}/rec0000/groups/{group_name}"]


def _recording_root(file_handle: "h5py.File", well: int) -> "h5py.Group":
    return file_handle[f"wells/well{well:03d}/rec0000"]


def _extract_event_frame_numbers(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return np.asarray([], dtype=np.int64)
    if events.dtype.names:
        candidate_names = ("frame", "frame_no", "frameNo", "frameno")
        for name in candidate_names:
            if name in events.dtype.names:
                return np.asarray(events[name], dtype=np.int64)
        return np.asarray(events[events.dtype.names[0]], dtype=np.int64)
    if events.ndim == 1:
        return np.asarray(events, dtype=np.int64)
    return np.asarray(events[:, 0], dtype=np.int64)


def _detect_negative_spikes(
    trace: np.ndarray,
    threshold: float,
    refractory_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    crossings = np.where(trace <= threshold)[0]
    if crossings.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    selected_indices: List[int] = []
    selected_peaks: List[float] = []
    last_index = -refractory_samples
    for crossing in crossings:
        if crossing - last_index < refractory_samples:
            continue
        window_end = min(trace.shape[0], crossing + max(4, refractory_samples // 4))
        window = trace[crossing:window_end]
        if window.size == 0:
            continue
        local_offset = int(np.argmin(window))
        peak_index = int(crossing + local_offset)
        selected_indices.append(peak_index)
        selected_peaks.append(float(trace[peak_index]))
        last_index = peak_index
    return np.asarray(selected_indices, dtype=np.int64), np.asarray(selected_peaks, dtype=np.float64)


def _waveform_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    first_norm = np.linalg.norm(first_centered)
    second_norm = np.linalg.norm(second_centered)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / (first_norm * second_norm))


def _build_average_waveform(
    trace: np.ndarray,
    spike_indices: np.ndarray,
    pre_samples: int = 20,
    post_samples: int = 40,
    max_waveforms: int = 128,
) -> np.ndarray:
    snippets: List[np.ndarray] = []
    for spike_index in spike_indices[:max_waveforms]:
        start = int(spike_index - pre_samples)
        end = int(spike_index + post_samples)
        if start < 0 or end >= trace.shape[0]:
            continue
        snippets.append(np.asarray(trace[start:end], dtype=np.float64))
    if not snippets:
        return np.asarray([], dtype=np.float64)
    return np.mean(np.stack(snippets, axis=0), axis=0)


def _spike_time_overlap(first: np.ndarray, second: np.ndarray, tolerance_samples: int = 3) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    matched = 0
    second_index = 0
    for spike in first:
        while second_index < second.size and second[second_index] < spike - tolerance_samples:
            second_index += 1
        if second_index < second.size and abs(int(second[second_index]) - int(spike)) <= tolerance_samples:
            matched += 1
    denom = max(first.size, second.size)
    return float(matched / denom) if denom else 0.0


def analyze_spontaneous_recording(
    recording_path: Path,
    metadata_path: Path,
    output_path: Path,
    threshold_multiplier: float = 5.0,
    refractory_samples: int = 20,
    min_spike_count: int = 20,
    top_k: int = 64,
) -> Dict[str, Any]:
    _require_h5py()
    metadata = _load_recording_metadata(metadata_path)
    sample_rate_hz = float(metadata["sample_rate_hz"])
    channel_to_electrode = {
        int(item["channel"]): int(item["electrode"])
        for item in metadata["recording_channels"]
    }

    with h5py.File(recording_path, "r") as file_handle:
        traces = np.asarray(_recording_group(file_handle, int(metadata["well"]), "all_channels")["raw"])

    duration_s = float(traces.shape[1] / sample_rate_hz)
    candidate_rows: List[Dict[str, Any]] = []
    waveforms: Dict[int, np.ndarray] = {}
    spike_indices_by_channel: Dict[int, np.ndarray] = {}

    for channel in range(traces.shape[0]):
        trace = np.asarray(traces[channel], dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(trace))))
        threshold = -threshold_multiplier * rms
        spike_indices, peaks = _detect_negative_spikes(trace, threshold, refractory_samples)
        spike_count = int(spike_indices.size)
        median_negative_peak = float(np.median(np.abs(peaks))) if peaks.size else 0.0
        firing_rate_hz = float(spike_count / duration_s) if duration_s > 0 else 0.0
        score = float(spike_count * median_negative_peak)
        spike_indices_by_channel[channel] = spike_indices
        waveforms[channel] = _build_average_waveform(trace, spike_indices)
        candidate_rows.append(
            {
                "channel": channel,
                "electrode": channel_to_electrode.get(channel, -1),
                "spike_count": spike_count,
                "firing_rate_hz": firing_rate_hz,
                "rms": rms,
                "threshold": threshold,
                "median_negative_peak": median_negative_peak,
                "score": score,
            }
        )

    candidate_rows.sort(key=lambda row: row["score"], reverse=True)
    selected_channels: List[int] = []
    putative_units: List[PutativeUnit] = []
    for row in candidate_rows:
        if len(putative_units) >= top_k:
            break
        channel = int(row["channel"])
        if row["spike_count"] < min_spike_count or row["electrode"] < 0:
            continue

        duplicate = False
        for selected_channel in selected_channels:
            waveform_corr = _waveform_correlation(waveforms[channel], waveforms[selected_channel])
            overlap = _spike_time_overlap(
                spike_indices_by_channel[channel],
                spike_indices_by_channel[selected_channel],
            )
            if waveform_corr >= 0.98 and overlap >= 0.5:
                duplicate = True
                break
        if duplicate:
            continue

        selected_channels.append(channel)
        putative_units.append(
            PutativeUnit(
                channel=channel,
                electrode=int(row["electrode"]),
                spike_count=int(row["spike_count"]),
                firing_rate_hz=float(row["firing_rate_hz"]),
                rms=float(row["rms"]),
                threshold=float(row["threshold"]),
                median_negative_peak=float(row["median_negative_peak"]),
                score=float(row["score"]),
            )
        )

    result = RecordAnalysisResult(
        recording_path=str(recording_path),
        metadata_path=str(metadata_path),
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        threshold_multiplier=threshold_multiplier,
        refractory_samples=refractory_samples,
        min_spike_count=min_spike_count,
        putative_units=putative_units,
        channel_metrics=candidate_rows,
    )
    payload = {
        "record_analysis": {
            "recording_path": result.recording_path,
            "metadata_path": result.metadata_path,
            "sample_rate_hz": result.sample_rate_hz,
            "duration_s": result.duration_s,
            "threshold_multiplier": result.threshold_multiplier,
            "refractory_samples": result.refractory_samples,
            "min_spike_count": result.min_spike_count,
            "putative_units": [asdict(unit) for unit in result.putative_units],
            "channel_metrics": result.channel_metrics,
        }
    }
    _json_dump(output_path, payload)
    return payload


def _has_threshold_crossing(
    trace: np.ndarray,
    start_index: int,
    end_index: int,
    threshold: float,
) -> bool:
    if end_index <= start_index:
        return False
    bounded_start = max(0, start_index)
    bounded_end = min(trace.shape[0], end_index)
    if bounded_end <= bounded_start:
        return False
    return bool(np.any(trace[bounded_start:bounded_end] <= threshold))


def _event_sample_indices(frame_numbers: np.ndarray, event_frame_numbers: np.ndarray) -> np.ndarray:
    if frame_numbers.size == 0 or event_frame_numbers.size == 0:
        return np.asarray([], dtype=np.int64)
    first_frame_no = int(frame_numbers[0])
    return np.asarray(event_frame_numbers - first_frame_no, dtype=np.int64)


def _mean_probability(mask: Sequence[bool]) -> float:
    if not mask:
        return 0.0
    return float(np.mean(np.asarray(mask, dtype=np.float64)))


def _median_plus_mad_threshold(values: Sequence[float], mad_multiplier: float = 3.0) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median + mad_multiplier * mad


def _select_roles_from_probe_results(
    probe_results: Sequence[StimulusProbeSummary],
    min_training_units: int = 4,
    max_training_units: int = 8,
) -> SelectionConfig:
    if len(probe_results) < 4:
        raise RuntimeError("Stimulate analysis produced too few stimulated units to configure cartpole roles")

    channel_to_probe = {probe.source_channel: probe for probe in probe_results}
    source_score_rows: List[Tuple[float, StimulusProbeSummary]] = []
    for probe in probe_results:
        if not probe.target_probabilities:
            continue
        mean_first = float(np.mean([pair.first_order_probability for pair in probe.target_probabilities]))
        mean_multi = float(np.mean([pair.multi_order_probability for pair in probe.target_probabilities]))
        score = mean_first + 0.25 * mean_multi - 0.5 * probe.burst_probability
        source_score_rows.append((score, probe))
    source_score_rows.sort(key=lambda row: row[0], reverse=True)

    encoding_sources: List[StimulusProbeSummary] = []
    for _, probe in source_score_rows:
        if len(encoding_sources) >= 2:
            break
        if any(existing.source_electrode == probe.source_electrode for existing in encoding_sources):
            continue
        encoding_sources.append(probe)
    if len(encoding_sources) != 2:
        raise RuntimeError("Unable to choose two encoding units from stimulate analysis")

    target_scores: Dict[int, float] = {}
    for source in encoding_sources:
        for pair in source.target_probabilities:
            if pair.target_electrode in {enc.source_electrode for enc in encoding_sources}:
                continue
            contribution = pair.first_order_probability + 0.25 * pair.multi_order_probability
            target_scores[pair.target_channel] = target_scores.get(pair.target_channel, 0.0) + contribution

    selected_target_channels: List[int] = []
    for target_channel, _ in sorted(target_scores.items(), key=lambda item: item[1], reverse=True):
        target_probe = channel_to_probe.get(target_channel)
        target_electrode = target_probe.source_electrode if target_probe is not None else None
        if target_electrode is None:
            for source in encoding_sources:
                for pair in source.target_probabilities:
                    if pair.target_channel == target_channel:
                        target_electrode = pair.target_electrode
                        break
                if target_electrode is not None:
                    break
        if target_electrode is None or target_electrode in {enc.source_electrode for enc in encoding_sources}:
            continue
        selected_target_channels.append(target_channel)
        if len(selected_target_channels) >= 2:
            break
    if len(selected_target_channels) != 2:
        raise RuntimeError("Unable to choose two decoding units from stimulate analysis")

    decoding_electrodes: List[int] = []
    for target_channel in selected_target_channels:
        electrode = None
        for source in encoding_sources:
            for pair in source.target_probabilities:
                if pair.target_channel == target_channel:
                    electrode = pair.target_electrode
                    break
            if electrode is not None:
                break
        if electrode is None:
            raise RuntimeError(f"Missing electrode mapping for decoding channel {target_channel}")
        decoding_electrodes.append(electrode)

    training_candidates: List[StimulusProbeSummary] = []
    used_electrodes = {enc.source_electrode for enc in encoding_sources} | set(decoding_electrodes)
    for _, probe in source_score_rows:
        if probe.source_electrode in used_electrodes:
            continue
        if probe.burst_probability >= 0.5:
            continue
        training_candidates.append(probe)

    training_electrodes = [probe.source_electrode for probe in training_candidates[:max_training_units]]
    if len(training_electrodes) < min_training_units:
        fallback = [
            probe.source_electrode
            for _, probe in source_score_rows
            if probe.source_electrode not in used_electrodes
        ]
        training_electrodes = fallback[: max(min_training_units, min(max_training_units, len(fallback)))]

    return SelectionConfig(
        encoding_stim_electrodes=[enc.source_electrode for enc in encoding_sources],
        decoding_left_electrodes=[decoding_electrodes[0]],
        decoding_right_electrodes=[decoding_electrodes[1]],
        training_stim_electrodes=training_electrodes,
        source_record_analysis="",
        source_stim_analysis="",
    )


def analyze_stimulation_manifest(
    manifest_path: Path,
    output_path: Path,
    selection_output_path: Path,
    first_order_window_ms: Tuple[float, float] = (10.0, 18.0),
    multi_order_window_ms: Tuple[float, float] = (10.0, 200.0),
    detection_multiplier: float = 3.0,
    burst_mad_multiplier: float = 3.0,
) -> Dict[str, Any]:
    _require_h5py()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_analysis_path = Path(manifest["record_analysis_path"])
    record_analysis = json.loads(record_analysis_path.read_text(encoding="utf-8"))["record_analysis"]
    record_metadata = _load_recording_metadata(Path(record_analysis["metadata_path"]))
    sample_rate_hz = float(record_analysis["sample_rate_hz"])
    putative_units = record_analysis["putative_units"]
    threshold_by_channel = {
        int(unit["channel"]): -detection_multiplier * float(unit["rms"])
        for unit in putative_units
    }

    first_start = int(first_order_window_ms[0] * sample_rate_hz / 1000.0)
    first_end = int(first_order_window_ms[1] * sample_rate_hz / 1000.0)
    multi_start = int(multi_order_window_ms[0] * sample_rate_hz / 1000.0)
    multi_end = int(multi_order_window_ms[1] * sample_rate_hz / 1000.0)

    probe_summaries: List[StimulusProbeSummary] = []
    for probe_item in manifest["stimulus_recordings"]:
        recording_path = Path(probe_item["recording_path"])
        with h5py.File(recording_path, "r") as file_handle:
            root = _recording_root(file_handle, int(record_metadata["well"]))
            traces = np.asarray(root["groups/all_channels/raw"])
            frame_numbers = np.asarray(root["groups/all_channels/frame_nos"])
            events = np.asarray(root["events"])

        event_frame_numbers = _extract_event_frame_numbers(events)
        event_sample_indices = _event_sample_indices(frame_numbers, event_frame_numbers)
        if event_sample_indices.size == 0:
            continue

        target_probabilities: List[PairConnectivity] = []
        responder_fractions: List[float] = []
        for event_index in event_sample_indices:
            responders = 0
            for target_unit in putative_units:
                target_channel = int(target_unit["channel"])
                trace = np.asarray(traces[target_channel], dtype=np.float64)
                threshold = float(threshold_by_channel[target_channel])
                if _has_threshold_crossing(trace, int(event_index + multi_start), int(event_index + multi_end), threshold):
                    responders += 1
            responder_fractions.append(responders / max(1, len(putative_units)))
        burst_threshold = _median_plus_mad_threshold(responder_fractions, burst_mad_multiplier)
        burst_events = [value >= burst_threshold for value in responder_fractions]

        for target_unit in putative_units:
            target_channel = int(target_unit["channel"])
            trace = np.asarray(traces[target_channel], dtype=np.float64)
            threshold = float(threshold_by_channel[target_channel])
            first_hits: List[bool] = []
            multi_hits: List[bool] = []
            for event_index in event_sample_indices:
                first_hits.append(
                    _has_threshold_crossing(trace, int(event_index + first_start), int(event_index + first_end), threshold)
                )
                multi_hits.append(
                    _has_threshold_crossing(trace, int(event_index + multi_start), int(event_index + multi_end), threshold)
                )
            target_probabilities.append(
                PairConnectivity(
                    source_channel=int(probe_item["source_channel"]),
                    source_electrode=int(probe_item["source_electrode"]),
                    target_channel=target_channel,
                    target_electrode=int(target_unit["electrode"]),
                    first_order_probability=_mean_probability(first_hits),
                    multi_order_probability=_mean_probability(multi_hits),
                )
            )

        probe_summaries.append(
            StimulusProbeSummary(
                source_channel=int(probe_item["source_channel"]),
                source_electrode=int(probe_item["source_electrode"]),
                resolved_stim_electrode=int(probe_item["resolved_stim_electrode"]),
                repetitions=int(probe_item["repetitions"]),
                burst_probability=_mean_probability(burst_events),
                target_probabilities=target_probabilities,
            )
        )

    selection = _select_roles_from_probe_results(probe_summaries)
    selection.source_record_analysis = str(record_analysis_path)
    selection.source_stim_analysis = str(output_path)

    payload = {
        "stimulate_analysis": {
            "manifest_path": str(manifest_path),
            "record_analysis_path": str(record_analysis_path),
            "sample_rate_hz": sample_rate_hz,
            "first_order_window_ms": list(first_order_window_ms),
            "multi_order_window_ms": list(multi_order_window_ms),
            "burst_detection_method": "median_plus_3mad" if burst_mad_multiplier == 3.0 else "median_plus_k_mad",
            "burst_mad_multiplier": burst_mad_multiplier,
            "probe_summaries": [
                {
                    "source_channel": probe.source_channel,
                    "source_electrode": probe.source_electrode,
                    "resolved_stim_electrode": probe.resolved_stim_electrode,
                    "repetitions": probe.repetitions,
                    "burst_probability": probe.burst_probability,
                    "target_probabilities": [asdict(pair) for pair in probe.target_probabilities],
                }
                for probe in probe_summaries
            ],
        }
    }
    _json_dump(output_path, payload)
    _json_dump(selection_output_path, {"selection_config": asdict(selection)})
    return payload

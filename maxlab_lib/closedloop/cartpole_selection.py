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
    peak_to_peak_amplitude: float
    footprint_electrodes: List[int]
    footprint_waveform_summary: Dict[str, float]
    log_rate_norm: float
    amp_norm: float
    eta_score: float
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
    multi_order_spike_count_mean: float
    event_count: int
    nonburst_event_count: int


@dataclass
class StimulusProbeSummary:
    source_channel: int
    source_electrode: int
    resolved_stim_electrode: int
    repetitions: int
    burst_probability: float
    target_probabilities: List[PairConnectivity]
    salpa_stats: Dict[str, Any]


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


SALPA_EVENT_LABEL = "stimulus_probe"
SALPA_POLY_ORDER = 2
SALPA_FIT_PRE_WINDOW_MS = (-1.5, -0.1)
SALPA_REPLACE_WINDOW_MS = (0.0, 1.0)
SALPA_FIT_POST_WINDOW_MS = (1.0, 2.5)


def _ms_to_samples(window_ms: float, sample_rate_hz: float) -> int:
    return int(round(window_ms * sample_rate_hz / 1000.0))


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_stimulus_event_frame_numbers(
    events: np.ndarray,
    stimulus_event_label: str = SALPA_EVENT_LABEL,
) -> Tuple[np.ndarray, str]:
    all_frame_numbers = _extract_event_frame_numbers(events)
    if all_frame_numbers.size == 0:
        return all_frame_numbers, "no_events"
    if not events.dtype.names:
        return all_frame_numbers, "all_events_unstructured"

    message_field = None
    for candidate in ("eventmessage", "message", "event_msg", "msg"):
        if candidate in events.dtype.names:
            message_field = candidate
            break
    if message_field is None:
        return all_frame_numbers, "all_events_no_message"

    selected_indices: List[int] = []
    for idx, message in enumerate(events[message_field]):
        if stimulus_event_label in _to_text(message):
            selected_indices.append(idx)
    if selected_indices:
        return np.asarray(all_frame_numbers[selected_indices], dtype=np.int64), "stimulus_probe_filtered"
    return all_frame_numbers, "all_events_fallback"


def _fit_event_local_polynomial(
    working_trace: np.ndarray,
    event_index: int,
    sample_rate_hz: float,
    poly_order: int = SALPA_POLY_ORDER,
    fit_pre_window_ms: Tuple[float, float] = SALPA_FIT_PRE_WINDOW_MS,
    replace_window_ms: Tuple[float, float] = SALPA_REPLACE_WINDOW_MS,
    fit_post_window_ms: Tuple[float, float] = SALPA_FIT_POST_WINDOW_MS,
) -> bool:
    trace_len = int(working_trace.shape[0])
    pre_start = event_index + _ms_to_samples(fit_pre_window_ms[0], sample_rate_hz)
    pre_end = event_index + _ms_to_samples(fit_pre_window_ms[1], sample_rate_hz)
    replace_start = event_index + _ms_to_samples(replace_window_ms[0], sample_rate_hz)
    replace_end = event_index + _ms_to_samples(replace_window_ms[1], sample_rate_hz)
    post_start = event_index + _ms_to_samples(fit_post_window_ms[0], sample_rate_hz)
    post_end = event_index + _ms_to_samples(fit_post_window_ms[1], sample_rate_hz)

    bounded_pre_start = max(0, min(trace_len, pre_start))
    bounded_pre_end = max(0, min(trace_len, pre_end))
    bounded_replace_start = max(0, min(trace_len, replace_start))
    bounded_replace_end = max(0, min(trace_len, replace_end))
    bounded_post_start = max(0, min(trace_len, post_start))
    bounded_post_end = max(0, min(trace_len, post_end))

    if bounded_replace_end <= bounded_replace_start:
        return False

    fit_pre_idx = np.arange(bounded_pre_start, bounded_pre_end, dtype=np.int64)
    fit_post_idx = np.arange(bounded_post_start, bounded_post_end, dtype=np.int64)
    fit_idx = np.concatenate([fit_pre_idx, fit_post_idx])
    if fit_idx.size <= poly_order:
        return False

    replace_idx = np.arange(bounded_replace_start, bounded_replace_end, dtype=np.int64)
    if replace_idx.size == 0:
        return False

    x_fit = fit_idx.astype(np.float64) - float(event_index)
    y_fit = working_trace[fit_idx].astype(np.float64)
    x_replace = replace_idx.astype(np.float64) - float(event_index)

    try:
        coeff = np.polyfit(x_fit, y_fit, poly_order)
        y_replace = np.polyval(coeff, x_replace)
    except Exception:
        return False

    working_trace[replace_idx] = y_replace
    return True


def _apply_event_aligned_salpa(
    trace: np.ndarray,
    event_sample_indices: np.ndarray,
    sample_rate_hz: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    working_trace = np.asarray(trace, dtype=np.float64).copy()
    fit_success_count = 0
    fit_fallback_count = 0
    for event_index in event_sample_indices:
        ok = _fit_event_local_polynomial(
            working_trace=working_trace,
            event_index=int(event_index),
            sample_rate_hz=sample_rate_hz,
        )
        if ok:
            fit_success_count += 1
        else:
            fit_fallback_count += 1
    return working_trace, {
        "event_count": int(event_sample_indices.size),
        "fit_success_count": int(fit_success_count),
        "fit_fallback_count": int(fit_fallback_count),
    }


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


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_value = float(min(values))
    max_value = float(max(values))
    if max_value <= min_value:
        return [0.0 for _ in values]
    scale = max_value - min_value
    return [float((value - min_value) / scale) for value in values]


def _estimate_footprint(
    center_channel: int,
    channel_to_electrode: Mapping[int, int],
    p2p_by_channel: Mapping[int, float],
    waveforms: Mapping[int, np.ndarray],
    max_footprint_electrodes: int = 12,
) -> Tuple[List[int], Dict[str, float]]:
    center_waveform = np.asarray(waveforms.get(center_channel, np.asarray([])), dtype=np.float64)
    center_p2p = float(p2p_by_channel.get(center_channel, 0.0))
    center_electrode = int(channel_to_electrode.get(center_channel, -1))
    threshold = max(1e-9, 0.25 * center_p2p)
    candidates: List[Tuple[float, int]] = []
    for channel, p2p in p2p_by_channel.items():
        electrode = int(channel_to_electrode.get(int(channel), -1))
        if electrode < 0:
            continue
        if float(p2p) < threshold:
            continue
        candidates.append((float(p2p), int(channel)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected_electrodes: List[int] = []
    seen = set()
    if center_electrode >= 0:
        selected_electrodes.append(center_electrode)
        seen.add(center_electrode)
    for _, channel in candidates:
        electrode = int(channel_to_electrode.get(channel, -1))
        if electrode < 0 or electrode in seen:
            continue
        selected_electrodes.append(electrode)
        seen.add(electrode)
        if len(selected_electrodes) >= max_footprint_electrodes:
            break

    electrode_to_channel = {int(e): int(c) for c, e in channel_to_electrode.items() if int(e) >= 0}
    center_peak_latency = int(np.argmax(center_waveform)) if center_waveform.size else -1
    center_trough_latency = int(np.argmin(center_waveform)) if center_waveform.size else -1
    corr_values: List[float] = []
    for electrode in selected_electrodes:
        if electrode == center_electrode:
            continue
        neighbor_channel = electrode_to_channel.get(int(electrode))
        if neighbor_channel is None:
            continue
        neighbor_waveform = np.asarray(waveforms.get(neighbor_channel, np.asarray([])), dtype=np.float64)
        if center_waveform.size == 0 or neighbor_waveform.size == 0:
            continue
        size = min(center_waveform.size, neighbor_waveform.size)
        first = center_waveform[:size]
        second = neighbor_waveform[:size]
        if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
            continue
        corr_values.append(float(np.corrcoef(first, second)[0, 1]))

    summary = {
        "center_min": float(np.min(center_waveform)) if center_waveform.size else 0.0,
        "center_max": float(np.max(center_waveform)) if center_waveform.size else 0.0,
        "center_p2p": center_p2p,
        "sample_count": float(center_waveform.size),
        "footprint_electrode_count": float(len(selected_electrodes)),
        "sta_peak_latency_samples": float(center_peak_latency),
        "sta_trough_latency_samples": float(center_trough_latency),
        "sta_neighbor_corr_mean": float(np.mean(np.asarray(corr_values, dtype=np.float64))) if corr_values else 0.0,
        "sta_neighbor_corr_max": float(np.max(np.asarray(corr_values, dtype=np.float64))) if corr_values else 0.0,
    }
    return selected_electrodes, summary


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
    p2p_by_channel: Dict[int, float] = {}

    for channel in range(traces.shape[0]):
        trace = np.asarray(traces[channel], dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(trace))))
        threshold = -threshold_multiplier * rms
        spike_indices, peaks = _detect_negative_spikes(trace, threshold, refractory_samples)
        spike_count = int(spike_indices.size)
        median_negative_peak = float(np.median(np.abs(peaks))) if peaks.size else 0.0
        firing_rate_hz = float(spike_count / duration_s) if duration_s > 0 else 0.0
        waveform = _build_average_waveform(trace, spike_indices)
        peak_to_peak_amplitude = float(np.ptp(waveform)) if waveform.size else 0.0
        score = float(spike_count * max(peak_to_peak_amplitude, median_negative_peak))
        spike_indices_by_channel[channel] = spike_indices
        waveforms[channel] = waveform
        p2p_by_channel[channel] = peak_to_peak_amplitude
        candidate_rows.append(
            {
                "channel": channel,
                "electrode": channel_to_electrode.get(channel, -1),
                "spike_count": spike_count,
                "firing_rate_hz": firing_rate_hz,
                "rms": rms,
                "threshold": threshold,
                "median_negative_peak": median_negative_peak,
                "peak_to_peak_amplitude": peak_to_peak_amplitude,
                "score": score,
            }
        )

    candidate_rows.sort(key=lambda row: row["score"], reverse=True)
    selected_rows: List[Dict[str, Any]] = []
    for row in candidate_rows:
        if len(selected_rows) >= top_k:
            break
        channel = int(row["channel"])
        if row["spike_count"] < min_spike_count or row["electrode"] < 0:
            continue

        duplicate_index: Optional[int] = None
        for index, selected_row in enumerate(selected_rows):
            selected_channel = int(selected_row["channel"])
            overlap = _spike_time_overlap(
                spike_indices_by_channel[channel],
                spike_indices_by_channel[selected_channel],
            )
            if overlap > 0.5:
                duplicate_index = index
                break
        if duplicate_index is not None:
            selected_p2p = float(selected_rows[duplicate_index].get("peak_to_peak_amplitude", 0.0))
            current_p2p = float(row.get("peak_to_peak_amplitude", 0.0))
            if current_p2p > selected_p2p:
                selected_rows[duplicate_index] = row
            continue

        selected_rows.append(row)

    selected_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    log_rates = [float(np.log1p(float(row["firing_rate_hz"]))) for row in selected_rows]
    amp_values = [float(row.get("peak_to_peak_amplitude", 0.0)) for row in selected_rows]
    log_rate_norm = _minmax_normalize(log_rates)
    amp_norm = _minmax_normalize(amp_values)

    putative_units: List[PutativeUnit] = []
    for row, r_hat, amp_hat in zip(selected_rows, log_rate_norm, amp_norm):
        channel = int(row["channel"])
        footprint_electrodes, footprint_summary = _estimate_footprint(
            center_channel=channel,
            channel_to_electrode=channel_to_electrode,
            p2p_by_channel=p2p_by_channel,
            waveforms=waveforms,
        )
        mu_amp = float(amp_hat)
        eta_score = float((1.0 + float(r_hat)) * (1.0 + 0.1 * abs(mu_amp)))
        putative_units.append(
            PutativeUnit(
                channel=channel,
                electrode=int(row["electrode"]),
                spike_count=int(row["spike_count"]),
                firing_rate_hz=float(row["firing_rate_hz"]),
                rms=float(row["rms"]),
                threshold=float(row["threshold"]),
                median_negative_peak=float(row["median_negative_peak"]),
                peak_to_peak_amplitude=float(row.get("peak_to_peak_amplitude", 0.0)),
                footprint_electrodes=[int(item) for item in footprint_electrodes],
                footprint_waveform_summary=footprint_summary,
                log_rate_norm=float(r_hat),
                amp_norm=float(amp_hat),
                eta_score=eta_score,
                score=float(row["score"]),
            )
        )

    eta_ranked_units = sorted([asdict(unit) for unit in putative_units], key=lambda item: item["eta_score"], reverse=True)
    eta_ranked_units_top32 = eta_ranked_units[:32]

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
            "eta_ranked_units": eta_ranked_units,
            "eta_ranked_units_top32": eta_ranked_units_top32,
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


def _count_negative_spikes_in_window(
    trace: np.ndarray,
    start_index: int,
    end_index: int,
    threshold: float,
    refractory_samples: int = 20,
) -> int:
    if end_index <= start_index:
        return 0
    bounded_start = max(0, start_index)
    bounded_end = min(trace.shape[0], end_index)
    if bounded_end <= bounded_start:
        return 0
    window = np.asarray(trace[bounded_start:bounded_end], dtype=np.float64)
    spike_indices, _ = _detect_negative_spikes(window, threshold, refractory_samples)
    return int(spike_indices.size)


def _summarize_target_connectivity(
    trace: np.ndarray,
    event_sample_indices: np.ndarray,
    threshold: float,
    first_start: int,
    first_end: int,
    multi_start: int,
    multi_end: int,
    burst_events: Sequence[bool],
) -> Dict[str, float]:
    first_hits: List[bool] = []
    multi_counts_nonburst: List[int] = []
    for event_idx, event_index in enumerate(event_sample_indices):
        event_base = int(event_index)
        first_spike_count = _count_negative_spikes_in_window(
            trace,
            event_base + first_start,
            event_base + first_end,
            threshold,
        )
        first_hits.append(first_spike_count > 0)

        multi_spike_count = _count_negative_spikes_in_window(
            trace,
            event_base + multi_start,
            event_base + multi_end,
            threshold,
        )
        if event_idx < len(burst_events) and not bool(burst_events[event_idx]):
            multi_counts_nonburst.append(multi_spike_count)

    first_order_probability = _mean_probability(first_hits)
    multi_order_probability = _mean_probability([count > 0 for count in multi_counts_nonburst])
    multi_order_spike_count_mean = (
        float(np.mean(np.asarray(multi_counts_nonburst, dtype=np.float64)))
        if multi_counts_nonburst
        else 0.0
    )
    return {
        "first_order_probability": first_order_probability,
        "multi_order_probability": multi_order_probability,
        "multi_order_spike_count_mean": multi_order_spike_count_mean,
        "event_count": float(len(event_sample_indices)),
        "nonburst_event_count": float(len(multi_counts_nonburst)),
    }


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
    min_training_units: int = 5,
    max_training_units: int = 12,
) -> Tuple[SelectionConfig, Dict[str, Any]]:
    if len(probe_results) < 4:
        raise RuntimeError("Stimulate analysis produced too few stimulated units to configure cartpole roles")

    source_by_electrode = {probe.source_electrode: probe for probe in probe_results}
    source_score_rows: List[Tuple[float, Dict[str, float], StimulusProbeSummary]] = []
    response_threshold = 0.1
    for probe in probe_results:
        if not probe.target_probabilities:
            continue
        first_values = [float(pair.first_order_probability) for pair in probe.target_probabilities]
        multi_values = [float(pair.multi_order_spike_count_mean) for pair in probe.target_probabilities]
        responder_ratio = _mean_probability([value >= response_threshold for value in first_values])
        mean_first = float(np.mean(first_values))
        mean_multi_count = float(np.mean(multi_values))
        selectivity = 1.0 - responder_ratio
        score = selectivity + 0.1 * mean_first + 0.05 * mean_multi_count - 0.5 * probe.burst_probability
        source_score_rows.append(
            (
                score,
                {
                    "selectivity": float(selectivity),
                    "responder_ratio_c1_ge_0p1": float(responder_ratio),
                    "mean_first_order_probability": mean_first,
                    "mean_multi_order_spike_count": mean_multi_count,
                    "burst_probability": float(probe.burst_probability),
                },
                probe,
            )
        )
    source_score_rows.sort(key=lambda row: (-row[0], int(row[2].source_electrode)))
    if len(source_score_rows) < 4:
        raise RuntimeError("Stimulate analysis produced too few valid source probes to configure cartpole roles")

    def _lookup_pair(source: StimulusProbeSummary, target_electrode: int) -> Optional[PairConnectivity]:
        for pair in source.target_probabilities:
            if int(pair.target_electrode) == int(target_electrode):
                return pair
        return None

    eligible_encoding = [row for row in source_score_rows if row[1]["burst_probability"] < 0.5]
    if len(eligible_encoding) < 2:
        eligible_encoding = source_score_rows

    encoding_sources: List[StimulusProbeSummary] = []
    best_pair_score: Optional[float] = None
    for first_idx in range(len(eligible_encoding)):
        for second_idx in range(first_idx + 1, len(eligible_encoding)):
            first_row = eligible_encoding[first_idx]
            second_row = eligible_encoding[second_idx]
            first_probe = first_row[2]
            second_probe = second_row[2]
            first_to_second = _lookup_pair(first_probe, second_probe.source_electrode)
            second_to_first = _lookup_pair(second_probe, first_probe.source_electrode)
            mutual_c1 = max(
                float(first_to_second.first_order_probability) if first_to_second is not None else 0.0,
                float(second_to_first.first_order_probability) if second_to_first is not None else 0.0,
            )
            pair_score = float(first_row[0] + second_row[0] - 1.5 * mutual_c1)
            if best_pair_score is None or pair_score > best_pair_score:
                best_pair_score = pair_score
                encoding_sources = [first_probe, second_probe]
    if len(encoding_sources) != 2:
        encoding_sources = [eligible_encoding[0][2], eligible_encoding[1][2]]

    used_electrodes = {probe.source_electrode for probe in encoding_sources}
    decoding_electrodes: List[int] = []
    decode_selection_notes: List[Dict[str, Any]] = []
    for source_probe in encoding_sources:
        candidates = [
            pair
            for pair in source_probe.target_probabilities
            if int(pair.target_electrode) not in used_electrodes
        ]
        if not candidates:
            continue
        best_c1 = max(candidates, key=lambda pair: float(pair.first_order_probability))
        if float(best_c1.first_order_probability) >= response_threshold:
            chosen = best_c1
            chosen_reason = "c1_strongest"
        else:
            chosen = max(
                candidates,
                key=lambda pair: (
                    float(pair.multi_order_spike_count_mean),
                    float(pair.first_order_probability),
                ),
            )
            chosen_reason = "cm_fallback"
        used_electrodes.add(int(chosen.target_electrode))
        decoding_electrodes.append(int(chosen.target_electrode))
        decode_selection_notes.append(
            {
                "source_electrode": int(source_probe.source_electrode),
                "selected_target_electrode": int(chosen.target_electrode),
                "first_order_probability": float(chosen.first_order_probability),
                "multi_order_spike_count_mean": float(chosen.multi_order_spike_count_mean),
                "reason": chosen_reason,
            }
        )

    if len(decoding_electrodes) != 2:
        raise RuntimeError("Unable to choose two decoding units from stimulate analysis")

    training_candidates: List[Tuple[float, StimulusProbeSummary]] = []
    for score, metrics, probe in source_score_rows:
        if probe.source_electrode in used_electrodes:
            continue
        if metrics["burst_probability"] >= 0.5:
            continue
        training_score = float(
            metrics["mean_first_order_probability"]
            + 0.25 * metrics["mean_multi_order_spike_count"]
            + 0.25 * metrics["selectivity"]
        )
        training_candidates.append((training_score, probe))

    training_candidates.sort(key=lambda row: (-row[0], int(row[1].source_electrode)))
    training_electrodes = [int(probe.source_electrode) for _, probe in training_candidates[:max_training_units]]
    if len(training_electrodes) < min_training_units:
        fallback = [
            int(probe.source_electrode)
            for _, _, probe in source_score_rows
            if int(probe.source_electrode) not in used_electrodes
        ]
        max_fallback = max(min_training_units, min(max_training_units, len(fallback)))
        training_electrodes = fallback[:max_fallback]

    selection = SelectionConfig(
        encoding_stim_electrodes=[int(enc.source_electrode) for enc in encoding_sources],
        decoding_left_electrodes=[int(decoding_electrodes[0])],
        decoding_right_electrodes=[int(decoding_electrodes[1])],
        training_stim_electrodes=training_electrodes,
        source_record_analysis="",
        source_stim_analysis="",
    )
    selection_audit = {
        "selection_constraints": {
            "encoding_requires_low_mutual_c1": True,
            "encoding_prefers_high_selectivity": True,
            "training_requires_burst_lt_0p5": True,
            "decode_prefers_c1_then_cm_fallback": True,
        },
        "ranked_source_rows": [
            {
                "source_electrode": int(probe.source_electrode),
                "source_channel": int(probe.source_channel),
                "score": float(score),
                **metrics,
            }
            for score, metrics, probe in source_score_rows
        ],
        "decode_selection": decode_selection_notes,
    }
    return selection, selection_audit


def analyze_stimulation_manifest(
    manifest_path: Path,
    output_path: Path,
    selection_output_path: Path,
    first_order_window_ms: Tuple[float, float] = (0.0, 10.0),
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
    sigma_multiplier = abs(float(detection_multiplier))

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

        event_frame_numbers, event_source = _extract_stimulus_event_frame_numbers(events)
        event_sample_indices = _event_sample_indices(frame_numbers, event_frame_numbers)
        if event_sample_indices.size == 0:
            continue

        target_probabilities: List[PairConnectivity] = []
        total_spike_counts: List[int] = []
        channel_trace_cache = {
            int(target_unit["channel"]): np.asarray(traces[int(target_unit["channel"])], dtype=np.float64)
            for target_unit in putative_units
        }
        salpa_trace_by_channel: Dict[int, np.ndarray] = {}
        threshold_by_channel: Dict[int, float] = {}
        salpa_success_total = 0
        salpa_fallback_total = 0
        for target_unit in putative_units:
            target_channel = int(target_unit["channel"])
            salpa_trace, salpa_stats = _apply_event_aligned_salpa(
                trace=channel_trace_cache[target_channel],
                event_sample_indices=event_sample_indices,
                sample_rate_hz=sample_rate_hz,
            )
            salpa_trace_by_channel[target_channel] = salpa_trace
            sigma = max(float(np.std(salpa_trace)), 1e-9)
            threshold_by_channel[target_channel] = -sigma_multiplier * sigma
            salpa_success_total += int(salpa_stats["fit_success_count"])
            salpa_fallback_total += int(salpa_stats["fit_fallback_count"])

        for event_index in event_sample_indices:
            total_spikes = 0
            for target_unit in putative_units:
                target_channel = int(target_unit["channel"])
                trace = salpa_trace_by_channel[target_channel]
                threshold = float(threshold_by_channel[target_channel])
                total_spikes += _count_negative_spikes_in_window(
                    trace,
                    int(event_index + multi_start),
                    int(event_index + multi_end),
                    threshold,
                )
            total_spike_counts.append(total_spikes)
        burst_threshold = _median_plus_mad_threshold(total_spike_counts, burst_mad_multiplier)
        burst_events = [float(value) >= burst_threshold for value in total_spike_counts]

        for target_unit in putative_units:
            target_channel = int(target_unit["channel"])
            trace = salpa_trace_by_channel[target_channel]
            threshold = float(threshold_by_channel[target_channel])
            summary = _summarize_target_connectivity(
                trace=trace,
                event_sample_indices=event_sample_indices,
                threshold=threshold,
                first_start=first_start,
                first_end=first_end,
                multi_start=multi_start,
                multi_end=multi_end,
                burst_events=burst_events,
            )
            target_probabilities.append(
                PairConnectivity(
                    source_channel=int(probe_item["source_channel"]),
                    source_electrode=int(probe_item["source_electrode"]),
                    target_channel=target_channel,
                    target_electrode=int(target_unit["electrode"]),
                    first_order_probability=float(summary["first_order_probability"]),
                    multi_order_probability=float(summary["multi_order_probability"]),
                    multi_order_spike_count_mean=float(summary["multi_order_spike_count_mean"]),
                    event_count=int(summary["event_count"]),
                    nonburst_event_count=int(summary["nonburst_event_count"]),
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
                salpa_stats={
                    "event_source": event_source,
                    "event_count": int(event_sample_indices.size),
                    "channel_count": int(len(putative_units)),
                    "fit_success_count_total": int(salpa_success_total),
                    "fit_fallback_count_total": int(salpa_fallback_total),
                },
            )
        )

    selection, selection_audit = _select_roles_from_probe_results(probe_summaries)
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
            "multi_order_metric": "mean_spike_count_per_stim_nonburst",
            "artifact_removal": {
                "method": "event_aligned_salpa_local_polynomial",
                "poly_order": SALPA_POLY_ORDER,
                "fit_pre_window_ms": list(SALPA_FIT_PRE_WINDOW_MS),
                "replace_window_ms": list(SALPA_REPLACE_WINDOW_MS),
                "fit_post_window_ms": list(SALPA_FIT_POST_WINDOW_MS),
                "sigma_threshold_rule": f"-{sigma_multiplier:g}*sigma_post_salpa",
                "sample_rate_hz": sample_rate_hz,
            },
            "probe_summaries": [
                {
                    "source_channel": probe.source_channel,
                    "source_electrode": probe.source_electrode,
                    "resolved_stim_electrode": probe.resolved_stim_electrode,
                    "repetitions": probe.repetitions,
                    "burst_probability": probe.burst_probability,
                    "salpa_stats": probe.salpa_stats,
                    "target_probabilities": [asdict(pair) for pair in probe.target_probabilities],
                }
                for probe in probe_summaries
            ],
            "selection_audit": selection_audit,
        }
    }
    _json_dump(output_path, payload)
    _json_dump(
        selection_output_path,
        {"selection_config": asdict(selection), "selection_audit": selection_audit},
    )
    return payload

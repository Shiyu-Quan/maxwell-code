# Cartpole 闭环刺激：Record / Stimulate / Train 关键流程与代码

这份文档聚焦你问的主问题：  
1) 如何记录电极信号  
2) 如何选择合适神经元（putative units 与角色分配）  
3) 如何进行数据分析  
4) 训练（adaptive training）何时触发、如何执行

---

## 总体流程图

```text
Record(自发活动采集)
  -> record_raw.h5 + record_meta.json
  -> analyze_spontaneous_recording
  -> putative_units.json

Stimulate(单元探测刺激)
  -> 对候选 source unit 逐个刺激并录制
  -> stimulate_manifest.json
  -> analyze_stimulation_manifest
  -> selection_config.json (encoding/decoding/training electrodes)

Train/Closed-loop(在线闭环)
  -> cartpole_selected_setup.py 读取 selection_config
  -> 生成 runtime config + stimulation sequences
  -> 启动 maxone_with_filter.cpp
  -> 每 200ms 解码活动并驱动 cartpole
  -> episode 结束后按条件触发 adaptive training
```

---

## 1) Record 阶段：如何记录电极信号

关键入口：`run_record_stage(...)`  
文件：`maxlab_lib/closedloop/cartpole_preexperiment.py`

核心代码（精简）：

```python
initialize_system()
mx.activate(list(wells))

array = configure_array(recording_electrodes, [])
array.download(list(wells))
mx.offset()
mx.clear_events()

_export_recording_metadata(...)
saving = start_recording(recording_name, list(wells))
try:
    time.sleep(duration_s)
finally:
    stop_recording(saving)
```

这一步完成的事情：
- 配置并下载 recording electrode routing
- 执行 offset 和 event 清理
- 录制 `raw.h5`
- 导出 `meta.json`（包含 electrode<->channel 映射）

相关代码位置：
- `run_record_stage`：`cartpole_preexperiment.py:85`
- 元数据导出 `_export_recording_metadata`：`cartpole_preexperiment.py:56`

---

## 2) Record 分析：如何从信号中选 putative units

关键入口：`analyze_spontaneous_recording(...)`  
文件：`maxlab_lib/closedloop/cartpole_selection.py`

核心分析逻辑（精简）：

```python
traces = np.asarray(...["raw"])
for channel in range(traces.shape[0]):
    trace = np.asarray(traces[channel], dtype=np.float64)
    rms = sqrt(mean(trace^2))
    threshold = -threshold_multiplier * rms
    spike_indices, peaks = _detect_negative_spikes(trace, threshold, refractory_samples)
    score = spike_count * median_negative_peak
```

随后按 `score` 排序，并做“重复单元去重”：

```python
waveform_corr = _waveform_correlation(...)
overlap = _spike_time_overlap(...)
if waveform_corr >= 0.98 and overlap >= 0.5:
    duplicate = True
```

产物：
- `*_putative_units.json`
- 包含每个候选通道的 `spike_count/rms/threshold/score`

相关代码位置：
- `analyze_spontaneous_recording`：`cartpole_selection.py:183`
- 去重逻辑：`cartpole_selection.py:242`

---

## 3) Stimulate 阶段：如何探测刺激响应并选择角色

### 3.1 逐 source unit 探测刺激

关键函数：`_probe_single_unit(...)`  
文件：`cartpole_preexperiment.py`

核心代码（精简）：

```python
candidate_electrodes = build_stim_candidate_electrodes([source_electrode], radius=stim_neighbor_radius)
array = configure_array(recording_electrodes, candidate_electrodes)
electrode_to_unit, resolved = connect_stim_units_to_stim_electrodes(...)
configure_and_powerup_stim_units([stim_unit])

saving = start_recording(recording_name, list(wells))
for _ in range(repetitions):
    sequence.send()
    time.sleep(1.0 / stim_frequency_hz)
stop_recording(saving)
```

这一步对每个 source channel 做：
- 路由与 unit 解析
- 重复脉冲刺激
- 同步记录刺激响应

代码位置：
- `_probe_single_unit`：`cartpole_preexperiment.py:262`

### 3.2 刺激响应分析与角色分配

关键函数：`analyze_stimulation_manifest(...)` + `_select_roles_from_probe_results(...)`  
文件：`cartpole_selection.py`

分析要点：
- 对每次刺激事件，统计 target unit 在窗口内是否越阈值
  - first-order window: 10~18ms
  - multi-order window: 10~200ms
- 计算每对 source-target 的响应概率
- 统计 burst probability（群体同步突发比例）
- 按 `score = mean_first + 0.25*mean_multi - 0.5*burst` 给 source 打分

角色选择逻辑：
- top 2 source -> `encoding_stim_electrodes`
- 从 target 里选 2 个 decoding（left/right）
- 剩余里选 training electrodes（过滤高 burst 候选）

代码位置：
- `analyze_stimulation_manifest`：`cartpole_selection.py:424`
- `_select_roles_from_probe_results`：`cartpole_selection.py:325`

产物：
- `cartpole_stimulate_<ts>_manifest.json`
- `cartpole_stimulate_<ts>_analysis.json`
- `cartpole_selection_<ts>.json`

---

## 4) Train 阶段：在线闭环里如何触发与执行训练

### 4.1 Python 侧：把 selection 配置转成 runtime config + sequences

关键入口：`run_selected_cartpole_experiment(...)`  
文件：`maxlab_lib/closedloop/cartpole_selected_setup.py`

核心步骤：

```python
selection = load_selection_config(...)
prepare_encoding_sequences(...)
training_pattern_names = prepare_training_sequences(...)
export_runtime_config(...)
cpp = CPPProcessManager(CPP_EXECUTABLE, config_path)
cpp.start()
cpp.send_start_signal()
```

代码位置：
- `run_selected_cartpole_experiment`：`cartpole_selected_setup.py:51`

### 4.2 C++ 侧：200ms 窗口闭环 + episode 结束后 adaptive training

关键函数：`updateWindow(...)`  
文件：`maxlab_lib/closedloop/maxone_with_filter.cpp`

核心逻辑（精简）：

```cpp
left_count  = sumCounts(spike_counts, decoding_left_channels);
right_count = sumCounts(spike_counts, decoding_right_channels);
left_rate  = ema_alpha * left_rate  + (1-ema_alpha) * left_count;
right_rate = ema_alpha * right_rate + (1-ema_alpha) * right_count;

force = clampUnitForce(..., left_rate, right_rate);
terminal = task.step(force);
emitRatePulse(encoding_left_sequence, ...);
emitRatePulse(encoding_right_sequence, ...);
```

episode 终止后：

```cpp
TrainingDecision decision = trainer.onEpisodeEnd(reward_seconds);
logger.writeEpisode(..., decision.delivered, decision.sequence_name, ...);
if (decision.delivered) {
    sendSequence(decision.sequence_name);
    training_until = now + training_window_ms;
}
```

代码位置：
- `updateWindow`：`maxone_with_filter.cpp:323`

---

## 5) 本次运行得到的“证据链”文件

你当前会话中已经生成并验证过的关键文件（示例）：
- `~/cartpole_experiments/cartpole_record_20260403_153608.raw.h5`
- `~/cartpole_experiments/cartpole_record_20260403_153608_putative_units.json`
- `~/cartpole_experiments/cartpole_stimulate_20260403_153658_manifest.json`
- `~/cartpole_experiments/cartpole_stimulate_20260403_153658_analysis.json`
- `~/cartpole_experiments/cartpole_selection_20260403_153658.json`
- `~/cartpole_experiments/cartpole_continuous_adaptive_20260403_153913_episodes.jsonl`

这些文件串起来，正好对应 Record -> Stimulate -> Train 的全链路。

---

## 6) 你之后可以用的标准命令

前置流程：

```bash
PYTHONPATH=/home/descfly/maxwell-code/.pydeps \
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 300 --wells 0 --repetitions 50 --max-probe-units 16
```

基于 `selection_config` 启动闭环：

```bash
PYTHONPATH=/home/descfly/maxwell-code/.pydeps \
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config /home/descfly/cartpole_experiments/cartpole_selection_<timestamp>.json \
  --duration 15 --mode cycled_adaptive --wells 0
```


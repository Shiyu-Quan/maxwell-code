# Cartpole 空芯片调试模式：关键代码段整理

本文档整理本次为“无神经元接入/putative units 为空”场景新增的关键代码实现，便于快速回顾。

## 1) 新增 CLI 参数（允许空 putative）

文件：`maxlab_lib/closedloop/cartpole_preexperiment.py`

```python
stimulate_parser.add_argument("--allow-empty-putative", action="store_true")
stimulate_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)

full_parser.add_argument("--allow-empty-putative", action="store_true")
full_parser.add_argument("--mock-putative-count", type=int, default=MIN_PUTATIVE_UNITS)
```

并在调用链中向下透传：

```python
run_stimulate_stage(
    ...,
    allow_empty_putative=args.allow_empty_putative,
    mock_putative_count=args.mock_putative_count,
)
```

## 2) putative units 为空时的合成逻辑

文件：`maxlab_lib/closedloop/cartpole_preexperiment.py`

```python
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
```

在 `run_stimulate_stage(...)` 中启用 fallback：

```python
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
```

## 3) 刺激分析选角失败时自动生成 debug selection_config

文件：`maxlab_lib/closedloop/cartpole_preexperiment.py`

```python
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
    ...
```

触发点：

```python
try:
    analyze_stimulation_manifest(...)
except RuntimeError as exc:
    if not allow_empty_putative:
        raise
    debug_reason = f"analysis_fallback: {exc}"
    print_info(f"Falling back to synthesized debug selection config ({debug_reason})")
    _write_debug_selection_from_manifest(...)
```

## 4) 同步测试脚本改为新版 config.json 接口

文件：`maxlab_lib/closedloop/test_sync.py`

关键点：
- 不再传旧版 positional args；
- 改为临时写 `config.json` 后调用：

```python
self.process = subprocess.Popen(
    [self.executable, self.config_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
```

- 识别同步日志：

```python
ready_marker = "[SYNC] Waiting for start signal"
start_ack_marker = "[SYNC] Start signal received"
```

## 5) 新增无硬件 smoke test

文件：`maxlab_lib/closedloop/no_hardware_smoke_test.py`

该脚本串联三步：
1. `make USE_QT=0 maxone_with_filter`
2. 运行 `test_sync.py`
3. 用最小 config 启动 `maxone_with_filter`，并验证出现无 `mxwserver` 的预期错误

---

## 建议命令

空芯片调试完整前置链路：

```bash
PYTHONPATH=/home/descfly/maxwell-code/.pydeps \
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 20 --wells 0 --repetitions 5 --max-probe-units 4 \
  --allow-empty-putative --mock-putative-count 8
```

使用生成的 selection 启动闭环：

```bash
PYTHONPATH=/home/descfly/maxwell-code/.pydeps \
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config /home/descfly/cartpole_experiments/cartpole_selection_<timestamp>.json \
  --duration 1 --mode continuous_adaptive --wells 0
```

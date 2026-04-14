# Cartpole 闭环实验流程总结（Record -> Stimulate -> Train）

本文总结当前仓库在 MaxOne 上执行闭环 Cartpole 的完整流程，并标注关键技术参数与实现细节。

## 1) 实验总流程

当前流程分 3 个阶段：

1. `Record`：两阶段全片扫描与自发活动记录，得到 putative units 与 `eta` 排序。
2. `Stimulate`：对候选刺激点逐个发脉冲，计算 `C1/Cm` 连通性并生成 `selection_config`。
3. `Train`：启动实时闭环 Cartpole，按 episode 成绩触发训练刺激（adaptive RL）。

---

## 2) Record 阶段（全片扫描 + 单元识别）

### 2.1 两阶段 chip scan

- Phase-1：分批扫描全片候选电极（每批最多 1024 通道）。
- 扫描预算 `--scan-budget`：
  - `speed`: 每批 8s
  - `balanced`: 每批 15s
  - `coverage`: 每批 30s
- 产物：
  - `activity_map`
  - `scan manifest`
  - `locked_recording_electrodes`（锁定 1024）

### 2.2 自发活动分析（spontaneous analysis）

- 阈值检测：`5 x RMS`
- 去重：当 spike time overlap `> 0.5` 时保留 `peak_to_peak_amplitude` 更大的单元
- 输出：`putative_units`、`eta_ranked_units`、`eta_ranked_units_top32`
- `eta` 指标：
  - `eta_score = (1 + r_hat) * (1 + 0.1 * abs(mu_amp))`

---

## 3) Stimulate 阶段（刺激测绘 + 连通性）

### 3.1 刺激发放协议

- 候选来源：优先 `eta_top32`
- 默认刺激重复：`50` 次/电极（可通过 CLI 调整）
- 默认刺激频率：`2 Hz`
- 刺激波形：双相脉冲（biphasic），正相在前，`0.2 mV/phase`，`200 us/phase`

### 3.2 连通性计算指标

- `C1`（first-order）：`0–10 ms` 窗口，统计至少 1 spike 的概率
- `Cm`（multi-order）：`10–200 ms` 窗口，统计非 burst trial 的平均 spike 数
- burst 检测：`median + 3 * MAD`
- burst 处理：burst trial 不计入 `Cm`

### 3.3 角色选择（selection）

- 生成 `selection_config`，包含：
  - `encoding_stim_electrodes`
  - `decoding_left_electrodes`
  - `decoding_right_electrodes`
  - `training_stim_electrodes`
  - `recording_electrodes`
- 额外写入 `selection_audit`（选择约束与来源证据）

---

## 4) Train 阶段（实时闭环 Cartpole）

### 4.1 实时循环时序

- `read_window_ms = 200`
- `training_window_ms = 300`
- cartpole 参数：
  - 角度失败阈值：`±16°`
  - 力尺度：`10 N`
  - 编码参数：`a=7`, `b=0.15`
- EMA 更新（论文对齐）：
  - `r_t = 0.2 * c_t + 0.8 * r_(t-1)`

### 4.2 训练刺激触发规则

- 仅在 episode 结束后判断是否给训练刺激
- 触发条件（严格）：`mean_5 < mean_20`
- 训练序列：
  - 双电极对组合
  - pair 内间隔 `5 ms`
  - 频率 `10 Hz`
  - 持续 `300 ms`
- value/eligibility：
  - `alpha = 0.3`
  - `gamma = 0.3`
  - `min_reward = 10`

---

## 5) SALPA 说明（重要）

- 论文中强调了 `SALPA` 去伪影。
- 当前代码在 **Stimulate 离线分析链路** 已实现事件对齐 SALPA：
  - 事件优先使用 `stimulus_probe` 标签（无标签时回退全部事件）
  - 局部二阶多项式拟合替换伪影窗
  - 在 SALPA 后信号上做 `3σ` 阈值检测，再计算 `C1/Cm`
  - 输出 `artifact_removal` 与每个 probe 的 `salpa_stats` 便于追溯
- 当前代码在 **C++ 实时闭环链路** 仍是配置入口与日志告警：
  - 默认：`iir`
  - 若设置 `salpa`，当前会打印告警并回退到 `iir`（SALPA 实时算法主体尚未接入）。
- 结论：论文中 Stimulus-response characterization 的 SALPA 要求已在离线刺激分析落地；实时闭环 SALPA 仍是后续 Phase-2 对齐项。

---

## 6) 实验执行命令（你当前常用）

### 6.1 完整前置流程（短测）

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 20 \
  --wells 0 \
  --scan-budget speed \
  --repetitions 5 \
  --stim-frequency-hz 2 \
  --stim-neighbor-radius 2 \
  --max-probe-units 4 \
  --allow-empty-putative \
  --mock-putative-count 8
```

### 6.2 闭环训练

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 15 \
  --mode cycled_adaptive \
  --wells 0
```

---

## 7) 关键输出文件

默认目录：`~/cartpole_experiments`

- `cartpole_record_scan_<timestamp>_manifest.json`
- `cartpole_record_scan_<timestamp>_activity_map.json`
- `cartpole_record_<timestamp>_putative_units.json`
- `cartpole_stimulate_<timestamp>_manifest.json`
- `cartpole_selection_<timestamp>.json`
- `cartpole_<mode>_<timestamp>_episodes.jsonl`

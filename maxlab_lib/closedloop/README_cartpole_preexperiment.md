# Cartpole 前置流程说明

## 简介

这份 README 只负责论文前置的 `Record -> Stimulate -> selection_config -> selected cartpole` 使用链路。

它不重复已有文档的职责：

- [README_cartpole.md](/D:/片上脑/maxwell-code/maxlab_lib/closedloop/README_cartpole.md)：主闭环运行说明
- [README_cartpole_code.md](/D:/片上脑/maxwell-code/maxlab_lib/closedloop/README_cartpole_code.md)：关键代码导读
- 本文档：前置筛选流程说明

## 相关脚本

- [cartpole_preexperiment.py](/D:/片上脑/maxwell-code/maxlab_lib/closedloop/cartpole_preexperiment.py)
- [cartpole_selection.py](/D:/片上脑/maxwell-code/maxlab_lib/closedloop/cartpole_selection.py)
- [cartpole_selected_setup.py](/D:/片上脑/maxwell-code/maxlab_lib/closedloop/cartpole_selected_setup.py)

`cartpole_preexperiment.py` 负责前置实验入口，`cartpole_selection.py` 负责离线分析，`cartpole_selected_setup.py` 负责消费 `selection_config` 并启动闭环 `cartpole`。

## 输出文件

所有文件默认写到 `~/cartpole_experiments`。

### Record 阶段

- `cartpole_record_<timestamp>.raw.h5`
- `cartpole_record_<timestamp>_meta.json`
- `cartpole_record_<timestamp>_putative_units.json`

### Stimulate 阶段

- `cartpole_stimulate_<timestamp>_manifest.json`
- `cartpole_stimulate_<timestamp>_analysis.json`
- `cartpole_selection_<timestamp>.json`

## 运行顺序

### 1. Record

```powershell
python maxlab_lib/closedloop/cartpole_preexperiment.py record --duration-s 300 --wells 0
```

这一步会：

- 做 spontaneous recording
- 导出记录元数据
- 默认运行 `putative units` 分析
- 生成 `*_putative_units.json`

如果你只想录制、不立刻分析：

```powershell
python maxlab_lib/closedloop/cartpole_preexperiment.py record --duration-s 300 --wells 0 --no-analysis
```

### 2. Stimulate

```powershell
python maxlab_lib/closedloop/cartpole_preexperiment.py stimulate `
  --record-analysis C:\Users\<用户名>\cartpole_experiments\cartpole_record_<timestamp>_putative_units.json `
  --wells 0 `
  --repetitions 50 `
  --stim-frequency-hz 2 `
  --max-probe-units 16
```

这一步会：

- 读取 `putative units` 结果
- 对前若干个候选 unit 逐个做刺激 probe
- 每个 unit 默认刺激 `50` 次，频率 `2 Hz`
- 生成 `manifest`、`analysis` 和 `selection_config`

### 3. Full

```powershell
python maxlab_lib/closedloop/cartpole_preexperiment.py full --duration-s 300 --wells 0
```

这一步会顺序执行 `Record -> putative units 分析 -> Stimulate -> selection_config 生成`。

### 4. 用 selection_config 启动 selected cartpole

```powershell
python maxlab_lib/closedloop/cartpole_selected_setup.py `
  --selection-config C:\Users\<用户名>\cartpole_experiments\cartpole_selection_<timestamp>.json `
  --duration 15 `
  --mode cycled_adaptive `
  --wells 0
```

这一步会读取 `selection_config`，解析 `encoding_stim_electrodes`、`training_stim_electrodes`、`decoding_left_electrodes`、`decoding_right_electrodes`，然后启动现有 C++ `cartpole` 闭环。

## 参数说明

### `cartpole_preexperiment.py record`

- `--duration-s`：spontaneous recording 时长，单位秒
- `--wells`：目标 well，默认 `0`
- `--no-analysis`：只录制，不运行 `putative units` 分析

### `cartpole_preexperiment.py stimulate`

- `--record-analysis`：`Record` 阶段生成的 `*_putative_units.json`
- `--wells`：目标 well，默认 `0`
- `--repetitions`：每个 unit 的刺激重复次数，默认 `50`
- `--stim-frequency-hz`：刺激频率，默认 `2 Hz`
- `--stim-neighbor-radius`：路由失败时向邻近电极扩展搜索的半径
- `--max-probe-units`：最多 probe 的候选 unit 数，默认 `16`

### `cartpole_preexperiment.py full`

- 组合使用 `record` 和 `stimulate` 的主要参数

### `cartpole_selected_setup.py`

- `--selection-config`：`Stimulate` 阶段输出的 `cartpole_selection_<timestamp>.json`
- `--duration`：闭环实验时长，单位分钟
- `--mode`：`cycled_adaptive` 或 `continuous_adaptive`
- `--wells`：目标 well
- `--show-gui`：是否显示 Qt `cartpole` viewer

## 当前近似与限制

### Record 阶段

- `putative units` 海选使用单通道负阈值 crossing
- 阈值采用 `5xRMS`
- 去重使用 waveform correlation + spike overlap 的启发式近似
- 还不是完整的 `spike-triggered averaging + footprint deduplication`

### Stimulate 阶段

- `evoked response` 用 raw trace 上的阈值 crossing 近似
- `first-order causal connectivity` 和 `multi-order connectivity` 基于窗口统计
- `burst probability` 基于刺激后响应单元比例阈值近似
- 某些电极如果无法路由到 stimulation unit，会被跳过

### Configure 阶段

- `selection_config` 由启发式打分生成
- `encoding / decoding / training` 角色选择还不是论文最终版自动搜索

## 依赖与注意事项

运行这套前置流程时，需要：

- `maxlab`
- `numpy`
- `h5py`
- MaxOne / MaxLab 可用的硬件和服务环境

特别注意：

- `Record` 和 `Stimulate` 的离线分析依赖 `h5py`
- 我当前检查到的环境里 `numpy` 可用，但 `h5py` 未安装
- 如果没有安装 `h5py`，分析阶段会直接报错

建议先确认：

```powershell
python -c "import numpy, h5py"
```

## 典型使用顺序

1. 先跑 `record`，确认 spontaneous recording 和 `putative units` 正常
2. 再跑 `stimulate`，确认能生成 `selection_config`
3. 最后用 `cartpole_selected_setup.py` 启动闭环 `cartpole`

如果你想快速打通整条链路：

```powershell
python maxlab_lib/closedloop/cartpole_preexperiment.py full --duration-s 300 --wells 0
python maxlab_lib/closedloop/cartpole_selected_setup.py --selection-config C:\Users\<用户名>\cartpole_experiments\cartpole_selection_<timestamp>.json --duration 15 --mode cycled_adaptive --wells 0
```

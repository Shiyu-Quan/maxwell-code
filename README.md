# maxwell-code

## Cartpole 闭环实验：完整执行流程（Record -> Stimulate -> Train）

本文档给出你当前仓库可直接执行的一整套流程，覆盖：
- 环境准备
- C++ 编译（GUI / 无 GUI）
- 前置实验（record + stimulate + selection）
- 使用 `selection_config` 启动闭环训练
- 空芯片（无神经元）调试方案
- 关键输出文件检查

---

## 0) 进入仓库

```bash
cd /home/descfly/maxwell-code
```

---

## 1) 环境准备

安装 Python 依赖：

```bash
python3 -m pip install --user --break-system-packages \
  ./maxlab-1.1.0-py3-none-any.whl numpy h5py
```

如果你使用本地依赖目录 `.pydeps`，运行前设置：

```bash
export PYTHONPATH=/home/descfly/maxwell-code/.pydeps:$PYTHONPATH
```

可选自检：

```bash
python3 maxlab_lib/closedloop/test_sync.py
```

---

## 2) 编译 C++ 程序

### 2.1 无 GUI 版本（推荐先打通）

```bash
cd /home/descfly/maxwell-code/maxlab_lib
make USE_QT=0 maxone_with_filter
```

### 2.2 GUI 版本（需要 Qt6）

```bash
cd /home/descfly/maxwell-code/maxlab_lib
sudo apt-get update
sudo apt-get install -y qt6-base-dev
make maxone_with_filter
```

如果出现 `cannot find -lQt6Core/-lQt6Widgets/-lQt6Gui`，说明 Qt6 开发库未安装完整，先执行上面的 `apt install` 再编译。

---

## 3) 前置实验（record + stimulate + selection）

### 3.1 分步执行

Record（记录自发活动）：

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_preexperiment.py record \
  --duration-s 300 \
  --wells 0
```

Stimulate（使用 record 结果做刺激探测；把 `RECORD_ANALYSIS_JSON` 替换成上一步生成的 `*_putative_units.json`）：

```bash
python3 maxlab_lib/closedloop/cartpole_preexperiment.py stimulate \
  --record-analysis RECORD_ANALYSIS_JSON \
  --wells 0 \
  --repetitions 50 \
  --stim-frequency-hz 2 \
  --stim-neighbor-radius 2 \
  --max-probe-units 16
```

### 3.2 一条命令跑完整前置流程

```bash
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 300 \
  --wells 0 \
  --repetitions 50 \
  --stim-frequency-hz 2 \
  --stim-neighbor-radius 2 \
  --max-probe-units 16
```

---

## 4) 启动闭环 Cartpole 训练（使用 selection_config）

把 `SELECTION_JSON` 替换成前面生成的 `cartpole_selection_<timestamp>.json`。

### 4.1 标准运行（无 GUI）

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 15 \
  --mode cycled_adaptive \
  --wells 0
```

持续训练模式：

```bash
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 15 \
  --mode continuous_adaptive \
  --wells 0
```

### 4.2 启动 GUI（你当前环境推荐命令）

如果在 Snap/IDE 环境下直接启动 GUI 报系统库冲突，可用干净环境启动：

```bash
cd /home/descfly/maxwell-code
env -i HOME=$HOME PATH=/usr/bin:/bin DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY LANG=C.UTF-8 \
  PYTHONPATH=/home/descfly/maxwell-code/.pydeps \
  python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 5 \
  --mode continuous_adaptive \
  --wells 0 \
  --show-gui
```

---

## 5) 空芯片（无神经元）调试流程

当芯片暂时没有接入神经元、`putative_units=0` 时，可用 mock 参数先验证闭环软件链路：

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 20 \
  --wells 0 \
  --repetitions 5 \
  --stim-frequency-hz 2 \
  --stim-neighbor-radius 2 \
  --max-probe-units 4 \
  --allow-empty-putative \
  --mock-putative-count 8
```

然后用生成的 `selection_config` 做短时训练验证：

```bash
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 1 \
  --mode continuous_adaptive \
  --wells 0 \
  --show-gui
```

---

## 6) 无硬件 smoke test（可选）

```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/no_hardware_smoke_test.py
```

---

## 7) 输出文件与结果检查

默认输出目录：

```bash
~/cartpole_experiments
```

常见产物：
- `cartpole_record_<timestamp>.raw.h5`
- `cartpole_record_<timestamp>_meta.json`
- `cartpole_record_<timestamp>_putative_units.json`
- `cartpole_stimulate_<timestamp>_manifest.json`
- `cartpole_stimulate_<timestamp>_analysis.json`
- `cartpole_selection_<timestamp>.json`
- `cartpole_<mode>_<timestamp>_config.json`
- `cartpole_<mode>_<timestamp>_episodes.jsonl`

快速查看最近结果：

```bash
ls -lt ~/cartpole_experiments | head -n 30
```

查看训练回合日志（每行一个 episode）：

```bash
tail -n 20 ~/cartpole_experiments/cartpole_*_episodes.jsonl
```

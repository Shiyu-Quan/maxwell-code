# maxwell-code

## Cartpole 闭环实验（STAR Methods 对齐版）

当前流程已统一为两阶段 `chip_scan`（不再使用 `fixed_pool`）：
1. Phase-1 全盘分批扫描（1024/批）生成 `activity_map`。
2. Phase-2 锁定 `locked_1024` 做持续记录，并产出 `eta_ranked_units_top32`。
3. Stimulate 默认优先使用 `eta_top32`。
4. 连通性分析默认：`C1=0-10ms`，`Cm=10-200ms`（剔除 burst 试次后统计平均 spike 数）。

## 1) 环境与编译

```bash
cd /home/descfly/maxwell-code
python3 -m pip install --user --break-system-packages ./maxlab-1.1.0-py3-none-any.whl numpy h5py
```

Headless 编译：
```bash
cd /home/descfly/maxwell-code/maxlab_lib
make USE_QT=0 maxone_with_filter
```

GUI 编译（需要 Qt6）：
```bash
cd /home/descfly/maxwell-code/maxlab_lib
sudo apt-get update
sudo apt-get install -y qt6-base-dev
make maxone_with_filter
```

## 2) 前置实验（Record + Stimulate + Selection）

推荐一条命令跑完整前置流程：
```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_preexperiment.py full \
  --duration-s 300 \
  --wells 0 \
  --scan-budget balanced \
  --repetitions 50 \
  --stim-frequency-hz 2 \
  --stim-neighbor-radius 2
```

说明：
- `--scan-budget`: `speed`(8s/批), `balanced`(15s/批), `coverage`(30s/批)
- `stimulate` 默认 `max_probe_units=32`（来源优先 `eta_top32`）

## 3) 启动闭环训练（使用 selection_config）

把 `SELECTION_JSON` 替换成上一步生成的 `cartpole_selection_<timestamp>.json`：
```bash
cd /home/descfly/maxwell-code
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 15 \
  --mode cycled_adaptive \
  --wells 0
```

持续训练：
```bash
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --duration 15 \
  --mode continuous_adaptive \
  --wells 0
```

按实验周期运行（`1`周期=`15`分钟训练+`45`分钟休息）：
```bash
python3 maxlab_lib/closedloop/cartpole_selected_setup.py \
  --selection-config SELECTION_JSON \
  --mode cycled_adaptive \
  --num-cycles 3 \
  --wells 0
```

GUI（Snap/IDE 冲突时使用干净环境）：
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

## 4) 无神经元调试

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

## 5) 结果文件

默认目录：
```bash
~/cartpole_experiments
```

关键产物：
- `cartpole_record_scan_<timestamp>_manifest.json`
- `cartpole_record_scan_<timestamp>_activity_map.json`
- `cartpole_record_<timestamp>_putative_units.json`（含 `locked_recording_electrodes`、`eta_ranked_units_top32`）
- `cartpole_stimulate_<timestamp>_manifest.json`
- `cartpole_selection_<timestamp>.json`
- `cartpole_<mode>_<timestamp>_episodes.jsonl`

快速检查：
```bash
ls -lt ~/cartpole_experiments | head -n 40
tail -n 20 ~/cartpole_experiments/cartpole_*_episodes.jsonl
```

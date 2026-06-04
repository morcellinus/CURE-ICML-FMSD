# CURE(Context management via Uncertainty-aware admission and Redundancy-aware Eviction) reproduction with TabICLv2

## Install

```bash
pip install -r requirements.txt
```

## Run Main 7-Dataset Setting

```bash
bash scripts/run_main_7datasets.sh
```

The main per-dataset `tau` values are stored in:

```text
config/expected_main_taus.json
```

## Run One Dataset

```bash
python scripts/run_tabicl_cure.py \
  --dataset noaa \
  --tau 0.4 \
  --data-root "$PWD/data/uspds" \
  --output-root "$PWD/results" \
  --device cuda \
  --threads 4
```

## Output

Results are saved under:

```text
results/{dataset}/
```

Each dataset directory contains:

```text
summary.json
run_config.json
step_metrics.csv
```

## Directory Structure

```text
.
├── data/
│   └── uspds/
├── config/
│   └── expected_main_taus.json
├── scripts/
│   ├── run_tabicl_cure.py
│   └── run_main_7datasets.sh
├── src/
│   └── stream/
├── requirements.txt
└── results/
```

## Checkpoint

The TabICLv2 model checkpoint is not included in this bundle.
Running the code requires either:
- online checkpoint resolution/download through the `tabicl` package, or
- a local checkpoint path supplied through `TABICL_MODEL_PATH`

For offline use, set:

```bash
export TABICL_MODEL_PATH=/path/to/tabicl-classifier-v2-20260212.ckpt
```

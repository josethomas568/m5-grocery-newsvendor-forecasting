#!/usr/bin/env bash
# Downloads the M5 dataset into ./data.
# Option A (default): Nixtla's public mirror of the M5 files (includes the evaluation-period actuals).
# Option B: Kaggle, `kaggle competitions download -c m5-forecasting-accuracy` (needs a Kaggle account).
set -euo pipefail
mkdir -p data
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Nixtla/m5-forecasts /tmp/m5-forecasts
unzip -o /tmp/m5-forecasts/datasets/m5.zip -d data

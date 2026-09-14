# Cold Diffusion — UIEB + DIV2K

## 1. 環境建置

```bash
git clone https://github.com/ABparadise33/Cold-Diffusion---GrayScaleTest.git
cd Cold-Diffusion---GrayScaleTest

# Ubuntu/Debian；非 root 帳號請在 apt-get 前加 sudo
apt-get update
apt-get install -y git-lfs python3-venv
git lfs install

# 建立 .venv、安裝依賴、檢查 CUDA，下載並準備 UIEB + DIV2K
bash scripts/setup_mixed.sh
```

## 2. 訓練

sRGB、0% 完整灰階、飽和度 1.0，UIEB reference 與 DIV2K HR 以 1：1 機率抽樣。
預設 10k 步，每 1k 步驗證、儲存 checkpoint 與除錯 log。

```bash
bash scripts/train_mixed_uieb_div2k.sh
```

中斷後續訓：

```bash
bash scripts/train_mixed_uieb_div2k.sh --resume
```

輸出：`outputs/uieb_div2k_rgb_fullgray_pilot/`。

## 3. Evaluate

使用混合訓練的最佳 checkpoint，在 UIEB Test90 上從 raw 的完整灰階版本反推：

```bash
.venv/bin/python evaluate.py \
  --checkpoint outputs/uieb_div2k_rgb_fullgray_pilot/checkpoints/best.pt \
  --raw-dir data/UIEB/raw-890 \
  --reference-dir data/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json --split test \
  --device cuda --original-size --batch-size 1 \
  --tile-size 256 --tile-overlap 32 --sampler paper_algorithm2 \
  --preview-count 4 --preview-max-side 512 \
  --extended-metrics --extended-metric-size 256 \
  --output-layout compact --output-dir evaluation/uieb_div2k_test90
```

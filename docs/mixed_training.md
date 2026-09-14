# UIEB + DIV2K full-gray pilot

This run uses **UIEB reference images** and **DIV2K HR images** as color targets.
Both are converted by the same sRGB channel-mean grayscale operator. It does not
train UIEB raw→reference restoration, and adds no saturation scaling or spatial masks.
Domain sampling is 50:50 in expectation (replacement), not exactly in every batch.
Defaults: fresh weights, T20, upstream ConvNeXt, FP32, effective batch32, 10k steps.
UIEB's existing train/val/test membership is preserved; test images are excluded.
The preparer rejects cross-split identical file bytes and paths, records SHA256s,
and makes namespaced symlinks without copying image data.

## 新 GPU instance：從 git clone 開始

適用於 Linux NVIDIA GPU（4090 24GB 或符合既有 CUDA 環境檢查的機器）。
使用供應商已安裝 NVIDIA driver 的映像；Python 3.10–3.12，建議至少40GB可用磁碟。
在持久磁碟的工作目錄開啟終端，先 clone 專案：

```bash
git clone https://github.com/ABparadise33/Cold-Diffusion---GrayScaleTest.git
cd Cold-Diffusion---GrayScaleTest
```

Ubuntu/Debian instance 若尚未安裝 Git LFS / venv，執行：

```bash
apt-get update
apt-get install -y git-lfs python3-venv
git lfs install
```

上述套件安裝需要 root；非 root 帳號在 `apt-get` 前加 `sudo`。
若連 `git` 都不存在，先以相同方式安裝 `git`，再執行 clone。

建置環境並準備兩個資料集：

```bash
bash scripts/setup_mixed_4090.sh
```

腳本會建立專案 `.venv`、安裝 PyTorch2.5.1 / torchvision0.20.1（CUDA12.1）
及專案依賴，執行 CUDA 檢查與測試，再沿用現有 UIEB mirror 下載器、官方 DIV2K
下載器，建立混合資料。UIEB使用既有 Hugging Face Git LFS mirror；已備妥官方
reference 的使用者可指定下方 `UIEB_REFERENCE_DIR`，略過 mirror 下載。
資料預設放在 `data/UIEB`、`data/DIV2K`、`data/UIEB_DIV2K`。
下載器產生的 UIEB split 另存於資料目錄，**不覆寫 repository 原有切分**。
腳本可重跑；會重查依賴與資料，已有完整資料時不重新下載。

接著開始訓練：

```bash
bash scripts/train_mixed_uieb_div2k_4090.sh
```

預設10k步。每次都是新訓練；已有輸出時會停止，續訓需明確加 `--resume`。
環境建置不會自動開始訓練。較新的、需要其他 PyTorch/CUDA 版本的 GPU
不屬於這份4090固定環境的保證範圍。

### 已掛載資料的 instance

在建置及訓練前，於同一終端 export 路徑；新終端需要重新 export：

```bash
export UIEB_REFERENCE_DIR=/absolute/path/UIEB/reference-890
export DIV2K_DATA_ROOT=/absolute/path/DIV2K
bash scripts/setup_mixed_4090.sh
bash scripts/train_mixed_uieb_div2k_4090.sh
```

DIV2K root 必須包含 `DIV2K_train_HR` 與 `DIV2K_valid_HR`。
若要把自動下載的 UIEB 放在其他位置，可 export `UIEB_DATA_ROOT`；
混合 symlink 目錄可用 `MIXED_DATA_ROOT` 設定。
訓練預設 Python 是 `.venv/bin/python`，可用 `MIXED_PYTHON` 覆寫。
模型來源 fingerprint 會在續訓時嚴格比對，不能混用舊的單資料集 checkpoint。

instance 關閉前，將 `outputs/uieb_div2k_rgb_fullgray_pilot/` 與 `logs/`
保存在持久磁碟；新 instance 僅 clone 專案不會取回 checkpoint 或訓練結果。

## Records

All model outputs are under `outputs/uieb_div2k_rgb_fullgray_pilot/`:

- `debug_diagnostics.jsonl`: append-only exposure counts every50 steps; fixed
  center-crop diagnostics at startup and every1000 steps plus final validation.
  Each domain contributes two fixed train and two fixed val images. These are
  diagnostic subsets, **not** full-domain validation aggregates.
- Each diagnostic records target/gray/direct online/direct EMA/sampled EMA RGB
  state range, nonfinite check, clipping fraction, unclipped chroma RMS,
  Lab a/b means, chroma ratio, Delta-E76 and PSNR. Trajectory contains every step.
  RGB/Lab roundtrip and full-gray endpoint errors check representation and operator.
- Training timestep histograms include actual t=T exposure; domain and timestep
  counts restart on resume and are explicitly labelled as such.
- `debug_previews/step_*/train|val/`: fixed target/gray/direct/sample strips.
- `full_gray_metrics.csv`: full combined validation center-crop metrics.
- `previews/step_*/`: existing random validation full-scene comparisons/trajectories.
- `run_manifest.json`: config, source fingerprints, dataset hashes, sampling rule.
- `checkpoints/`: latest/best/final weights; `logs/mixed_*.log` at repository root
  captures console output including failures.

## How to interpret failures

1. Target chroma absent or roundtrip error large: inspect source/normalization/export.
2. No t=T exposures after substantial training or nonzero endpoint color: inspect
   timestep selection and forward operator.
3. Direct output colored, sample gray: inspect first reverse step and subsequent
   chroma decay, clipping, update formula and timestep labels.
4. Both gray: inspect loss/fit/targets. Online vs EMA separates EMA lag from online fit.
5. Fixed train samples improve but fixed val samples do not: investigate
   generalization; subset observations alone do not prove overfitting.
6. More colorful but worse Delta-E: appearance change is not accurate restoration.

Diagnostics do not increase target saturation or alter endpoint sampling/loss.
Review at10k before a longer run. GPU training was not executed on the Mac.

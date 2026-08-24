# Cold Diffusion - GrayScaleTest

50k-step 小型實驗，驗證：

> 水下退化影像先到灰階 anchor，再逐步加回亮度與色彩資訊時，Cold
> Diffusion 的 middle steps 是否形成可解釋的修復路徑？

模型使用 UIEB 成對影像，並在 Lab 色彩空間建立
`reference -> (L_raw, 0, 0)` 的退化路徑。

## GPUTA RTX 4090：從 instance 終端機開始

以下假設 instance 已經能執行 `nvidia-smi`。不需要下載 Stable Diffusion、SAM
或 FlowIE 權重。

### 1. Clone

```bash
git clone https://github.com/ABparadise33/Cold-Diffusion---GrayScaleTest.git
cd Cold-Diffusion---GrayScaleTest
```

### 2. 建立環境

```bash
bash scripts/setup_gputa_4090.sh
```

腳本會用 Conda（若存在）或 Python venv 建立 `.venv`，安裝 PyTorch 2.5.1
CUDA 12.1、安裝本專案、確認 CUDA，最後執行測試。

成功時應看到：

```text
cuda_available: True
device: NVIDIA GeForce RTX 4090
ALL CHECKS PASSED
```

### 3. 下載 UIEB 並建立固定切分

```bash
.venv/bin/python tools/prepare_uieb.py
```

此步驟沿用
[Underwater_FlowIE](https://github.com/ABparadise33/Underwater_FlowIE)
使用的 Hugging Face UIEB mirror，下載約 1.49 GB，並建立：

```text
data/UIEB/raw-890/
data/UIEB/reference-890/
splits/uieb_seed42.json    # train 720 / val 80 / test 90
```

UIEB 限學術、非商業用途。原始下載入口與使用條款請見
[UIEB 官方頁面](https://li-chongyi.github.io/proj_benchmark.html)。

### 4. 先跑 CUDA smoke test

```bash
bash scripts/smoke_test_cuda.sh
```

它會測試 CUDA 訓練、存 checkpoint 與 `--resume`，約一分鐘內完成。

### 5. 開始 50k 訓練

```bash
bash scripts/train_4090.sh
```

4090 預設：

- 128×128
- physical batch 16
- gradient accumulation 1
- effective batch 16
- AMP FP16
- 每 5k 驗證並存 checkpoint

訓練 log 會顯示 `step/s`、預估剩餘時數和 peak VRAM。先用實測速度估時，
不要直接採用其他 GPU 的估計。

若要離開終端機仍繼續執行：

```bash
nohup bash scripts/train_4090.sh > train_50k.log 2>&1 &
tail -f train_50k.log
```

## 續訓

同一個實驗續到 100k：

```bash
bash scripts/train_4090.sh --resume auto --max-steps 100000
```

`latest.pt` 包含 model、EMA、optimizer、AMP scaler、step 與 RNG state。

## 記憶體不足時

維持 effective batch 16：

```bash
COLD_BATCH_SIZE=8 COLD_GRAD_ACCUM=2 bash scripts/train_4090.sh
```

## 輸出

```text
outputs/cold_gray_50k/
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── step_005000.pt
├── metrics.csv
├── samples/
└── trajectories/
```

第一個檢查點是 5k；第一個研究決策點是 10k；硬上限是 50k。

## 必要比較

確認 Cold 版本能正常訓練後，再使用相同 split 比較：

- `configs/rgb_oneshot_50k.yaml`
- `configs/gray_oneshot_50k.yaml`
- `configs/cold_gray_50k.yaml`

評估指標：PSNR、SSIM、Lab Delta-E76，以及 reverse steps 色差是否單調下降。

## 評估 checkpoint

更新舊 instance 的評測依賴：

```bash
git pull
.venv/bin/pip install -e .
```

高解析評測（4090 建議 512 crop、batch 1）：

```bash
.venv/bin/python evaluate.py \
  --checkpoint outputs/cold_gray_50k/checkpoints/best.pt \
  --raw-dir data/UIEB/raw-890 \
  --reference-dir data/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json \
  --split test \
  --device cuda \
  --image-size 512 \
  --batch-size 1 \
  --extended-metrics \
  --extended-metric-size 256 \
  --output-dir evaluation/cold_gray_50k_512
```

第一次執行會下載 LPIPS、MUSIQ、CLIP-IQA 等評測權重；之後會使用快取。

輸出包含：

```text
evaluation/cold_gray_50k_512/
├── metrics.json
├── training_curves.png
├── training_summary.json
├── extended_metrics.csv  # 90 張逐張結果；最後一列是 __mean__
├── extended_metrics.md   # 可直接閱讀的平均表
├── predictions/          # 90 張模型輸出
├── references/           # 與輸出對齊的 90 張 GT
├── batch_000.png          # raw → gray → prediction → reference
└── trajectory_000.png     # reverse 0/8 → 8/8，由左至右
```

`extended_metrics.csv` 包含 FlowIE 原本的 14 項：

- 越高越好：PSNR-Y/RGB、SSIM-Y/RGB、MS-SSIM、MUSIQ、CLIP-IQA、Entropy、MI、UIQM
- 越低越好：LPIPS、NIQE、KL、DUCD
- Cold 額外指標：Delta-E76 越低越好；trajectory monotonic 越高越好

目前 Test 90 的檔名與 `Underwater_FlowIE` 舊實驗是 90/90 相同。14 項舊指標固定在
256×256 計算；Delta-E76 與 monotonic 在 512 crop 計算。若要做嚴格的跨模型表格，
所有模型仍須使用相同的 crop/resize 流程重新評測。

`training_summary.json` 會根據最近三次 validation 提供：

- `continue_candidate`：PSNR、Delta-E 或 monotonic 仍有實質改善。
- `plateau_or_regressing`：三次 validation 已停滯或退步，先不要盲目續訓。

這只是續訓提示；仍需同時查看曲線和輸出圖片。比較不同 checkpoint 或 baseline
時，必須使用相同的 `--image-size`。

概念來源：[Cold Diffusion 論文](https://arxiv.org/abs/2208.09392)與
[官方實作](https://github.com/arpitbansal297/Cold-Diffusion-Models)。

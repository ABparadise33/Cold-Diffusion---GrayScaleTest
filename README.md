# Cold Diffusion - GrayScaleTest

50k-step 小型實驗，驗證：

> 水下退化影像先到灰階 anchor，再逐步加回亮度與色彩資訊時，Cold
> Diffusion 的 middle steps 是否形成可解釋的修復路徑？

模型使用 UIEB 成對影像，並在 Lab 色彩空間建立
`reference -> (L_raw, 0, 0)` 的退化路徑。

## 新實驗：DIV2K natural-image color prior

這個實驗不取代原本的 UIEB/Lab 實驗。它新增獨立的
`natural_rgb_colorization` 模式，測試只用自然影像學到的色彩 prior 能否轉移到
水下影像。

| Config | Training target | Output |
|---|---|---|
| `configs/div2k_rgb_sat1_50k.yaml` | DIV2K 原始色彩，saturation 1.0 | `outputs/div2k_rgb_sat1_50k/` |
| `configs/div2k_rgb_sat1_25_50k.yaml` | DIV2K saturation 1.25 | `outputs/div2k_rgb_sat1_25_50k/` |
| `configs/div2k_rgb_sat1_5_50k.yaml` | DIV2K saturation 1.5 | `outputs/div2k_rgb_sat1_5_50k/` |
| `configs/div2k_rgb_sat2_50k.yaml` | DIV2K saturation 2.0 | `outputs/div2k_rgb_sat2_50k/` |

這些倍率不是 HSV 濾鏡，也不會改寫硬碟上的 PNG。每次載入 crop 後即時計算：

```text
g = (R + G + B) / 3
target = clip(g + saturation_factor * (RGB - g), 0, 1)
```

1.0 是原圖；1.25、1.5、2.0 分別將 RGB channels 離灰階中心的距離放大
25%、50%、100%。超出 sRGB 範圍的值會 clipping。四組使用相同原圖、crop、
seed、grayscale input 與模型；Cold endpoint 固定使用未修改原圖的 `g`，只有
target chroma 不同。Clipping 是這輪 RGB baseline 的已知限制；之後可另做 Lab、
HSV 或 gamut-aware color-space ablation，但不要在本輪中途改定義。

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

### 3B. 下載 DIV2K HR 並驗證

只下載本實驗需要的官方 Train HR 800 張與 Validation HR 100 張：

```bash
.venv/bin/python tools/prepare_div2k.py --delete-archives
```

下載可續傳。ZIP 只會在成功解壓及驗證後，因 `--delete-archives` 被刪除；若要
保留 ZIP，移除該參數。完成後：

```text
data/DIV2K/DIV2K_train_HR/   # 0001.png ... 0800.png
data/DIV2K/DIV2K_valid_HR/   # 0801.png ... 0900.png
```

若已手動下載並解壓，可只驗證現有資料：

```bash
.venv/bin/python tools/prepare_div2k.py \
  --data-root /path/to/DIV2K \
  --skip-download
```

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

## DIV2K saturation 1.0 / 1.25 / 1.5 / 2.0 訓練

共同設定：128×128 random crop、T=20、seed 42、50k steps、每 5k validation 與
checkpoint。4090 預設 physical/effective batch 都是 16。

一次依序完成四組；同一張 GPU 不會同時執行多組：

```bash
bash scripts/train_div2k_4090.sh all
```

只檢查四組執行順序與指令，不啟動訓練：

```bash
DIV2K_DRY_RUN=1 bash scripts/train_div2k_4090.sh all
```

若關閉 terminal 後仍要繼續：

```bash
nohup bash scripts/train_div2k_4090.sh all > train_div2k_all.log 2>&1 &
tail -f train_div2k_all.log
```

如果整批被中斷，續跑已存在的 checkpoint，尚未開始的組別會自動從 step 0 開始：

```bash
bash scripts/train_div2k_4090.sh all --resume-if-exists
```

四組一起續訓／訓練到 100k：

```bash
bash scripts/train_div2k_4090.sh all --resume-if-exists --max-steps 100000
```

也可以只執行其中一組：

```bash
bash scripts/train_div2k_4090.sh 1.0
bash scripts/train_div2k_4090.sh 1.25
bash scripts/train_div2k_4090.sh 1.5
bash scripts/train_div2k_4090.sh 2.0
```

記憶體不足時維持 effective batch 16：

```bash
DIV2K_BATCH_SIZE=8 DIV2K_GRAD_ACCUM=2 \
  bash scripts/train_div2k_4090.sh all
```

訓練後使用相同 UIEB seed-42 Test 90 評測。UIEB reference 只參與評測，不會進入
模型 input：

```bash
.venv/bin/python evaluate.py \
  --checkpoint outputs/div2k_rgb_sat1_50k/checkpoints/best.pt \
  --raw-dir data/UIEB/raw-890 \
  --reference-dir data/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json \
  --split test \
  --device cuda \
  --original-size \
  --batch-size 1 \
  --output-dir evaluation/div2k_rgb_sat1_uieb
```

其他倍率評測時，把 checkpoint/output 名稱改成 `div2k_rgb_sat1_25_50k`、
`div2k_rgb_sat1_5_50k` 或 `div2k_rgb_sat2_50k`。Trajectory 會由左到右輸出
T=20 的完整 reverse process。

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

若舊環境曾出現 `No module named 'pkg_resources'`：

```bash
.venv/bin/python -m pip install "setuptools<82"
```

原始尺寸評測（保留每張影像的寬、高與長寬比）：

```bash
.venv/bin/python evaluate.py \
  --checkpoint outputs/cold_gray_50k/checkpoints/best.pt \
  --raw-dir data/UIEB/raw-890 \
  --reference-dir data/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json \
  --split test \
  --device cuda \
  --original-size \
  --batch-size 1 \
  --extended-metrics \
  --extended-metric-size 256 \
  --output-dir evaluation/cold_gray_50k_original
```

第一次執行會下載 LPIPS、MUSIQ、CLIP-IQA 等評測權重；之後會使用快取。

輸出包含：

```text
evaluation/cold_gray_50k_original/
├── metrics.json
├── training_curves.png
├── training_summary.json
├── direct_vs_algorithm2.md  # 先看這個：同 checkpoint 的平均比較
├── direct_vs_algorithm2.csv
├── direct_metrics.csv       # Direct 的 90 張逐張結果
├── extended_metrics.csv     # Algorithm 2 的 90 張逐張結果
├── direct_predictions/      # Direct 輸出
├── predictions/             # Algorithm 2 輸出
├── references/           # 與輸出對齊的 90 張 GT
├── batch_000.png          # raw → gray → Direct → Algorithm 2 → reference
└── trajectory_000.png     # reverse 0/8 → 8/8，由左至右
```

Direct 與 Algorithm 2 使用同一個 Cold checkpoint，不需要重新訓練：

- Direct：`R(gray, T)`，模型只執行一次。
- Algorithm 2：從相同 gray 開始，依序執行 `T → 1`。
- Trajectory：Algorithm 2 每一步的影像；不是第三個模型。

兩種方法都評測 FlowIE 原本的 14 項：

- 越高越好：PSNR-Y/RGB、SSIM-Y/RGB、MS-SSIM、MUSIQ、CLIP-IQA、Entropy、MI、UIQM
- 越低越好：LPIPS、NIQE、KL、DUCD
- Cold 額外指標：Delta-E76 越低越好；trajectory monotonic 越高越好

目前 Test 90 的檔名與 `Underwater_FlowIE` 舊實驗是 90/90 相同。`predictions/`、
`direct_predictions/` 與 `references/` 的 PNG 都維持原始尺寸，不會裁切或拉伸成
正方形。14 項舊指標只在記憶體內暫時縮放至 256×256，以延續 FlowIE 的舊評測
定義；Delta-E76 與 monotonic 使用完整原始尺寸。

`training_summary.json` 會根據最近三次 validation 提供：

- `continue_candidate`：PSNR、Delta-E 或 monotonic 仍有實質改善。
- `plateau_or_regressing`：三次 validation 已停滯或退步，先不要盲目續訓。

這只是續訓提示；仍需同時查看曲線和輸出圖片。跨模型比較時，統一使用
`--original-size --batch-size 1`。

概念來源：[Cold Diffusion 論文](https://arxiv.org/abs/2208.09392)與
[官方實作](https://github.com/arpitbansal297/Cold-Diffusion-Models)。

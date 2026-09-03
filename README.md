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

另有一組乾淨的色彩空間對照：

| Config | Training target | Color space | Output |
|---|---|---|---|
| `configs/div2k_lab_sat1_50k.yaml` | DIV2K 原始色彩，saturation 1.0 | CIE Lab | `outputs/div2k_lab_sat1_50k/` |

它與 RGB saturation 1.0 使用相同 DIV2K、seed、crop、模型、T=20 與 50k
設定；唯一實驗變因是 RGB bridge 改為先前 UIEB 實驗的 Lab bridge
`reference Lab -> (L_raw, 0, 0)`。

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

## Linux + RTX 4090 / CUDA：從終端機開始

以下假設主機已安裝 NVIDIA driver，並且能執行 `nvidia-smi`。不需要下載 Stable Diffusion、SAM
或 FlowIE 權重。

### 1. Clone

```bash
git clone https://github.com/ABparadise33/Cold-Diffusion---GrayScaleTest.git
cd Cold-Diffusion---GrayScaleTest
```

### 2. 建立環境

```bash
bash scripts/setup_cuda_4090.sh
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
使用完全相同的 Hugging Face repository 與 `git clone` / Git LFS 下載方式，
下載約 1.49 GB。腳本下載後會建立與 Underwater_FlowIE 相同的 seed-42 test 90：

```text
data/UIEB/raw-890/
data/UIEB/reference-890/
splits/uieb_seed42.json    # train 720 / val 80 / test 90
```

UIEB 限學術、非商業用途。原始下載入口與使用條款請見
[UIEB 官方頁面](https://li-chongyi.github.io/proj_benchmark.html)。

### 3B. 下載 DIV2K HR 並驗證

執行這一個指令即可：

```bash
.venv/bin/python tools/prepare_div2k.py --delete-archives
```

它會依序自動下載 Train HR 800 張與 Validation HR 100 張、解壓縮、確認全部
900 張圖片可用，最後刪除 ZIP 節省空間。你不需要另外解壓縮。完成後：

```text
data/DIV2K/DIV2K_train_HR/   # 0001.png ... 0800.png
data/DIV2K/DIV2K_valid_HR/   # 0801.png ... 0900.png
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

訓練後一次評測四組。它使用與 Underwater_FlowIE 相同的 UIEB seed-42 Test 90，
保留原始圖片尺寸，並計算全部擴充指標：

```bash
bash scripts/evaluate_div2k_uieb_4090.sh all
```

UIEB reference 只參與評測，不會進入模型 input。每組輸出 90 張 Algorithm 2、
90 張 Direct、90 張 reference，以及完整指標 CSV。輸出資料夾分別標示為
`1.00x`、`1.25x`、`1.50x`、`2.00x`。Trajectory 預覽會由左到右輸出 T=20
的完整 reverse process。

## DIV2K saturation 1.0：CIE Lab 對照

開始或從現有 `latest.pt` 續訓：

```bash
bash scripts/train_div2k_lab_4090.sh --resume-if-exists
```

若 VRAM 不足，維持 effective batch 16：

```bash
DIV2K_BATCH_SIZE=8 DIV2K_GRAD_ACCUM=2 \
  bash scripts/train_div2k_lab_4090.sh --resume-if-exists
```

訓練完成後，以相同 UIEB seed-42 Test 90、原始解析度與完整指標評測：

```bash
bash scripts/evaluate_div2k_lab_uieb_4090.sh
```

訓練輸出為 `outputs/div2k_lab_sat1_50k/`，評測輸出為
`evaluation/div2k_lab_sat_1.00x_uieb/`，不會覆蓋先前 RGB 實驗。

## 從零驗證目前 UIEB Lab 實作

這個 run 不讀取舊 checkpoint，輸出位置也與舊實驗分開：

```bash
bash scripts/train_uieb_lab_retrain_4090.sh
```

每 1,000 steps 固定輸出同一張 validation 圖：

```text
outputs/uieb_lab_retrain_50k/samples/step_001000.png
outputs/uieb_lab_retrain_50k/samples/step_002000.png
...
```

同時保留對應 trajectory。若中斷後要續訓，必須明確指定：

```bash
bash scripts/train_uieb_lab_retrain_4090.sh --resume auto
```

完成後評測相同 seed-42 Test90：

```bash
bash scripts/evaluate_uieb_lab_retrain_4090.sh
```

先確認它能恢復原本 UIEB 的偏淡／冷色結果；若新 UIEB 也變成褐色，停止
DIV2K，先查共用實作。

## UIEB-style Lab 設定重新訓練 DIV2K

UIEB regression 通過後，先只跑 factor 1：

```bash
bash scripts/train_div2k_uieb_style_4090.sh 1.0
```

這組改用與 UIEB 相同的 `cold_gray`、CIE Lab、T=8、128 crop、seed 42。
DIV2K 沒有退化／GT 成對資料，因此同一張原圖同時作為 raw 與 reference。
固定使用彩色內容明確的 `0803.png`，每 1,000 steps 寫入：

```text
outputs/div2k_uieb_style_lab_sat_1.00x_50k/samples/step_001000.png
```

只有 factor 1 的 validation 圖正常後，才執行其餘 Lab chroma target：

```bash
bash scripts/train_div2k_uieb_style_4090.sh 1.25
bash scripts/train_div2k_uieb_style_4090.sh 1.5
bash scripts/train_div2k_uieb_style_4090.sh 2.0
```

一次評測四組原始尺寸 DIV2K Val100：

```bash
bash scripts/evaluate_div2k_uieb_style_4090.sh all
```

### 先確認模型在 DIV2K 本身是否已經偏褐色

以官方 validation `0801.png`–`0900.png` 全部 100 張作為 input 與 reference，
一次輸出 Lab 與 RGB factor-1 的結果：

```bash
bash scripts/evaluate_div2k_validation_4090.sh both
```

只跑新的 Lab 模型：

```bash
bash scripts/evaluate_div2k_validation_4090.sh lab
```

每張 `predictions/*.png`（Algorithm 2）、`direct_predictions/*.png`（Direct）與
`references/*.png` 都保留 DIV2K 原始尺寸與長寬比。推論內部預設使用
512px、overlap 64px 的重疊 tiles，避免整張 HR 圖直接進入 4090 而 OOM；
只有六張橫向比較圖與 trajectory 預覽會縮到最長邊 512px。

若 DIV2K prediction 本身就普遍偏褐色，問題已存在於訓練／模型，不是 UIEB
domain shift。若 DIV2K 顏色正常、只有 UIEB 偏褐色，才支持 domain shift 解釋。

輸出：

```text
evaluation/div2k_lab_sat_1.00x_div2k_val/
evaluation/div2k_rgb_sat_1.00x_div2k_val/
```

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

更新舊環境的評測依賴：

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

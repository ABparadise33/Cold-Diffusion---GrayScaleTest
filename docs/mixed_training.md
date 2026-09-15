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

適用於 Linux NVIDIA GPU（RTX 3090／4090／5090）。
使用供應商已安裝 NVIDIA driver 的映像；Python 3.10–3.12，建議至少40GB可用磁碟。
在持久磁碟的工作目錄開啟終端，先 clone 專案：

```bash
git clone https://github.com/ABparadise33/Cold-Diffusion-in-UIE.git
cd Cold-Diffusion-in-UIE
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
bash scripts/setup_mixed.sh
```

腳本會建立專案 `.venv`，依GPU選擇配對版本：3090／4090 使用 PyTorch2.5.1 + torchvision0.20.1 + CUDA12.1；5090 使用 PyTorch2.7.1 + torchvision0.22.1 + CUDA12.8。
版本選擇記錄在 `.venv/gpu_environment.json`；安裝依賴時使用 constraints 防止版本被改回。
執行 CUDA 前向／反向檢查與測試，再沿用現有 UIEB mirror 下載器、官方 DIV2K
下載器，建立混合資料。UIEB使用既有 Hugging Face Git LFS mirror；已備妥官方
reference 的使用者可指定下方 `UIEB_REFERENCE_DIR`，略過 mirror 下載。
資料預設放在 `data/UIEB`、`data/DIV2K`、`data/UIEB_DIV2K`。
下載器產生的 UIEB split 另存於資料目錄，**不覆寫 repository 原有切分**。
腳本可重跑；會重查依賴與資料，已有完整資料時不重新下載。

接著開始訓練：

```bash
bash scripts/train_mixed_uieb_div2k.sh
```

預設10k步。每次都是新訓練；已有輸出時會停止，續訓需明確加 `--resume`。
環境建置不會自動開始訓練。無法辨識的 GPU 會停止並顯示錯誤，不會猜測型號。驅動由 instance 映像提供；CUDA 執行失敗時需先檢查驅動。

### 已掛載資料的 instance

在建置及訓練前，於同一終端 export 路徑；新終端需要重新 export：

```bash
export UIEB_REFERENCE_DIR=/absolute/path/UIEB/reference-890
export DIV2K_DATA_ROOT=/absolute/path/DIV2K
bash scripts/setup_mixed.sh
bash scripts/train_mixed_uieb_div2k.sh
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
  Each domain contributes up to eight fixed train and eight fixed val images (32 total). These are
  diagnostic subsets, **not** full-domain validation aggregates.
- Each diagnostic records target/gray/direct online/direct EMA/sampled EMA RGB
  state range, nonfinite check, clipping fraction, unclipped chroma RMS,
  Lab a/b means, chroma ratio, Delta-E76 and PSNR. Trajectory contains every step.
  RGB/Lab roundtrip and full-gray endpoint errors check representation and operator.
- Training timestep histograms include actual t=T exposure; domain and timestep
  counts restart on resume and are explicitly labelled as such.
- `debug_previews/step_*/train|val/`: fixed target/gray/direct/sample strips.
- `fixed_color_per_image.csv`: paired RGB MAE/RMSE, Lab Delta-E76/ab error,
  chroma, target tensor and preview hashes for gray/direct EMA/sample EMA.
- `fixed_color_summary.csv`: separate train/val and UIEB/DIV2K subset means.
  Each completed diagnostic pass has an ID to distinguish retries.
- `metrics.csv`: training L1 averaged over the logged window (default50 optimizer
  updates, each already averaged across gradient accumulation), plus validation.
- `full_gray_metrics.csv`: full combined validation center-crop metrics.
- `previews/step_*/`: existing random validation full-scene comparisons/trajectories.
- `run_manifest.json`: config, source fingerprints, dataset hashes, sampling rule.
- `checkpoints/`: latest/best and permanent checkpoints every10k (also final); `logs/mixed_*.log` at repository root
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
Pilot defaults to10k; --max-steps explicitly selects a longer budget. GPU training was not executed on the Mac.

GPU版本來源：[PyTorch安裝配對](https://pytorch.org/get-started/previous-versions/)、[PyTorch2.7 Blackwell支援](https://pytorch.org/blog/pytorch-2-7/)。三種GPU仍需在各自instance通過CUDA測試，Mac上的測試不代表實體GPU已驗證。

## 2026-09-15：整張推論與自然影像診斷

使用者的固定checkpoint對照確認tile造成方正色區。現在Evaluate與訓練的完整場景
preview均先整張推論；只有CUDA OOM才重新從原始輸入跑完整的tile反推。
`--tile-size`仍可用來明確重做歷史對照；`--oom-tile-size 0`則禁止fallback。
Evaluate預設OOM備援512/64，訓練preview沿用config的tile大小作為OOM備援。
非OOM錯誤不會被當成顯存問題吞掉；備援若仍失敗，會停止。
每張圖的實際路徑記錄在`其餘/inference_routes.jsonl`及metrics；preview記錄在preview.json。
舊checkpoint可繼續Evaluate；已知的前版preview fingerprint可安全續訓，其他訓練來源差異仍會拒絕。

```bash
bash scripts/diagnose_natural_fullgray.sh
```

預設同一個混合訓練best.pt、固定seed42、DIV2K train/val各8張完整圖片，分別用
`model`（目前訓練權重）和`ema`執行Direct與paper Algorithm2。輸入只由該自然圖片
轉完整灰階而得；target是原彩色圖片。這是自然影像補色診斷，不是水下增強分數。
可用`MIXED_CHECKPOINT`指定權重、`DIV2K_DATA_ROOT`指定資料根目錄，或加`--count 16`。
不啟動訓練，不增加飽和度，不使用局部彩色提示。

輸出在新建的`evaluation/natural_fullgray_XXXXXX/`：
- `source_manifest.json`：固定選圖、source/checkpoint SHA256、完成狀態。
- `train|val/model|ema/`：Direct、反推圖片、比較圖與每一步trajectory。
- `其餘/color_diagnostics.json`：逐圖target/gray/direct/sample彩度與色差、trajectory。
- 根目錄`summary.json`：四組子集平均、比gray改善的張數、發生tile備援的圖片。

先確認未切塊圖片中的自然物體是否也灰階，再看train/val及online/EMA/Direct/sample
差異。若某圖進入fallback，不能把方塊直接歸因於模型。8張子集不是全資料集驗證，
整張圖的平均彩度也不是物體辨識指標，需一起看固定比較圖。

## 2026-09-16：50k階段監測與報告

新訓練（0→50k）：

```bash
bash scripts/train_mixed_uieb_div2k.sh \
  --max-steps 50000 --output-dir outputs/uieb_div2k_rgb_fullgray_50k
```

同一輸出目錄中斷後續訓：

```bash
bash scripts/train_mixed_uieb_div2k.sh \
  --max-steps 50000 --output-dir outputs/uieb_div2k_rgb_fullgray_50k --resume
```

- 每1k驗證及固定圖記錄、更新latest/best；best以完整混合validation的Delta-E76選擇。
- 每10k保留 `checkpoints/step_010000.pt`、`step_020000.pt`、`step_030000.pt`、`step_040000.pt`、`step_050000.pt`。
- 每個step的固定圖在 `debug_previews/step_<六位step>/train|val/`，每張依序target/gray/direct EMA/sample EMA。輸出是同一張圖片的128中心crop，與完整圖片推論是否tile無關。
- 相同預覽原圖與tensor hash可逐階段比較。UIEB/DIV2K每個split各8張是小型診斷集，不是各資料集全體分數；完整混合validation仍另外記錄。

訓練到任何階段後可在另一個terminal產生報告（只讀紀錄，不載入GPU模型）：

```bash
.venv/bin/python tools/report_mixed_monitor.py \
  --run-dir outputs/uieb_div2k_rgb_fullgray_50k
```

輸出 `monitor_report/`：`fixed_train_trends.png`、`fixed_val_trends.png`、`training_validation.png`，以及 `image_comparisons/` 的10k/20k/30k/40k/50k逐張階段對照。尚未完成的階段列在 `report.json` 的missing_stages，不會捏造或補插圖片。`--steps 1000 2000 3000`可改看早期階段。工具驗證固定來源hash、逐圖平均與預覽hash，只採用對應的已完成診斷pass；資料或圖片不一致會停止。

若原本50k訓練已在舊版啟動，先等保存checkpoint並停止程序，git pull後用同一輸出目錄 `--resume`。這次只允許已知前版mixed fingerprint的監測升級，仍嚴格檢查模型、退化方式、資料、學習率與有效batch；不允許混用CIFAR或其他設定權重。新增CSV／圖片從更新後開始，錯過的10k checkpoint不會自動生成。

## CUDA launch failure 的定位與進度保護

使用者回報在完整場景預覽遇到 `CUDA error: unspecified launch failure`。這不是已確認的OOM，也無法只靠最後的degrade()堆疊判定根因。CUDA錯誤可能非同步回報；原程序失敗後須用新程序除錯。

新版先完成完整crop validation，**保存checkpoint和固定crop監測，再做完整場景預覽**。若之後預覽失敗，`latest.pt`已保留該更新；`debug_diagnostics.jsonl`記錄validation階段、錯誤類型及訊息。這不保證GPU驅動／硬體故障時仍能成功存檔；只有先前已成功保存的檔案可用。非OOM錯誤仍向外拋出，不能捕捉後繼續使用可能失效的CUDA context。

確認CUDA仍可使用：

```bash
nvidia-smi
.venv/bin/python tools/check_environment.py --require-cuda
```

用同步模式從最後成功存檔定位錯誤（除錯時速度會變慢）：

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 \
  bash scripts/train_mixed_uieb_div2k.sh \
  --max-steps 50000 --output-dir outputs/uieb_div2k_rgb_fullgray_50k --resume
```

若要隔離完整場景預覽這條路徑，可明確設定以下選項；固定crop監測、完整crop validation與checkpoint仍保留，而且會記錄預覽被停用。這是定位用開關，不是修復證據，也不是切塊：

```bash
MIXED_FULL_SCENE_PREVIEWS=0 CUDA_LAUNCH_BLOCKING=1 \
  bash scripts/train_mixed_uieb_div2k.sh \
  --max-steps 50000 --output-dir outputs/uieb_div2k_rgb_fullgray_50k --resume
```

如果連環境檢查都失敗，先處理instance/GPU健康狀態，不能用取消檢查或強制tile掩蓋。若尚未產生任何checkpoint，`--resume`不會憑空恢復進度；須從新輸出目錄重新開始。

依據：[PyTorch CUDA除錯說明](https://github.com/pytorch/pytorch/wiki/CUDA-basics)。上述保護與合成測試不代表已在使用者GPU重現或修復launch failure。

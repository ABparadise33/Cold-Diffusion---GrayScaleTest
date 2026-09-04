# DIV2K：官方 RGB colorization 全灰階 baseline，飽和度 1

這是新的 baseline，**不會改掉或覆蓋舊 UIEB／Lab／RGB 實驗**。
先在 DIV2K 看能不能從全灰階恢復多種色彩，不先做水下推論或飽和度 sweep。

## 在現有 instance 執行

確認這次修改已推送後，更新程式；沿用現有 CUDA 環境和已下載的 DIV2K：

```bash
git pull
.venv/bin/python -m pip install 'einops>=0.6,<1'
.venv/bin/python -m pytest -q tests/test_official_colorization.py
.venv/bin/python tools/check_official_baseline.py --device cuda
bash scripts/train_official_div2k_4090.sh
```

最後一個指令才開始正式訓練。`check_official_baseline.py` 只做真實大小模型的
一個合成資料 forward/backward，不會產生正式實驗 checkpoint。尚未建環境時，
先用主 README 的 `bash scripts/setup_cuda_4090.sh`。

資料仍是 `data/DIV2K/DIV2K_train_HR/` 800 張、`DIV2K_valid_HR/` 100 張。
訓練腳本先驗證已有資料，**不重新下載**。不同路徑使用 `DIV2K_DATA_ROOT`。

## 這次實際設定

| 項目 | 設定 |
|---|---|
| 模型 | 固定版本官方 ConvNeXt U-Net＋linear attention，56,615,708 參數 |
| 色彩／目標 | RGB，原圖本身，飽和度 1.0，不額外調色 |
| 正向 | `D(x,s) = (1-s/T)*x + (s/T)*mean_RGB(x)` |
| 推論起點 | **100% 灰階**，不保留 5%／10%／25% 色彩 |
| diffusion steps | T=20 |
| 訓練 | 從零，50k optimizer updates；Adam，lr=2e-5，FP32 |
| batch | 4 × 累積 8 = 有效 32；128px crop |
| EMA | .995、每 10 次更新；沿用官方 2k warmup 與零起算計數時機 |
| 種子 | 42 |
| 驗證 | 每 1k，全 100 張固定 128px center crops；另外存固定 0803 完整照片 |

這是**官方模型／RGB 退化＋論文反推的 DIV2K 50k 適配版**，不是論文的完整
CIFAR-10／CelebA 700k 復現。仍有 DIV2K、128px random crops/flip、較少訓練、
較小實體 batch 等差異。有效 batch32 對齊官方 128px 的16×2，而非 CIFAR32px
的32×2。Microbatch 內沒有 BatchNorm，但不能聲稱跨硬體／累積方式 bitwise 相同。

若 OOM，可改成同樣有效 batch32：

```bash
OFFICIAL_BATCH_SIZE=2 OFFICIAL_GRAD_ACCUM=16 bash scripts/train_official_div2k_4090.sh
```

4090 完整 batch 的 VRAM／時間尚待 instance 實測；不要用 CPU smoke 推估。
進度列會顯示 step/s、ETA 與 peak allocated VRAM；驗證也有額外耗時。

## 圖片與紀錄在哪裡

```text
outputs/div2k_official_rgb_sat1.00x_50k/
  run_manifest.json                # 設定、版本、檔案清單、硬體、模型大小
  metrics.csv                      # training L1、驗證 PSNR/SSIM/Delta-E
  training_curves.png               # 每次存檔刷新
  full_gray_metrics.csv             # Direct/Algorithm2/gray色差、chroma、完整驗證L1
  samples/
    step_001000.png                 # 固定0803完整場景比較圖
    preview.json                    # 圖片、原始尺寸、tile與顯示設定
    predictions/step_001000.png     # 原始寬高的Algorithm2預測
    direct_predictions/step_001000.png
  trajectories/step_001000.png      # 左→右，與sample分開
  checkpoints/
    latest.pt                      # 最近一次；可續訓
    best.pt                        # 全灰階驗證Delta-E最低，未必50k
    step_050000.pt                  # 固定50k比較用
```

完整照片以 256px tile／32px overlap 推論；這會改變 attention 可見範圍，是明確的
full-resolution 適配，不宣稱等於整張一次進模型。非8倍數邊界只補齊再裁回，不 resize。
獨立 prediction PNG 保持原尺寸。對照圖／trajectory 的**顯示縮圖**限制最長邊512，
不改獨立預測；要看細節請開 `samples/predictions/`。

不要只看 PSNR。至少一起看固定場景的色彩、`delta_e76`、`direct_delta_e76`、
`gray_delta_e76`、`chroma_ratio`（預測／目標平均彩度）。彩度接近1本身不代表色相正確。
`direct_val_l1` 是全灰階模型輸出的L1，`val_l1` 是反推最終L1；訓練L1混合不同t，
與兩者的難度不同。5k／10k先看趨勢，50k後再決定是否續訓。

checkpoint 約0.9GB／份；預設不存每個1k的編號權重，只留 latest／best、50k里程碑
與手動停止上限的終點。存檔前保留5GiB空間檢查；不會自動刪除任何舊實驗。

## 續訓：只能接這次的新 baseline

```bash
bash scripts/train_official_div2k_4090.sh --resume --max-steps 100000
```

這是備用指令，**不代表現在建議直接跑100k**。未帶 `--resume` 的再次執行會拒絕
覆蓋非空輸出。舊 UIEB／小 U-Net checkpoint 不相容，也不會自動載入。
續訓保留模型／optimizer／EMA／RNG；DataLoader重新建立，因此 crop／洗牌順序
不保證與不中斷訓練逐位元相同。模式、模型、原始碼指紋、資料清單與有效batch有檢查。

## 訓練完成後：固定50k的兩種反推對照

```bash
bash scripts/evaluate_official_div2k_4090.sh
```

它使用同一個 `step_050000.pt`，對 DIV2K Val100 全灰階原尺寸推論兩次：

- `paper_algorithm2`：論文公式，20次更新；主要 baseline。
- `official_code`：保留固定版本原始碼的時間索引，20次模型呼叫但19次有效更新。

兩者的 Direct 使用相同權重與全灰階輸入。**它們不是兩次訓練**，也不是
`gray_oneshot`。不因為都叫Algorithm2就把索引差異藏起來。
結果在 `evaluation/div2k_official_rgb_sat1.00x_step050000/<sampler>/`。
原圖尺寸預測全部輸出，即使不跑額外IQA；batch／trajectory 各自一個資料夾。
需要舊14項指標時在指令後加 `--extended-metrics`，可能下載其評分模型。
第一次可先用預設，避免評分模型下載阻擋看圖；核心分數會先存下來。

## 上游程式來源與移植範圍

固定來源：[`arpitbansal297/Cold-Diffusion-Models`, f8b1379151ff0cccba49112cf61d439bd4dd4ad9](https://github.com/arpitbansal297/Cold-Diffusion-Models/tree/f8b1379151ff0cccba49112cf61d439bd4dd4ad9/decolor-diffusion)。

- `diffusion/model/unet_convnext.py`：逐字保留模型本體，僅在檔頭增加來源說明。
  上游完整檔 SHA256：`e6d50fa6b45a6e7d017428c138680959f05e0c5143a2a9c2883c0f609fd7d571`。
  移植後外層只將狀態s轉成官方t=s−1、處理任意影像尺寸；沒有新增可訓練層。
- `diffusion/forward_process_impl.py` 的 `DeColorization`：正向1×1混色核的累積
  與本地封閉形式逐step核對；不把灰階權重換成常見亮度加權。
- `diffusion/diffusion.py`：對照 `q_sample`、`p_losses`、`sample_one_step`、Trainer。
  本地保留現代環境／續訓／報表支援，沒有整包引入不相關的 Snow、Apex、torchgeometry。
- `train.py`、`diffusion/model/get_model.py`：模型設定、Adam、EMA、FP32與梯度累積依據。

上游根目錄未提供可核實的LICENSE檔；此處保留作者與來源，不自行聲稱上游授權條款。
本次可確認「模型與反推存在實作差異」，但尚未證明這些差異就是偏褐色的唯一原因。

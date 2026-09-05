# DIV2K：官方 RGB colorization 全灰階 baseline，飽和度 1

這是新的 baseline，**不會改掉或覆蓋舊 UIEB／Lab／RGB 實驗**。
在 DIV2K 訓練後，使用同一份權重推論 UIEB Test90，檢查自然影像學到的色彩能否
轉移到水下影像；DIV2K 驗證保留作為域內對照。這輪仍只做飽和度 1。

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
| 驗證 | 每 1k，全 100 張固定 128px center crops；另外隨機抽 5 張完整照片存預覽 |

這是**官方模型／RGB 退化＋論文反推的 DIV2K 50k 適配版**，不是論文的完整
CIFAR-10／CelebA 700k 復現。仍有 DIV2K、128px random crops/flip、較少訓練、
較小實體 batch 等差異。有效 batch32 對齊官方 128px 的16×2，而非 CIFAR32px
的32×2。Microbatch 內沒有 BatchNorm，但不能聲稱跨硬體／累積方式 bitwise 相同。

若 OOM，可改成同樣有效 batch32：

```bash
OFFICIAL_BATCH_SIZE=2 OFFICIAL_GRAD_ACCUM=16 bash scripts/train_official_div2k_4090.sh
```

### 自動找能跑的 batch（新增）

```bash
bash scripts/train_official_div2k_4090.sh --auto-batch
```

先在**獨立測試程序**依序測 `32 → 16 → 8 → 4 → 2 → 1`。確定 CUDA OOM
就結束該程序、完全釋放它的顯存，再往下試。挑第一個能通過、且保留
`max(2 GiB, 顯卡容量10%)` 餘裕的 batch，自動設定 `grad_accum = 32 / batch`，
之後才開始正式訓練。因此有效batch仍是32，不改學習率、模型、飽和度與50k上限。

試跑包含3次合成資料 optimizer update、gradient accumulation、Adam狀態、EMA、
全灰階crop驗證與trajectory指標。它使用真實模型和128px尺寸，但不是完整DIV2K
或完整場景preview壓力測試；保留顯存餘裕仍不保證其他程序搶顯存時永遠不OOM。
選出的是**能通過的最大候選batch**，不是任意整數的極限，也不保證是最快batch。
為維持有效batch32，不測64以上。

**只有CUDA OOM或顯存餘裕不足會繼續試。** Shape mismatch、NaN、worker被kill、
逾時等其他錯誤會停止，不會被當成OOM吞掉。正式訓練開始後不做自動降batch／重跑。
原本的訓練如果已在跑，不要同時啟動此指令；先讓它保存checkpoint並停止。

探測紀錄在 `outputs/batch_probes/official_<時間>_<唯一ID>/`：
`batch_probe.json` 記錄每次結果、峰值顯存、選定batch和正式訓練command；
`batch_<大小>/worker.log` 保留錯誤原文。探測不存模型權重、不動原訓練輸出。

已存在這次baseline的權重時，明確續訓：

```bash
bash scripts/train_official_div2k_4090.sh --auto-batch --resume --max-steps 100000
```

這仍只是續訓指令範例，不自動增加訓練預算。`--auto-batch` 不能與手動
`OFFICIAL_BATCH_SIZE`／`OFFICIAL_GRAD_ACCUM` 或 `--batch-size`／`--grad-accum` 混用。

如果舊版本出現 `set_device ... specified index ... got:cuda`，這是探測程序的
裝置編號bug，不是顯存不足。更新後直接重跑 `--auto-batch`，不需重裝環境；
若正式訓練尚未開始，也不需 `--resume`。新版本遇到子程序錯誤會直接顯示原始
traceback，完整紀錄仍保留在對應 `worker.log`。

4090 完整 batch 的 VRAM／時間尚待 instance 實測；不要用 CPU smoke 推估。
進度列會顯示 step/s、ETA 與 peak allocated VRAM；驗證也有額外耗時。

## 圖片與紀錄在哪裡

```text
outputs/div2k_official_rgb_sat1.00x_50k/
  run_manifest.json                # 設定、版本、檔案清單、硬體、模型大小
  metrics.csv                      # training L1、驗證 PSNR/SSIM/Delta-E
  training_curves.png               # 每次存檔刷新
  full_gray_metrics.csv             # Direct/Algorithm2/gray色差、chroma、完整驗證L1
  previews/
    step_001000/                   # 這次隨機抽到的5張完整驗證圖片
      samples/                     # 5張對照圖，以原圖檔名命名
      trajectories/                # 同5張的左→右軌跡
      predictions/                 # 同5張的原始寬高Algorithm2輸出
      direct_predictions/          # 同5張的原始寬高Direct輸出
      preview.json                 # 抽樣種子、檔名、原始尺寸、完成狀態
    step_002000/                   # 重新抽5張；後續每1k同樣建立資料夾
  checkpoints/
    latest.pt                      # 最近一次；可續訓
    best.pt                        # 全灰階驗證Delta-E最低，未必50k
    step_050000.pt                  # 固定50k比較用
```

完整照片以 256px tile／32px overlap 推論；這會改變 attention 可見範圍，是明確的
full-resolution 適配，不宣稱等於整張一次進模型。非8倍數邊界只補齊再裁回，不 resize。
獨立 prediction PNG 保持原尺寸。對照圖／trajectory 的**顯示縮圖**限制最長邊512，
不改獨立預測；要看細節請開 `previews/step_001000/predictions/` 等資料夾。

預覽沿用驗證頻率 `training.validate_every: 1000`，最終停止的 step 也會保存一次。
`training.preview_count: 5`：每次從 validation 100 張中無放回抽5張；跨step可重複。
抽樣使用獨立的 seed+step 亂數，不消耗訓練的洗牌／crop 亂數。同一seed、step、
圖片清單可重現相同選圖；不足5張的小型 smoke set 則全部輸出。
舊 `data.validation_preview_name` 僅保留歷史設定相容性，official 模式不再固定選0803。
完整驗證仍是100張；不能把每次不同的5張預覽當成固定樣本的收斂比較。
逐張推論避免同時將5張大圖放上GPU，但預覽耗時與存圖空間會增加。
舊 `samples/`、`trajectories/` 不搬動、不刪除；新紀錄使用 `previews/`。
若中斷後重新走到同一步，已有預覽會保留，新的一次放在該step底下的 `retry_001/`
（依序編號），不會因存圖尚未完成就卡住續訓，也不會覆蓋前次證據。
更新程式不會改變已在執行的訓練程序，也不會替已完成的steps補出另外4張。

不要只看 PSNR。至少一起看場景的色彩、`delta_e76`、`direct_delta_e76`、
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
本次只有預覽的已知版本更新允許接續上一版 official checkpoint，並在 manifest
記錄舊指紋與原因；其他模型／訓練原始碼或資料差異仍拒絕。不會把舊模型放寬成可續訓。

## 訓練完成後：UIEB Test90，固定50k的兩種反推對照

```bash
bash scripts/evaluate_official_uieb_4090.sh
```

它使用 DIV2K 訓練的同一個 `step_050000.pt`，對 UIEB 水下 Test90 全灰階原尺寸
推論兩次。切分仍是 `splits/uieb_seed42.json`，與先前 FlowIE 的測試圖片一致。
輸入由 `data/UIEB/raw-890/` 轉為 RGB 三通道平均灰階；
`data/UIEB/reference-890/` **只用於評分及對照圖，不作模型輸入**。

- `paper_algorithm2`：論文公式，20次更新；主要 baseline。
- `official_code`：保留固定版本原始碼的時間索引，20次模型呼叫但19次有效更新。

兩者的 Direct 使用相同權重與全灰階輸入。**它們不是兩次訓練**，也不是
`gray_oneshot`。不因為都叫Algorithm2就把索引差異藏起來。
結果在 `evaluation/div2k_official_rgb_sat1.00x_uieb_test90_step050000/<sampler>/`：

```text
<sampler>/
  predictions/                  # 90張，反推最終輸出，原始寬高
  direct_predictions/           # 90張，同一權重的Direct，暫時保留
  batches/                      # 對照圖，包含GT；預設4張預覽
  trajectories/                 # 左→右軌跡；預設4張預覽
  其餘/
    metrics.json
    training_curves.png
    training_summary.json
```

曲線及訓練摘要需要 checkpoint 所屬訓練資料夾中的 `metrics.csv`；找不到時會警告，
不捏造曲線。可用 `--training-metrics` 指定檔案。`references/` 不另存，原始 GT
仍留在資料集。預設獨立 PNG 不縮放，對照圖／軌跡顯示縮圖最長邊 512；
需要 90 張對照圖及軌跡可在腳本後加 `--preview-count 90`。

需要舊14項指標時在指令後加 `--extended-metrics`，可能下載其評分模型。
第一次可先用預設，避免評分模型下載阻擋看圖；核心分數會先存下來。
額外產生的 `extended_metrics`、`direct_metrics`、`direct_vs_algorithm2` 的 CSV／MD
也放在 `其餘/`。額外評分直接重新載入相同前處理的 GT，再做與舊 PNG 匯出相同的
8-bit 量化，不需要重新輸出 references。

已完成的舊格式 UIEB 結果可整理，**不用重新推論**：

```bash
.venv/bin/python tools/organize_official_evaluation.py \
  --output-dir evaluation/div2k_official_rgb_sat1.00x_uieb_test90_step050000 \
  --apply
```

也可以指向單一 sampler 子資料夾。省略 `--apply` 只預覽。工具先檢查完整預測數量，
只搬已知報表、不覆蓋既有檔案；只移除與預測同名的 reference 副本，遇到不明檔案或
符號連結會停止。預測、Direct、batch、trajectory 檔案不變；reference 副本可由原始
UIEB 資料重建，資料集本身不會被修改。舊報表內容保留為當次推論紀錄，不重算分數。

新的 official 評測預設採這個精簡結構（`--output-layout auto`）；舊 Lab／小 U-Net
模式的預設輸出不變。明確指定 `--output-layout legacy` 可保留舊格式。

DIV2K Val100 域內對照仍可單獨執行 `bash scripts/evaluate_official_div2k_4090.sh`，
輸出在 `evaluation/div2k_official_rgb_sat1.00x_step050000/<sampler>/`，不要與 UIEB
的分數混為同一測試集。既有非空 official 輸出目錄不會覆蓋；需重推論時可設定
`OFFICIAL_EVAL_OUTPUT` 使用新資料夾。

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

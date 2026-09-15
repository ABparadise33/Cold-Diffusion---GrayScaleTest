# 官方 CIFAR-10：10k 接續至總共 100k

## 訓練

在原本 GPU instance 的外層專案執行，沿用原環境與 CIFAR-10 資料，不需重新 clone 或 setup：

```bash
cd /workspace/Cold-Diffusion---GrayScaleTest
git pull --ff-only
bash scripts/resume_official_cifar10.sh
```

預設 Python 是外層 `.venv/bin/python`，可用 `CIFAR_PYTHON` 指定既有環境。
官方程式位置為 `data/official_cold_diffusion/decolor-diffusion`，固定 upstream commit
`f8b1379151ff0cccba49112cf61d439bd4dd4ad9`。不會重新安裝依賴。

- 第一次從 `results/cifar10_rgb_fullgray_10k/model_10000.pt` 接續；總步數100,000，所以再跑90,000步。
- 舊10k只有模型、EMA與step，缺少Adam及亂數狀態。第一次接續會重新初始化Adam；這是延續權重的實驗，不能視為完整不中斷的100k對照。
- 新輸出獨立放在 `results/cifar10_rgb_fullgray_100k/`，保留舊10k資料夾。
- 每1k更新 `model.pt` 和官方訓練集預覽；每10k保留 `model_<step>.pt`。結束保留 `model_100000.pt`。
- 新存檔包含模型、EMA、Adam、step、Python/NumPy/PyTorch/CUDA亂數狀態。中斷後重跑同一指令，自動接續新資料夾的 `model.pt`。DataLoader重新建立，並非逐筆重播中途的shuffle/prefetch。
- 終端log：`logs/cifar10_rgb_fullgray_100k_resume_<時間>.log`；載入紀錄：新結果資料夾的 `resume_log.jsonl`；官方程式修改雜湊：`cifar_resume_patch.json`。

模型、損失、T20、Linear完整去色、sRGB、官方batch32×累積2、學習率2e-5和EMA設定延續原方案。存檔步數改為已完成的optimizer更新數；EMA仍沿用官方更新時機。

每1k預覽另外寫入同一結果資料夾：

- `preview_color_metrics_v2.csv`：每個step各有xt/direct_recons/recon三列，方便畫曲線。
- `preview_color_metrics.jsonl`：每張圖片的指標、原圖tensor SHA256、時間及預覽編號，對應 `sample-*-<編號>.png`。
- 終端log中的 `PREVIEW_COLOR`：三種輸出的平均ΔE76摘要。

RGB新增 `rgb_mae`（平均絕對誤差）、`rgb_mse`、`rgb_rmse`，使用sRGB [0,1]通道值，越低越好；這些是通道誤差，不是感知色差。V2 CSV獨立建立，保留舊CSV；JSONL以schema_version區分。

相對於同一批og計算：`delta_e76`是Lab D65整體色差（包含亮度），`ab_error`是只比較a/b色彩的距離，兩者越低越好；`chroma`與`target_chroma`是輸出/目標彩度，接近目標只代表彩度接近，不代表色相正確。`delta_e76_gain_vs_gray`是灰階色差減去輸出色差，正值代表比不補色改善。另記錄超出RGB範圍的通道比例 `clipped_fraction`；色差使用與顯示一致的裁切範圍，在PNG量化前計算。

這些是當次訓練預覽的批次平均及逐圖數值，不是固定驗證集；各step圖片可能不同。使用同一步的gray基準比較，跨step需觀察多次趨勢並以固定測試集確認。只增加記錄，不額外取樣或更動loss/模型。既有圖片不自動回填指標。

更新程式需在目前訓練程序停止後，於外層 `git pull --ff-only` 再執行相同續訓指令；已執行中的Python不會即時套用更新，不要同時啟動第二份訓練。新記錄從下一個1k預覽開始。

## Loss與固定圖片監測

新版本在續訓開始前建立當前checkpoint的固定監測基準，此後每1k更新一次。原本隨機訓練預覽保留。

- `train_loss_steps.csv` / `.jsonl`：每個已完成optimizer step的平均loss（兩個累積microbatch的原始loss平均，沒有重複除以2），JSONL另含兩個原始值。記錄step、run_id、lr。
- `train_loss_windows.csv`：每1k更新的mean/std/min/max與實際樣本數；中途起跑不足1k會如實記錄count/start_step。未完成的區間跟checkpoint一起保存，續訓接回；rollback後run_id可區分重跑紀錄。
- `fixed_monitor/manifest.json`：CIFAR10 test每類100張，共1,000張，固定seed42選樣、索引排序、無隨機augmentation；保留索引與原圖hash，資料變更會拒絕沿用同一監測目錄。
- `fixed_monitor/summary.csv` / `.jsonl`：同一批1,000張的灰階、EMA Direct、EMA recon，RGB MAE/MSE/逐圖RMSE平均、Lab CIE76、ab誤差、彩度、裁切比例與相對灰階改善量。
- `fixed_monitor/preview_color_metrics.jsonl` / `preview_color_metrics_v2.csv`：逐圖／逐批資料；`step_<step>/{og,xt,direct_recons,recon}.png`：固定前32張對照。

這1,000張test資料已用於監測與續訓決策，因此扮演validation角色，不是未接觸過的最終test。初始化與監測隔離Python/NumPy/PyTorch/CUDA RNG、使用EMA eval/no_grad並還原原有module模式；不消耗訓練DataLoader、不修改loss或訓練資料。第一次需要原CIFAR10 test資料仍在train dataset root。

### 下一段預算／停訓建議（尚未自動啟用）

- 建議先考慮總共120k（從100k再20k），不要直接延長到700k。理由是50k→100k完整test CIE76僅改善0.0303。
- 以固定監測集recon CIE76為主，RGB MAE與ab誤差為交叉檢查；彩度僅作診斷。將最近3次1k監測的CIE76平均作為觀察曲線，保留原始值。
- 初步實用門檻可用CIE76下降0.05；連續10k沒有達到相對先前有效最佳值的改善，就先暫停檢查。這是計算預算的暫定門檻，不是經統計推導的普遍收斂標準；應根據監測集的逐圖配對差異與短期波動重新確認。未達0.05的小幅單次改善不重設觀察期間。
- 若loss持續下降但RGB/Lab色差持平／惡化，不把loss下降當作續訓充分理由；若固定集有改善，在120k做較大範圍的評估後再決定下一段。
- 目前只記錄，**不自動早停**，預設訓練上限仍是100k。上述條件需人工檢查，並非已實作的自動停止器。

只有決定續訓後，才執行以下指令（會開始訓練）：

```bash
cd /workspace/Cold-Diffusion---GrayScaleTest
git pull --ff-only
CIFAR_TRAIN_STEPS=120000 bash scripts/resume_official_cifar10.sh
```

新120k實驗從100k的 `model_100000.pt` 恢復Adam/EMA/step/RNG，寫入獨立 `results/cifar10_rgb_fullgray_120k/`，保留原100k結果；中斷後同一指令優先讀120k目錄的 `model.pt`。DataLoader順序仍非逐筆重播。腳本要求上限為1k倍數；不會自行從100k提升到120k。

## Evaluate

100k已完成後不用再訓練，於外層專案執行：

```bash
cd /workspace/Cold-Diffusion---GrayScaleTest
git pull --ff-only
bash scripts/evaluate_official_cifar10.sh
```

預設使用官方 `results/cifar10_rgb_fullgray_100k/model_100000.pt` 的EMA，固定seed42、CIFAR-10 **test split全部10,000張**、按資料索引排序、batch32、32px、sRGB、Linear完整去色、T20及官方x0_step_down sampler。沿用原有環境與已下載的CIFAR資料，不需重建環境。末批16張也會評分。

輸出在官方程式目錄 `data/official_cold_diffusion/decolor-diffusion/evaluation/cifar10_test_step100000/`：

- `summary.json`：全測試集xt/direct_recons/recon的RGB與Lab平均誤差、相對灰階改善量；正改善量代表優於灰階。
- `preview_color_metrics_v2.csv` / `preview_color_metrics.jsonl`：逐批／逐圖指標、test split標記、資料索引、目標hash。RGB RMSE摘要是逐圖RMSE的平均。
- `og.png` / `xt.png` / `direct_recons.png` / `recon.png`：同一批固定前32張的四種預覽；分數使用全部10,000張。
- `evaluation_config.json`：checkpoint hash、step、設定、程式hash、環境；`sampler.log`保存官方取樣訊息。外層 `logs/cifar10_evaluate_<時間>.log`保存進度及摘要。

新腳本直接呼叫官方模型與sampler，不依賴官方 `test.py` 的手動CIFAR修改。該版本 `test_from_data()` 在第一批存圖後就return，舊test命令只能用於預覽，不能宣稱完整test評分。

比較10k時指定舊checkpoint（仍用相同test資料和排序）：

```bash
bash scripts/evaluate_official_cifar10.sh \
  --checkpoint data/official_cold_diffusion/decolor-diffusion/results/cifar10_rgb_fullgray_10k/model_10000.pt
```

50k改指向 `results/cifar10_rgb_fullgray_100k/model_50000.pt`。每個step有獨立輸出目錄；重跑同一步請加 `--output-dir <新目錄>`，避免覆寫已完成結果。可用 `--limit 32 --output-dir <新目錄>`檢查流程，但只會標記為test_subset，不能代表完整測試結果。

訓練每1k的新RGB紀錄需要下次啟動訓練時才會生效，已結束的100k訓練不會補寫歷史數值；本次完整評測可立即計算100k的RGB/Lab誤差。

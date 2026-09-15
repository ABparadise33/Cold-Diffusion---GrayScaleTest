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

## Evaluate

沿用先前已修正且成功執行的官方 `test.py`（CIFAR使用torchvision test split），以EMA評估100k：

```bash
cd /workspace/Cold-Diffusion---GrayScaleTest/data/official_cold_diffusion/decolor-diffusion
CIFAR_PYTHON=/workspace/Cold-Diffusion---GrayScaleTest/.venv/bin/python
mkdir -p logs
set -o pipefail
CIFAR_INFERENCE_ONLY=1 "$CIFAR_PYTHON" -u test.py \
  --dataset cifar10 --dataset_folder ./data --model UnetConvNext \
  --forward_process_type Decolorization --decolor_routine Linear \
  --decolor_total_remove --time_steps 20 --sample_steps 20 \
  --train_routine Final --sampling_routine x0_step_down \
  --save_folder_train ./results --save_folder_test ./evaluation \
  --exp_name cifar10_rgb_fullgray_100k --load_model_steps 100000 \
  --test_type test_data --order_seed 42 --test_postfix step100000 \
  2>&1 | tee logs/cifar10_eval_100k.log
```

`CIFAR_INFERENCE_ONLY=1`只載入模型與EMA，避免以測試batch等設定檢查訓練續接狀態。50k比較改用 `--load_model_steps 50000`，並將postfix/log改為50k。使用相同測試集、順序seed及取樣設定對比10k/50k/100k；訓練預覽本身不是驗證集結果。

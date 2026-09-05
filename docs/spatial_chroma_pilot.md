# Spatial chroma-mask pilot

## 要驗證的問題

自然影像模型能否補出**真正缺失**的局部色彩，而不是把每個像素裡殘留的微弱水下偏色放大回去？
這是新退化算子，不是原論文的RGB色差幅度縮放。

## Forward operator

對每個訓練crop、每個像素先抽一個連續score `u`。同一條trajectory固定使用同一張score map：

```text
q_t = (T - t) / T
M_t(p) = 1[u(p) < q_t]
D(x,t) = M_t * x + (1-M_t) * mean_RGB(x)
```

- `t=0`：q=100%，所有像素保留完整RGB。
- `t=10`：q=50%，約一半像素完整RGB，其餘三通道相同。
- `t=19`：q=5%，約5%像素完整RGB，其餘真正灰階。
- `t=20`：q=0%，全灰階。

這裡的百分比是隨機遮罩的期望比例；128×128 crop的實際數量會有小幅波動。
遮罩巢狀，所以forward只會持續刪色，不會讓已灰階像素隨機變回彩色。

## Training / reverse

每張DIV2K crop仍是target `x0`，每次均勻抽`t=1..20`與新的隨機score map，模型學：

```text
R(D(x0,t), t) -> x0
```

Algorithm2反推時固定同一張score map。從`t`到`t-1`時，只對該步新揭露的像素加入
模型預測的RGB chroma；oracle測試可從全灰階精確回到已知x0。真正模型是否能合理推測
未知顏色，必須由validation/UIEB實驗回答。

## 固定條件

- DIV2K Train800 / Val100、saturation1、128 crop、seed42。
- upstream ConvNeXt 56.6M、T20、paper Algorithm2、Adam 2e-5、FP32。
- effective batch32；每1k驗證並抽5張完整validation影像。
- 新output與舊checkpoint隔離；latest/best/final checkpoint可續訓。
- 不執行或輸出Direct；尚未改成saturation2或更多timesteps。

## 執行

```bash
bash scripts/train_spatial_chroma_div2k_4090.sh --auto-batch
```

預設10k。先看5k/10k的validation loss、Delta-E76與完整圖片。若10k仍只有大片灰色、
顏色呈隨機斑點或指標沒有改善，先停，不直接燒到50k。若有一致改善：

```bash
bash scripts/train_spatial_chroma_div2k_4090.sh --auto-batch --resume --max-steps 50000
```

這個pilot只處理colorization，不會學會UIEB GT的去霧、亮度或白平衡映射。

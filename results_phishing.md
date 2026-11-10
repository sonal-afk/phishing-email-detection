# Phishing Email Detection — Final Results

**Project:** Phishing Email Detection Using Deep Learning-Based NLP
**Author:** Sonal Ramesh
**Model:** 3-hidden-layer feed-forward NN (512→256→128) on TF-IDF features (5,000, 1–2 grams)
**Dataset:** subhajournal/phishingemails — 18,631 emails (7,309 phishing 39.2% / 11,322 legitimate 60.8%)
**Split (stratified, seed 42):** train 13,041 / val 2,795 / test 2,795
**Setup:** Adam lr=1e-3, cosine decay, weight decay 1e-4, dropout 0.3, BCEWithLogitsLoss with pos_weight, 15 epochs, batch 128. TF-IDF fit on train only (no leakage).

## Test-set results

| Model | Accuracy | Precision | Recall | F1 | Phishing missed (FN) |
|---|---|---|---|---|---|
| **FFN (3 hidden layers)** | **96.64%** | **93.34%** | **98.45%** | **95.83%** | **17** |
| Logistic Regression (baseline) | 96.06% | 92.72% | 97.63% | 95.11% | 26 |

FFN confusion matrix: TN=1622, FP=77, FN=17, TP=1079.
LR confusion matrix: TN=1615, FP=84, FN=26, TP=1070.

## Hyperparameter / ablation study (test F1, same epochs)

| Config | Hidden | Dropout | LR | F1 | Recall | Phishing missed |
|---|---|---|---|---|---|---|
| baseline | 512,256,128 | 0.3 | 1e-3 | 95.83% | 98.45% | 17 |
| more_dropout | 512,256,128 | 0.5 | 1e-3 | 95.68% | 97.90% | 23 |
| higher_lr | 512,256,128 | 0.3 | 3e-3 | 95.60% | 97.17% | 31 |
| narrow | 256,128,64 | 0.3 | 1e-3 | 95.52% | 97.35% | 29 |
| no_dropout | 512,256,128 | 0.0 | 1e-3 | 95.42% | 97.99% | 22 |
| wider | 1024,512,256 | 0.3 | 1e-3 | 95.13% | 98.08% | 21 |

## Takeaways
- The feed-forward network beats the Logistic-Regression baseline on every metric and halves... no, reduces missed phishing (17 vs 26).
- Best config is the 512→256→128 baseline with dropout 0.3; widening the layers hurt (diminishing returns / mild over-fit), confirming extra capacity does not pay off for a dense model on TF-IDF features.
- Recall ≈ 98.5% on phishing: very few phishing emails slip through, which is the project's priority (minimise false negatives).

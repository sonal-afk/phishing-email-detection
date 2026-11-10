# Model Comparison — Phishing Email Detection (test set)

Same dataset (18,631 emails) and same stratified 70/15/15 split (seed 42) for every model.
Metrics on the held-out test set (2,795 emails); phishing is the positive class.

| Model | Accuracy | Precision | Recall | F1 | Phishing missed (FN) | Approx. cost |
|---|---|---|---|---|---|---|
| **BERT** (bert-base-uncased, fine-tuned) | 97.53% | 94.69% | 99.27% | **96.93%** | **8** | ~350 s/epoch, ~110M params, 440 MB |
| **FFN (TF-IDF)** — proposed | 96.67% | 93.65% | 98.18% | 95.86% | 20 | <1 s/epoch, ~2.73M params |
| CNN (TextCNN) | 96.31% | 94.45% | 96.26% | 95.35% | 41 | ~1 s/epoch |
| RNN (BiLSTM) | 95.81% | 92.38% | 97.35% | 94.80% | 29 | ~2 s/epoch |

(Earlier dedicated FFN run reported 96.64% / 95.83% F1 / 17 FN; small differences vs the joint run above are GPU non-determinism, i.e. within run-to-run noise.)

## Takeaways
- **BERT is best** (96.93% F1, only 8 phishing missed), as expected from the literature — but it is by far the heaviest model (~110M parameters, ~350 s/epoch vs <1 s for the FFN).
- **The proposed FFN is second** and beats both the CNN and the RNN, at a tiny fraction of BERT's cost — the best accuracy-per-compute.
- CNN and RNN do not improve on the FFN here: phishing emails carry strong lexical cues (keywords, URL/currency tokens) that TF-IDF already captures, so modelling word order adds little on this dataset.
- All models keep recall high (≥96%), i.e. few phishing emails slip through; BERT and the FFN give the fewest false negatives.

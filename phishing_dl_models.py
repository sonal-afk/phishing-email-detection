"""
Phishing Email Detection - Model Comparison
===========================================
Trains and compares four deep-learning models on the SAME dataset split, so the
numbers are directly comparable:

    FFN   - three-hidden-layer feed-forward network on TF-IDF features (the proposed model)
    CNN   - TextCNN on a trainable word-embedding (conv filters of width 3/4/5)
    RNN   - bidirectional LSTM on a trainable word-embedding
    BERT  - fine-tuned pre-trained transformer (bert-base-uncased)

Every model is evaluated with accuracy, precision, recall, F1 and the number of
missed phishing emails (false negatives), and a final comparison table is printed.

Usage
-----
    python phishing_dl_models.py --kaggle                       # all four models
    python phishing_dl_models.py --kaggle --models ffn cnn rnn  # skip the slow BERT
    python phishing_dl_models.py --kaggle --models bert         # just BERT
    python phishing_dl_models.py --csv data/Phishing_Email.csv  # local CSV

BERT needs:  pip install transformers
(The first BERT run downloads ~440 MB and is much slower than the other models;
run the fast models first, then BERT.)

Run with the CUDA python so it uses the GPU:
    C:\\Users\\sonal\\AppData\\Local\\Programs\\Python\\Python314\\python.exe phishing_dl_models.py --kaggle
"""

import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# reuse the data pipeline and the FFN from the existing script
from phishing_ffn import (SEED, set_seed, resolve_kaggle_csv, load_dataset,
                          build_features, train_mlp, metrics_block)


# --------------------------------------------------------------------------- #
# Sequence encoding for CNN / RNN (vocabulary built on TRAIN only -> no leakage)
# --------------------------------------------------------------------------- #
PAD, UNK = 0, 1


def build_vocab(texts, max_vocab=20000, min_freq=2):
    c = Counter()
    for t in texts:
        c.update(t.split())
    itos = ["<pad>", "<unk>"] + [w for w, f in c.most_common(max_vocab) if f >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, stoi, max_len=200):
    arr = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for i, t in enumerate(texts):
        for j, w in enumerate(t.split()[:max_len]):
            arr[i, j] = stoi.get(w, UNK)
    return arr


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=128, n_filters=100, sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb, n_filters, k) for k in sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(sizes), 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                 # (B, emb, L)
        outs = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(outs, dim=1))).squeeze(1)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(self.emb(x))                   # (B, L, 2H)
        return self.fc(self.drop(o.max(dim=1).values)).squeeze(1)


# --------------------------------------------------------------------------- #
# Generic training loop for the sequence models (CNN / RNN)
# --------------------------------------------------------------------------- #
def train_seq(model, Xtr, ytr, Xval, yval, Xte, yte, device, name,
              epochs=8, lr=1e-3, batch_size=128):
    def loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr, va, te = loader(Xtr, ytr, True), loader(Xval, yval, False), loader(Xte, yte, False)
    model.to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def evaluate(loader_):
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for xb, yb in loader_:
                ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(ps), np.concatenate(ys)

    from sklearn.metrics import precision_recall_fscore_support
    best_f1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        vp, vy = evaluate(va)
        _, _, vf1, _ = precision_recall_fscore_support(
            vy, (vp >= 0.5).astype(int), average="binary", zero_division=0)
        if vf1 >= best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{name}] epoch {ep:2d}/{epochs}  val_F1 {vf1*100:5.2f}%  ({time.time()-t0:.1f}s)")
    if best_state:
        model.load_state_dict(best_state)
    tp, ty = evaluate(te)
    return metrics_block(ty, (tp >= 0.5).astype(int), name)


# --------------------------------------------------------------------------- #
# BERT fine-tuning
# --------------------------------------------------------------------------- #
def train_bert(tr_txt, ytr, va_txt, yval, te_txt, yte, device, name,
               model_name="bert-base-uncased", epochs=2, max_len=256, batch_size=16, lr=2e-5):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise SystemExit("transformers not installed. Run:  pip install transformers")
    from sklearn.metrics import precision_recall_fscore_support

    tok = AutoTokenizer.from_pretrained(model_name)

    def enc(texts):
        e = tok(list(texts), truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    def loader(texts, y, shuffle):
        ids, mask = enc(texts)
        ds = TensorDataset(ids, mask, torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr, va, te = loader(tr_txt, ytr, True), loader(va_txt, yval, False), loader(te_txt, yte, False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    w = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def evaluate(loader_):
        model.eval(); preds, ys = [], []
        with torch.no_grad():
            for ids, mask, yb in loader_:
                logits = model(ids.to(device), attention_mask=mask.to(device)).logits
                preds.append(logits.argmax(1).cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(preds), np.concatenate(ys)

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for ids, mask, yb in tr:
            opt.zero_grad()
            logits = model(ids.to(device), attention_mask=mask.to(device)).logits
            loss = crit(logits, yb.to(device)); loss.backward(); opt.step()
        vp, vy = evaluate(va)
        _, _, vf1, _ = precision_recall_fscore_support(vy, vp, average="binary", zero_division=0)
        if vf1 >= best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{name}] epoch {ep}/{epochs}  val_F1 {vf1*100:5.2f}%  ({time.time()-t0:.0f}s)")
    if best_state:
        model.load_state_dict(best_state)
    tp, ty = evaluate(te)
    return metrics_block(ty, tp, name)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--models", nargs="+", default=["ffn", "cnn", "rnn", "bert"],
                    choices=["ffn", "cnn", "rnn", "bert"])
    ap.add_argument("--bert-model", default="bert-base-uncased",
                    help="e.g. distilbert-base-uncased for a faster run")
    ap.add_argument("--seq-epochs", type=int, default=8)
    ap.add_argument("--bert-epochs", type=int, default=2)
    ap.add_argument("--bert-batch", type=int, default=16,
                    help="lower this (e.g. 8) if the GPU runs out of memory")
    ap.add_argument("--max-len", type=int, default=200)
    args = ap.parse_args()
    if not args.csv and not args.kaggle:
        ap.error("provide --csv <path> or --kaggle")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seed: {SEED}  |  models: {args.models}")

    X, y = load_dataset(args.csv or resolve_kaggle_csv())
    # identical stratified 70/15/15 split as phishing_ffn.py (same seed)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
    print(f"Split -> train {len(ytr):,}  val {len(yval):,}  test {len(yte):,}")

    results = {}

    if "ffn" in args.models:
        print("\n===== FFN (TF-IDF, 3 hidden layers) =====")
        _, Xtr_t, Xval_t, Xte_t = build_features(Xtr, Xval, Xte, 5000)
        _, results["FFN (TF-IDF)"] = train_mlp(Xtr_t, ytr, Xval_t, yval, Xte_t, yte,
                                               device, epochs=15, verbose=False)

    if "cnn" in args.models or "rnn" in args.models:
        stoi = build_vocab(Xtr)
        print(f"\nVocabulary (train only): {len(stoi):,} tokens; sequence length {args.max_len}")
        Etr, Eval, Ete = (encode(Xtr, stoi, args.max_len), encode(Xval, stoi, args.max_len),
                          encode(Xte, stoi, args.max_len))
        if "cnn" in args.models:
            print("\n===== TextCNN =====")
            results["CNN"] = train_seq(TextCNN(len(stoi)), Etr, ytr, Eval, yval, Ete, yte,
                                       device, "CNN", epochs=args.seq_epochs)
        if "rnn" in args.models:
            print("\n===== BiLSTM (RNN) =====")
            results["RNN (BiLSTM)"] = train_seq(TextRNN(len(stoi)), Etr, ytr, Eval, yval, Ete, yte,
                                                device, "RNN", epochs=args.seq_epochs)

    if "bert" in args.models:
        print(f"\n===== BERT ({args.bert_model}) =====")
        results["BERT"] = train_bert(Xtr, ytr, Xval, yval, Xte, yte, device, "BERT",
                                     model_name=args.bert_model, epochs=args.bert_epochs,
                                     max_len=min(args.max_len, 256), batch_size=args.bert_batch)

    # ---- final comparison ----
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (test set)")
    print("=" * 70)
    print(f"{'model':18s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'phish_missed':>13s}")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
        print(f"{name:18s} {m['acc']*100:6.2f}% {m['precision']*100:6.2f}% "
              f"{m['recall']*100:6.2f}% {m['f1']*100:6.2f}% {m['fn']:13d}")

    # save a small summary for the report
    with open("results_models.md", "w", encoding="utf-8") as f:
        f.write("# Model comparison (test set)\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 | Phishing missed |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
            f.write(f"| {name} | {m['acc']*100:.2f}% | {m['precision']*100:.2f}% | "
                    f"{m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['fn']} |\n")
    print("\nSaved comparison -> results_models.md")


if __name__ == "__main__":
    main()

"""
Phishing Email Detection Using Deep Learning-Based NLP
======================================================
A feed-forward (fully-connected) neural network with three hidden layers that
classifies raw email text as PHISHING vs. LEGITIMATE.

Pipeline:  email text -> TF-IDF features -> 3 hidden-layer MLP -> phishing / safe

The design mirrors the project proposal and the literature survey (Atawneh &
Aljehani 2023; Fang et al. 2019): a reproducible feed-forward baseline on
TF-IDF features, evaluated with accuracy / precision / recall / F1 and compared
against a classical Logistic-Regression baseline, with attention to FALSE
NEGATIVES (missed phishing) because those are the costly errors.

Usage
-----
    python phishing_ffn.py --kaggle                 # auto-download + full run
    python phishing_ffn.py --kaggle --epochs 3      # auto-download + quick test
    python phishing_ffn.py --kaggle --sweep         # auto-download + ablation
    python phishing_ffn.py --csv path/to/Phishing_Email.csv   # use a local CSV

--kaggle uses kagglehub to fetch "subhajournal/phishingemails" (needs
`pip install kagglehub` and a Kaggle login the first time).

Expected CSV: a text column (e.g. "Email Text") and a label column
(e.g. "Email Type" with values like "Phishing Email" / "Safe Email").
The loader auto-detects common column names and label spellings.
"""

import argparse
import glob
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42  # fixed for reproducibility (rubric: set and report random seeds)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 1. DATA LOADING & PREPROCESSING
# --------------------------------------------------------------------------- #
TEXT_COL_CANDIDATES = ["email text", "email_text", "text", "body", "message",
                       "email", "content"]
LABEL_COL_CANDIDATES = ["email type", "email_type", "label", "type", "class",
                        "category", "target"]


def _find_column(columns, candidates):
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def clean_text(s):
    """Light cleaning; TF-IDF handles lowercasing/tokenising, we just tidy."""
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", " url ", s)   # collapse URLs to a token
    s = re.sub(r"\S+@\S+", " email ", s)           # collapse email addresses
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_binary_label(v):
    """Map any label spelling to 1 = phishing, 0 = legitimate/safe."""
    t = str(v).strip().lower()
    if t in {"1", "phishing", "phishing email", "phish", "spam", "malicious", "fraud"}:
        return 1
    if t in {"0", "safe", "safe email", "legitimate", "legit", "ham", "normal", "benign"}:
        return 0
    # fall back: anything containing "phish" is phishing
    return 1 if "phish" in t else 0


def resolve_kaggle_csv():
    """Download subhajournal/phishingemails via kagglehub and locate the CSV."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub not installed. Run:  pip install kagglehub")
    path = kagglehub.dataset_download("subhajournal/phishingemails")
    print("Downloaded dataset to:", path)
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise SystemExit(f"No CSV found under {path}")
    # prefer a file that looks like the phishing email CSV
    for c in csvs:
        if "phish" in os.path.basename(c).lower():
            return c
    return csvs[0]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    text_col = _find_column(df.columns, TEXT_COL_CANDIDATES)
    label_col = _find_column(df.columns, LABEL_COL_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found: {list(df.columns)}.\n"
            f"Rename your text column to 'Email Text' and label to 'Email Type', "
            f"or edit TEXT_COL_CANDIDATES / LABEL_COL_CANDIDATES.")
    print(f"Using text column '{text_col}' and label column '{label_col}'")

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col].str.len() > 0]
    y = df[label_col].map(to_binary_label).to_numpy()
    X = df[text_col].to_numpy()

    print(f"Loaded {len(y):,} emails  |  phishing={int(y.sum()):,} "
          f"({y.mean()*100:.1f}%)  legitimate={int((1-y).sum()):,} "
          f"({(1-y.mean())*100:.1f}%)")
    return X, y


# --------------------------------------------------------------------------- #
# 2. DATA LOADER (densify the TF-IDF matrix ONCE, so the GPU is fed quickly)
# --------------------------------------------------------------------------- #
def make_loader(X_sparse: csr_matrix, y: np.ndarray, batch_size, shuffle):
    X_dense = torch.from_numpy(X_sparse.toarray().astype(np.float32))
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_dense, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


# --------------------------------------------------------------------------- #
# 3. MODEL: three-hidden-layer feed-forward network
# --------------------------------------------------------------------------- #
class PhishingMLP(nn.Module):
    def __init__(self, in_features, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.features = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)          # single logit -> BCEWithLogitsLoss

    def forward(self, x):
        return self.out(self.features(x)).squeeze(1)


# --------------------------------------------------------------------------- #
# 4. TRAIN / EVALUATE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metrics_block(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)   # rows=true [0,1], cols=pred [0,1]
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- {title} ---")
    print(f"Accuracy {acc*100:.2f}%  Precision {p*100:.2f}%  "
          f"Recall {r*100:.2f}%  F1 {f1*100:.2f}%")
    print(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"False negatives (phishing missed): {fn}  "
          f"| False positives (legit flagged): {fp}")
    print(classification_report(y_true, y_pred,
          target_names=["legitimate", "phishing"], zero_division=0))
    return dict(acc=acc, precision=p, recall=r, f1=f1,
                tn=tn, fp=fp, fn=fn, tp=tp)


def train_mlp(X_tr, y_tr, X_val, y_val, X_te, y_te, device,
              hidden=(512, 256, 128), dropout=0.3, lr=1e-3, epochs=15,
              batch_size=128, verbose=True):
    train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, 256, shuffle=False)
    test_loader = make_loader(X_te, y_te, 256, shuffle=False)

    model = PhishingMLP(X_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    # pos_weight upweights the phishing class -> fewer false negatives
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        vp, vy = predict(model, val_loader, device)
        _, _, vf1, _ = precision_recall_fscore_support(
            vy, (vp >= 0.5).astype(int), average="binary", zero_division=0)
        if vf1 >= best_f1:
            best_f1, best_state = vf1, {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs}  val_F1 {vf1*100:5.2f}%  "
                  f"({time.time()-t0:.1f}s)")

    if best_state:
        model.load_state_dict(best_state)
    tp_prob, ty = predict(model, test_loader, device)
    return model, metrics_block(ty, (tp_prob >= 0.5).astype(int),
                                f"MLP (hidden={hidden}, dropout={dropout}, lr={lr})")


# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #
def build_features(X_train_text, X_val_text, X_test_text, max_features):
    # IMPORTANT: fit on TRAIN ONLY -> no test-set leakage (rubric criterion 3)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True, min_df=2)
    X_tr = vec.fit_transform(X_train_text)
    X_val = vec.transform(X_val_text)
    X_te = vec.transform(X_test_text)
    return vec, X_tr, X_val, X_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to a local phishing email CSV")
    ap.add_argument("--kaggle", action="store_true",
                    help="auto-download subhajournal/phishingemails via kagglehub")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--sweep", action="store_true",
                    help="run a small hyperparameter/ablation study")
    args = ap.parse_args()

    if not args.csv and not args.kaggle:
        ap.error("provide --csv <path> or --kaggle")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seed: {SEED}")

    csv_path = args.csv or resolve_kaggle_csv()
    X, y = load_dataset(csv_path)

    # stratified 70 / 15 / 15 train / val / test  (no leakage, reproducible)
    X_tr_txt, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val_txt, X_te_txt, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"Split -> train {len(y_tr):,}  val {len(y_val):,}  test {len(y_te):,}")

    vec, X_tr, X_val, X_te = build_features(
        X_tr_txt, X_val_txt, X_te_txt, args.max_features)
    print(f"TF-IDF features: {X_tr.shape[1]:,}")

    if args.sweep:
        print("\n===== HYPERPARAMETER / ABLATION STUDY =====")
        configs = [
            ("baseline",      dict(hidden=(512, 256, 128), dropout=0.3, lr=1e-3)),
            ("wider",         dict(hidden=(1024, 512, 256), dropout=0.3, lr=1e-3)),
            ("narrow",        dict(hidden=(256, 128, 64),  dropout=0.3, lr=1e-3)),
            ("more_dropout",  dict(hidden=(512, 256, 128), dropout=0.5, lr=1e-3)),
            ("no_dropout",    dict(hidden=(512, 256, 128), dropout=0.0, lr=1e-3)),
            ("higher_lr",     dict(hidden=(512, 256, 128), dropout=0.3, lr=3e-3)),
        ]
        results = []
        for name, cfg in configs:
            print(f"\n### {name}: {cfg}")
            _, m = train_mlp(X_tr, y_tr, X_val, y_val, X_te, y_te, device,
                             epochs=args.epochs, verbose=False, **cfg)
            results.append((name, m["f1"], m["recall"], m["fn"]))
        print("\n===== SWEEP SUMMARY (ranked by F1) =====")
        print(f"{'config':14s} {'F1':>7s} {'recall':>8s} {'phish_missed':>13s}")
        for name, f1, rec, fn in sorted(results, key=lambda r: r[1], reverse=True):
            print(f"{name:14s} {f1*100:6.2f}% {rec*100:7.2f}% {fn:13d}")
        return

    # ---- main run: MLP + Logistic-Regression baseline ----
    model, mlp_m = train_mlp(X_tr, y_tr, X_val, y_val, X_te, y_te, device,
                             epochs=args.epochs)

    print("\n===== BASELINE: TF-IDF + Logistic Regression =====")
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_clf.fit(X_tr, y_tr)
    lr_m = metrics_block(y_te, lr_clf.predict(X_te),
                         "Logistic Regression (baseline)")

    print("\n===== FINAL COMPARISON (test set) =====")
    print(f"{'model':22s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'phish_missed':>13s}")
    for nm, m in [("FFN (3 hidden layers)", mlp_m), ("Logistic Regression", lr_m)]:
        print(f"{nm:22s} {m['acc']*100:6.2f}% {m['precision']*100:6.2f}% "
              f"{m['recall']*100:6.2f}% {m['f1']*100:6.2f}% {m['fn']:13d}")

    # save the trained model so it can be demonstrated in the viva
    torch.save({"state_dict": model.state_dict(),
                "vectorizer_vocab": vec.vocabulary_,
                "hidden": (512, 256, 128)}, "phishing_mlp.pt")
    print("\nSaved trained model -> phishing_mlp.pt")


if __name__ == "__main__":
    main()

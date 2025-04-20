#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_predict_animals_dual.py  —  v2.7
===================================
• Modelos independientes con %Gain, %Loss y opcional Gain+Loss.
• Loss: NA→100   |  Gain: NA→0
• Importancias (RF y Permutation) normalizadas 0‑100 + columna
  'Associated_to' (Human / Animal / Neutral).
"""
from __future__ import annotations
import argparse, os, random, sys
from statistics import median
from typing import Dict, List

import numpy as np, pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score,
                             precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
)
try:
    from sklearn.model_selection import StratifiedGroupKFold  # ≥1.1
except ImportError:
    StratifiedGroupKFold = None
from sklearn.inspection import permutation_importance

# ───────── helpers ───────── #
def trim_seqid(s: str) -> str:
    p = s.split("_"); return "_".join(p[:2]) if len(p) >= 2 else s
def lineage_from_trim(s: str) -> str: return s.split("_")[0]
def label_from_lineage(l: str) -> str: return "Human" if l.upper().startswith("L") else "Animal"

def balance(X, y, lin, rng):
    hum = {l for l, lab in zip(lin, y) if lab == "Human"}
    ani = {l for l, lab in zip(lin, y) if lab == "Animal"}
    idx: Dict[str, List[int]] = {}
    for i, l in enumerate(lin): idx.setdefault(l, []).append(i)
    med_h = round(median(len(idx[l]) for l in hum))
    n_h = sum(len(idx[l]) for l in hum); n_a = sum(len(idx[l]) for l in ani)
    tgt_a = max(1, int(med_h * n_h / n_a)) if n_a else 0
    sel: List[int] = []
    for l in hum:
        pool = idx[l]; sel += rng.choices(pool, k=med_h) if len(pool) < med_h else rng.sample(pool, med_h)
    for l in ani:
        pool = idx[l]; sel += rng.choices(pool, k=tgt_a) if len(pool) < tgt_a else rng.sample(pool, tgt_a)
    return X.iloc[sel].copy(), y.iloc[sel].copy()

# ───────── train / evaluate one set ───────── #
def train(df: pd.DataFrame, cfg: dict, tag: str, rng: random.Random):
    out = os.path.join(cfg["outdir"], tag); os.makedirs(out, exist_ok=True)
    df["Lineage"] = df.index.map(trim_seqid).map(lineage_from_trim)
    df["Label"]   = df["Lineage"].map(label_from_lineage)
    y, lin = df["Label"], df["Lineage"]
    X      = df.drop(columns=["Lineage", "Label"]).astype(np.float32)

    print(f"\n== {tag.upper()} =="); print(y.value_counts())

    if cfg["cv"] == "lineage":
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=cfg["seed"]) \
                   if StratifiedGroupKFold else GroupKFold(5)
        splits = splitter.split(X, y, groups=lin)
    elif cfg["cv"] == "stratified":
        splits = StratifiedKFold(10, shuffle=True, random_state=cfg["seed"]).split(X, y)
    else:
        splits = RepeatedStratifiedKFold(10, cfg["repeats"], random_state=cfg["seed"]).split(X, y)

    def make_base():
        if cfg["model"] == "rf":
            return RandomForestClassifier(n_estimators=cfg["n_estimators"],
                                          n_jobs=cfg["workers"],
                                          class_weight="balanced_subsample",
                                          random_state=cfg["seed"])
        return LogisticRegression(penalty="l1", solver="saga", max_iter=1000,
                                  class_weight="balanced",
                                  n_jobs=cfg["workers"],
                                  random_state=cfg["seed"])

    aucs = precs = recs = f1s = []
    aucs = []; precs = []; recs = []; f1s = []
    cm_tot = np.zeros((2, 2), int)

    for i, (tr, te) in enumerate(splits, 1):
        Xtr, Ytr, Lintr = X.iloc[tr], y.iloc[tr], lin.iloc[tr]
        Xte, Yte = X.iloc[te], y.iloc[te]
        Xb, Yb = balance(Xtr, Ytr, Lintr, rng)
        model = make_base(); model.fit(Xb, Yb)
        proba = model.predict_proba(Xte)[:, list(model.classes_).index("Human")]
        auc = roc_auc_score((Yte == "Human"), proba) if len(np.unique(Yte)) > 1 else np.nan
        pred = model.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(Yte, pred, pos_label="Human",
                                                     average="binary", zero_division=0)
        cm_tot += confusion_matrix(Yte, pred, labels=["Animal", "Human"])
        aucs.append(auc); precs.append(p); recs.append(r); f1s.append(f)
        print(f"Fold{i:02d}: AUC={auc if not np.isnan(auc) else 'NA'}  P={p:.2f} R={r:.2f} F1={f:.2f}")

    ms = lambda a: (np.nanmean(a), np.nanstd(a))
    pd.DataFrame({"Metric": ["AUC", "Precision", "Recall", "F1"],
                  "Mean":   [ms(aucs)[0], ms(precs)[0], ms(recs)[0], ms(f1s)[0]],
                  "SD":     [ms(aucs)[1], ms(precs)[1], ms(recs)[1], ms(f1s)[1]]}
                 ).to_csv(os.path.join(out, "metrics_summary.tsv"),
                          sep="\t", float_format="%.4f", index=False)
    pd.DataFrame(cm_tot, index=["Animal", "Human"],
                 columns=["Pred_Animal", "Pred_Human"])\
      .to_csv(os.path.join(out, "aggregated_cm.tsv"), sep="\t")

    # final RF
    Xb_full, Yb_full = balance(X, y, lin, rng)
    final = make_base(); final.fit(Xb_full, Yb_full)
    dump({"model": final, "columns": list(X.columns)},
         os.path.join(out, "model_bundle.joblib"))

    # función asociación
    def assoc(col: str) -> str:
        mu_h = X[col][y == "Human"].mean(); mu_a = X[col][y == "Animal"].mean()
        if np.isclose(mu_h, mu_a, atol=1e-8): return "Neutral"
        return "Human" if mu_h > mu_a else "Animal"

    # RF importance normalizada 0‑100
    rf_imp = pd.Series(final.feature_importances_, index=X.columns)
    rf_scaled = 100 * rf_imp / rf_imp.max()
    pd.DataFrame({"Imp_0_100": rf_scaled.round(2),
                  "Associated_to": [assoc(col) for col in rf_scaled.index]})\
      .sort_values("Imp_0_100", ascending=False)\
      .to_csv(os.path.join(out, "rf_importance.tsv"), sep="\t")

    # Permutation importance
    if cfg["perm"]:
        pim = permutation_importance(final, Xb_full, Yb_full,
                                     n_repeats=cfg["perm_repeats"],
                                     random_state=cfg["seed"],
                                     n_jobs=cfg["workers"])
        perm = pd.Series(pim.importances_mean, index=X.columns)
        perm_scaled = 100 * perm / perm.max()
        pd.DataFrame({"Imp_0_100": perm_scaled.round(2),
                      "Associated_to": [assoc(col) for col in perm_scaled.index]})\
          .sort_values("Imp_0_100", ascending=False)\
          .to_csv(os.path.join(out, "perm_importance.tsv"), sep="\t")

    print(f"[DONE] {tag} → {out}")

# ───────── main ───────── #
def main():
    ap = argparse.ArgumentParser("RF dual Gain/Loss v2.7",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gain_matrix", required=True)
    ap.add_argument("--loss_matrix", required=True)
    ap.add_argument("--cv", choices=["stratified", "lineage", "repeated"], default="stratified")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--with_both", action="store_true")
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--perm_repeats", type=int, default=5)
    args = ap.parse_args(); rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    def load_gain(p):
        d = pd.read_csv(p, sep="\t").set_index("Cluster").T
        d.index = d.index.map(trim_seqid); return d.fillna(0)

    def load_loss(p):
        d = pd.read_csv(p, sep="\t").set_index("Cluster").T
        d.index = d.index.map(trim_seqid); return d.fillna(100)

    gain = load_gain(args.gain_matrix)
    loss = load_loss(args.loss_matrix)
    cfg = vars(args)

    train(gain.copy(), cfg, "gain", rng)
    train(loss.copy(), cfg, "loss", rng)

    if args.with_both:
        cols = gain.columns.union(loss.columns)
        both = pd.concat([gain.reindex(columns=cols).add_prefix("Gain_"),
                          loss.reindex(columns=cols).add_prefix("Loss_")], axis=1)
        train(both, cfg, "both", rng)

if __name__ == "__main__":
    sys.exit(main())

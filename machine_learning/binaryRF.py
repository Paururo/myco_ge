#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panaroo_classifier.py  (v1.3.6 — Importance 0‑100 + contribution counts)
=======================================================================

CLI para clasificar aislamientos **Animal** vs **Human** a partir de matrices
de presencia/ausencia de Panaroo.

• `gene_importances_selected.csv`  incluye Importance_0_100 (0–100).
• `gene_presence_contribution.csv` añade N_Animal y N_Human y Contrib.

Requisitos
----------
python ≥ 3.7 · pandas · numpy · scikit‑learn · joblib
Opcionales: shap · matplotlib · lightgbm
"""
###############################################################################
# Imports y configuraciones básicas
###############################################################################
import argparse
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# dependencias opcionales
try:
    import shap
    from shap import Explanation
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import matplotlib.pyplot as plt                 # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# --- globals ---
LOGGER = logging.getLogger("panaroo_classifier")
RANDOM_STATE = 42
SUFFIX = "-AMAP.panaroo"
DATE_FMT = "%Y%m%d-%H%M%S"

###############################################################################
# Helper de tiempo y logger
###############################################################################
def timestamp() -> str:
    return dt.datetime.now().strftime(DATE_FMT)


def setup_logger(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, stream=sys.stderr)

###############################################################################
# I/O helpers
###############################################################################
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "GeneCode" in df.columns:
        df = df.rename(columns={"GeneCode": "Gene"})
    elif "Gene" not in df.columns:
        alt = [c for c in df.columns if c.lower() in {"gene_id", "feature_id", "id"}]
        if alt:
            df = df.rename(columns={alt[0]: "Gene"})
        else:
            raise ValueError("No 'Gene' column found.")
    return df


def panaroo_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.endswith(SUFFIX)]
    if cols:
        return cols
    import re
    pat = re.compile(r"^(L|A)\d+(_.+)?$")
    return [c for c in df.columns if pat.match(str(c))]


def to_binary(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    bin_df = df[["Gene"] + list(cols)].copy()
    for c in cols:
        pres = bin_df[c].notna() & (bin_df[c].str.strip() != "")
        bin_df[c] = np.where(pres, 1, 0).astype(np.int8)
    return bin_df.set_index("Gene")

###############################################################################
# Meta y matriz de diseño
###############################################################################
def _meta_from_col(col: str) -> Tuple[Optional[str], str]:
    pre = col.split("_", 1)[0]
    if pre.upper().startswith("A") and pre[1:].isdigit():
        return "Animal", pre
    if pre.upper().startswith("L") and pre[1:].isdigit():
        return "Human", pre
    return None, pre


def design_matrix(bin_df: pd.DataFrame,
                  exclude: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series]:
    meta = pd.DataFrame([{"Sample": c,
                          "Class": _meta_from_col(c)[0],
                          "Lineage": _meta_from_col(c)[1]}
                         for c in bin_df.columns]).set_index("Sample")
    mask = meta["Class"].notna()
    meta = meta[mask]
    bin_df = bin_df.loc[:, mask]

    if exclude:
        mask2 = ~meta["Lineage"].isin(set(exclude))
        meta, bin_df = meta[mask2], bin_df.loc[:, mask2]

    X = bin_df.T
    y = meta["Class"]
    return X, y

###############################################################################
# Dataclass config
###############################################################################
@dataclass
class TrainCfg:
    table: Path
    outdir: Path = Path("results")
    cv_folds: int = 5
    n_estimators: int = 400
    max_depth: Optional[int] = None
    exclude_lineages: List[str] = field(default_factory=list)
    use_lgbm: bool = False
    skip_shap: bool = False
    log_level: str = "INFO"
    top_genes_plot: int = 20

###############################################################################
# Pipeline builder
###############################################################################
def build_pipeline(use_lgbm: bool,
                   n_estimators: int,
                   max_depth: Optional[int]) -> Pipeline:
    if use_lgbm and HAS_LGBM:
        clf: Any = LGBMClassifier(random_state=RANDOM_STATE,
                                  n_jobs=-1,
                                  class_weight="balanced",
                                  n_estimators=n_estimators,
                                  max_depth=max_depth or -1)
    else:
        clf = RandomForestClassifier(random_state=RANDOM_STATE,
                                     n_jobs=-1,
                                     class_weight="balanced",
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)
    selector = SelectFromModel(estimator=clf, threshold="median")
    return Pipeline([("selector", selector), ("clf", clf)])

###############################################################################
# TRAIN
###############################################################################
def train(cfg: TrainCfg) -> None:
    setup_logger(cfg.log_level)
    out = (cfg.outdir / timestamp()).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1 datos -------------------------------------------------------------
    raw = read_table(cfg.table)
    cols = panaroo_columns(raw)
    Xbin = to_binary(raw, cols)
    X, y = design_matrix(Xbin, cfg.exclude_lineages)

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    # 2 CV ---------------------------------------------------------------
    pipe = build_pipeline(cfg.use_lgbm, cfg.n_estimators, cfg.max_depth)
    cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    t, p = [], []
    cms = []
    for tr, te in cv.split(X, y_enc):
        pipe.fit(X.iloc[tr], y_enc[tr])
        pred = pipe.predict(X.iloc[te])
        p.append(pred)
        t.append(y_enc[te])
        cms.append(confusion_matrix(y_enc[te], pred, labels=le.transform(le.classes_)))

    report = classification_report(np.hstack(t), np.hstack(p),
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    (out / "classification_report_cv.json").write_text(json.dumps(report, indent=2))
    np.save(out / "confusion_matrices_cv.npy", np.stack(cms))

    # 3 ajuste final ------------------------------------------------------
    pipe.fit(X, y_enc)
    joblib.dump(pipe, out / "model.pkl")
    joblib.dump(le, out / "label_encoder.pkl")
    joblib.dump(X.columns.tolist(), out / "training_features.pkl")

    # 4 importancias 0‑100 ----------------------------------------------
    selector = pipe.named_steps["selector"]
    clf = pipe.named_steps["clf"]
    sel_feats = X.columns[selector.get_support()].tolist()
    joblib.dump(sel_feats, out / "selected_features.pkl")

    imp_df = pd.DataFrame()
    if hasattr(clf, "feature_importances_") and sel_feats:
        raw_imp = clf.feature_importances_
        imp_scaled = 100 * raw_imp / raw_imp.max() if raw_imp.max() > 0 else raw_imp
        imp_df = (pd.DataFrame({"Gene": sel_feats,
                                "Importance_0_100": imp_scaled})
                    .sort_values("Importance_0_100", ascending=False))
        imp_df.to_csv(out / "gene_importances_selected.csv",
                      index=False, float_format="%.2f")

    # 5 contribution counts ---------------------------------------------
    contrib_rows: List[Dict[str, Any]] = []
    X_sel = selector.transform(X)
    X_sel_df = pd.DataFrame(X_sel, columns=sel_feats, index=X.index)
    for i, g in enumerate(sel_feats):
        mask = X_sel_df.iloc[:, i] == 1
        cts = y.loc[mask].value_counts()
        n_an = int(cts.get("Animal", 0))
        n_hu = int(cts.get("Human", 0))
        if n_an + n_hu == 0:
            contrib = "NotObsPres"
        elif n_hu > n_an:
            contrib = "Favors_Human"
        elif n_an > n_hu:
            contrib = "Favors_Animal"
        else:
            contrib = "Neutral"
        contrib_rows.append({"Gene": g, "N_Animal": n_an,
                             "N_Human": n_hu, "Contrib": contrib})
    contrib_df = pd.DataFrame(contrib_rows)
    if not imp_df.empty:
        contrib_df = contrib_df.merge(imp_df, on="Gene", how="left")
    cols_order = ["Gene", "Importance_0_100",
                  "N_Animal", "N_Human", "Contrib"]
    contrib_df = contrib_df[cols_order]
    contrib_df.to_csv(out / "gene_presence_contribution.csv",
                      index=False, float_format="%.2f")

###############################################################################
# PREDICT
###############################################################################
def predict(table: Path, model_dir: Path, out_csv: Optional[Path]) -> None:
    setup_logger("INFO")
    pipe: Pipeline = joblib.load(model_dir / "model.pkl")
    le: LabelEncoder = joblib.load(model_dir / "label_encoder.pkl")
    feats: List[str] = joblib.load(model_dir / "training_features.pkl")

    raw = read_table(table)
    cols = panaroo_columns(raw)
    Xbin = to_binary(raw, cols)
    Xpred = Xbin.T.reindex(columns=feats, fill_value=0)

    pred = le.inverse_transform(pipe.predict(Xpred))
    res = pd.DataFrame({"Sample": Xpred.index, "Prediction": pred})
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(Xpred)
        res = res.join(pd.DataFrame(proba, columns=[f"Prob_{c}" for c in le.classes_],
                                    index=Xpred.index))
    dest = out_csv or model_dir / f"predictions_{timestamp()}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(dest, index=False, float_format="%.4f")
    LOGGER.info("Predictions saved → %s", dest)

###############################################################################
# CLI
###############################################################################
def _cli() -> argparse.ArgumentParser:
    pa = argparse.ArgumentParser("panaroo_classifier",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = pa.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="Train model")
    tr.add_argument("table", type=Path)
    tr.add_argument("--outdir", type=Path, default=Path("results"))
    tr.add_argument("--cv_folds", type=int, default=5)
    tr.add_argument("--n_estimators", type=int, default=400)
    tr.add_argument("--max_depth", type=int, default=None)
    tr.add_argument("--exclude", nargs="*", default=[])
    tr.add_argument("--lgbm", action="store_true")
    tr.add_argument("--skip_shap", action="store_true")
    tr.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    default="INFO")
    tr.add_argument("--top_genes_plot", type=int, default=20)
    pr = sub.add_parser("predict", help="Predict")
    pr.add_argument("table", type=Path)
    pr.add_argument("model_dir", type=Path)
    pr.add_argument("-o", "--output", type=Path, default=None)
    return pa

def main() -> None:
    args = _cli().parse_args()
    if args.cmd == "train":
        cfg = TrainCfg(table=args.table,
                       outdir=args.outdir,
                       cv_folds=args.cv_folds,
                       n_estimators=args.n_estimators,
                       max_depth=args.max_depth,
                       exclude_lineages=args.exclude,
                       use_lgbm=args.lgbm,
                       skip_shap=args.skip_shap,
                       log_level=args.log_level,
                       top_genes_plot=args.top_genes_plot)
        train(cfg)
    else:
        predict(args.table, args.model_dir, args.output)

if __name__ == "__main__":
    main()

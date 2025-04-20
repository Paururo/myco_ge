#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panaroo_classifier.py  (v1.3.4 – robust to SHAP ≥ 0.44) - MODIFIED v6 (Importance 0-100)
===================================================================
CLI para clasificar aislamientos **Animal** vs **Human**
a partir de matrices de presencia/ausencia de Panaroo.

Modifications v6:
* Scaled classifier `Importance` to 0-100 range in `gene_presence_contribution.csv`.
* Added `Importance_0_100` column and kept original `Importance`.

Modifications v5:
* Added `--top_genes_plot` argument to `train` command.
* Added generation of `gene_presence_contribution.csv`.
* Included classifier `Importance` and counts per class in contribution file.
* SHAP plots: Added plots showing only top N genes with positive mean
  contribution to the specific class (saved as *_positive.png).
  Includes fallback to standard plots if filtered plots fail.

Cambios clave en v1.3.4
-----------------------
* Manejo compatible con todas las formas de salida de `shap.TreeExplainer`.
* Importa `shap.Explanation`.
* Sane checks y logs más claros.

Requisitos
~~~~~~~~~~
python ≥ 3.7 · pandas · numpy · scikit‑learn · joblib
Opcionales: shap · matplotlib · lightgbm
"""
###############################################################################
# Imports
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

# Optional deps
try:
    import shap  # type: ignore
    from shap import Explanation  # type: ignore
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

###############################################################################
# Globals
###############################################################################
LOGGER = logging.getLogger("panaroo_classifier")
RANDOM_STATE = 42
# --- IMPORTANT: Update this suffix if your Panaroo output uses a different one ---
SUFFIX = "-AMAP.panaroo" # Example suffix, adjust as needed
DATE_FMT = "%Y%m%d-%H%M%S" # Format for timestamped output

###############################################################################
# Helper functions
###############################################################################

def timestamp() -> str:
    """Generates a timestamp string based on current local time."""
    return dt.datetime.now().strftime(DATE_FMT)


def setup_logger(level: str) -> None:
    """Configures the root logger for console output."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt, stream=sys.stderr)
    if log_level == logging.DEBUG:
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Reads the Panaroo TSV, checks for gene ID column, and standardizes it."""
    if not path.exists(): raise FileNotFoundError(f"Input file not found: {path}")
    try: df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    except Exception as e: raise IOError(f"Error reading TSV file {path}: {e}")

    if "GeneCode" in df.columns: df = df.rename(columns={"GeneCode": "Gene"})
    elif "Gene" not in df.columns:
        alt_cols = [c for c in df.columns if c.lower() in ['gene_id', 'feature_id', 'id']]
        if alt_cols: df = df.rename(columns={alt_cols[0]: "Gene"}); LOGGER.warning("Using '%s' as gene ID.", alt_cols[0])
        else: raise ValueError("Required gene ID column ('Gene', 'GeneCode', etc.) not found.")
    if df.empty: raise ValueError(f"Input file {path} is empty.")
    if 'Gene' not in df.columns or df['Gene'].isnull().all(): raise ValueError("Gene ID column ('Gene') missing or empty.")
    return df


def panaroo_columns(df: pd.DataFrame) -> List[str]:
    """Finds sample columns based on suffix or naming pattern."""
    cols = [c for c in df.columns if isinstance(c, str) and c.endswith(SUFFIX)]
    if cols: LOGGER.debug("Found %d columns with suffix '%s'.", len(cols), SUFFIX); return cols
    LOGGER.warning(f"No columns found with suffix '{SUFFIX}'. Trying L/A prefix pattern...")
    import re
    pattern = re.compile(r"^(L|A)\d+(_.+)?$")
    cols = [c for c in df.columns if isinstance(c, str) and pattern.match(c)]
    if cols: LOGGER.info(f"Found {len(cols)} columns based on L/A prefix pattern."); return cols
    LOGGER.error("Cannot identify sample columns by suffix or pattern."); return []


def to_binary(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Converts specified sample columns to binary presence (1) / absence (0)."""
    if "Gene" not in df.columns: raise ValueError("Internal error: 'Gene' column missing.")
    if df['Gene'].duplicated().any(): LOGGER.warning("Duplicate gene names found in 'Gene' column.")
    bin_df = df[['Gene'] + list(cols)].copy()
    for col in cols:
        is_present = bin_df[col].notna() & (bin_df[col].str.strip() != '')
        bin_df[col] = np.where(is_present, 1, 0).astype(np.int8)
    try: bin_df = bin_df.set_index("Gene", verify_integrity=False)
    except KeyError: raise ValueError("Failed to set 'Gene' column as index.")
    return bin_df

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _meta_from_col(col: str) -> Tuple[Optional[str], str]:
    """Extracts Class (Animal/Human) and Lineage prefix from sample ID."""
    parts = col.split("_", 1); lin_prefix = parts[0] if parts else col
    if lin_prefix.upper().startswith("A") and lin_prefix[1:].isdigit(): return "Animal", lin_prefix
    if lin_prefix.upper().startswith("L") and lin_prefix[1:].isdigit(): return "Human", lin_prefix
    LOGGER.debug("Prefix '%s' (from '%s') not recognized.", lin_prefix, col); return None, lin_prefix


def design_matrix(bin_df: pd.DataFrame, exclude: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Creates design matrix (X: samples x genes) and target vector (y: sample classes)."""
    meta_list = [{"Sample": c, "Class": _meta_from_col(c)[0], "Lineage": _meta_from_col(c)[1]} for c in bin_df.columns]
    meta = pd.DataFrame(meta_list).set_index("Sample")
    known_mask = meta["Class"].notna(); num_known = known_mask.sum()
    LOGGER.info("Found %d/%d samples with known Class (Animal/Human).", num_known, len(meta))
    if num_known == 0: raise ValueError("No samples found with recognizable 'Animal'/'Human' prefixes.")
    meta_known = meta[known_mask]; bin_df_filtered = bin_df.loc[:, known_mask]

    if exclude:
        exclude_set = set(exclude); exclude_mask = meta_known["Lineage"].isin(exclude_set)
        num_excluded = exclude_mask.sum()
        if num_excluded > 0:
            LOGGER.info("Excluding %d samples from lineages: %s", num_excluded, sorted(list(meta_known[exclude_mask]['Lineage'].unique())))
            keep_mask = ~exclude_mask; meta_final = meta_known[keep_mask]; bin_df_final = bin_df_filtered.loc[:, keep_mask]
        else: LOGGER.info("No samples matched exclusion list: %s", exclude); meta_final = meta_known; bin_df_final = bin_df_filtered
    else: meta_final = meta_known; bin_df_final = bin_df_filtered
    if meta_final.empty: raise ValueError("No samples remaining after filtering/exclusions.")

    X = bin_df_final.T; X.index.name = "Sample"; X.columns.name = "Gene"
    y = meta_final["Class"].copy(); y.index.name = "Sample"
    assert X.index.equals(y.index), "CRITICAL: X and y indices mismatch after processing."
    return X, y

###############################################################################
# Dataclass for Training Configuration
###############################################################################

@dataclass
class TrainCfg:
    """Configuration storage for the training process."""
    table: Path
    outdir: Path = Path("results")
    cv_folds: int = 5
    n_estimators: int = 400
    max_depth: Optional[int] = None
    exclude_lineages: List[str] = field(default_factory=list)
    use_lgbm: bool = False
    skip_shap: bool = False
    log_level: str = "INFO"
    top_genes_plot: int = 20 # Default number of genes for SHAP plots

###############################################################################
# Pipeline creation
###############################################################################

def build_pipeline(use_lgbm: bool, n_estimators: int, max_depth: Optional[int]) -> Pipeline:
    """Builds the scikit‑learn pipeline including feature selection and classifier."""
    if use_lgbm and HAS_LGBM:
        LOGGER.info("Using LightGBM classifier (n_estimators=%d, max_depth=%s).", n_estimators, max_depth or 'unlimited')
        clf: Any = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", n_estimators=n_estimators, max_depth=max_depth or -1, importance_type='gain') # MODIFICACIÓN: specify importance_type for consistency if needed
    else:
        if use_lgbm and not HAS_LGBM: LOGGER.warning("LightGBM not installed; falling back to RandomForest.")
        LOGGER.info("Using RandomForest classifier (n_estimators=%d, max_depth=%s).", n_estimators, max_depth or 'unlimited')
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", n_estimators=n_estimators, max_depth=max_depth)
    selector = SelectFromModel(estimator=clf, threshold="median", prefit=False)
    LOGGER.info("Pipeline: SelectFromModel(median) -> %s.", type(clf).__name__)
    return Pipeline([("selector", selector), ("clf", clf)])

###############################################################################
# Training Function
###############################################################################

def train(cfg: TrainCfg) -> None:
    """Trains the classifier, evaluates, calculates importances/SHAP, saves artifacts."""
    setup_logger(cfg.log_level)
    run_timestamp = timestamp(); out = (cfg.outdir / run_timestamp).resolve()
    try: out.mkdir(parents=True, exist_ok=True)
    except OSError as e: LOGGER.error(f"Failed to create output dir {out}: {e}"); raise
    LOGGER.info("="*60 + f"\n Starting Training Run: {run_timestamp}\n Output Directory: {out}\n" + "="*60)

    # --- 1. Load and Prepare Data ---
    LOGGER.info("--- Step 1: Loading and Preparing Data ---")
    try:
        raw = read_table(cfg.table); cols = panaroo_columns(raw)
        if not cols: raise ValueError(f"Cannot identify Panaroo sample columns in {cfg.table}.")
        Xbin = to_binary(raw, cols)
        LOGGER.info("Binary matrix: %d genes × %d samples.", Xbin.shape[0], Xbin.shape[1])
        X, y = design_matrix(Xbin, cfg.exclude_lineages)
        LOGGER.info("Training matrix: %d samples × %d genes.", X.shape[0], X.shape[1])
        LOGGER.info("Class distribution:\n%s", y.value_counts().to_string())
    except Exception as e: LOGGER.error("Data loading/prep failed: %s", e); raise

    # --- 2. Model Training and Cross-Validation ---
    LOGGER.info("--- Step 2: Model Training and Cross-Validation ---")
    le = LabelEncoder().fit(y); y_encoded = le.transform(y)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    LOGGER.info(f"Class encoding: {class_mapping}")
    n_classes = len(le.classes_)
    if n_classes != 2: LOGGER.warning("Binary classification assumed. Found %d classes: %s", n_classes, list(le.classes_))
    pipe = build_pipeline(cfg.use_lgbm, cfg.n_estimators, cfg.max_depth)
    LOGGER.info("Performing %d‑fold stratified CV...", cfg.cv_folds)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    t_all, p_all, cms = [], [], []
    try:
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), 1):
            pipe.fit(X.iloc[train_idx], y_encoded[train_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            t_all.append(y_encoded[test_idx]); p_all.append(y_pred)
            cms.append(confusion_matrix(y_encoded[test_idx], y_pred, labels=le.transform(le.classes_)))
    except Exception as e: LOGGER.error("CV error: %s", e, exc_info=True); raise
    y_true_cv, y_pred_cv = np.hstack(t_all), np.hstack(p_all)
    try:
        report = classification_report(y_true_cv, y_pred_cv, target_names=le.classes_, output_dict=True, zero_division=0)
        (out / "classification_report_cv.json").write_text(json.dumps(report, indent=2))
        LOGGER.info("CV Report saved. Accuracy=%.3f, Macro-F1=%.3f", report.get("accuracy",-1), report.get("macro avg",{}).get("f1-score",-1))
    except Exception as e: LOGGER.error("Failed saving CV report: %s", e)
    try: np.save(out / "confusion_matrices_cv.npy", np.stack(cms)); LOGGER.info("CV confusion matrices saved.")
    except Exception as e: LOGGER.error("Failed saving CV confusion matrices: %s", e)

    # --- 3. Final Model Fit ---
    LOGGER.info("--- Step 3: Fitting Final Model ---")
    try: pipe.fit(X, y_encoded)
    except Exception as e: LOGGER.error("Final fit error: %s", e, exc_info=True); raise
    try:
        joblib.dump(pipe, out / "model.pkl"); joblib.dump(le, out / "label_encoder.pkl")
        joblib.dump(X.columns.tolist(), out / "training_features.pkl")
        LOGGER.info("Final model, encoder, and feature list saved.")
    except Exception as e: LOGGER.error("Failed saving final model artifacts: %s", e)

    # --- 4. Feature Importance and Selection Results ---
    LOGGER.info("--- Step 4: Feature Importances & Selection ---")
    sel_features = []; shap_ready = False; imp_df = pd.DataFrame()
    try:
        selector = pipe.named_steps["selector"]; clf_step = pipe.named_steps["clf"]
        selected_mask = selector.get_support(); sel_features = X.columns[selected_mask].tolist()
        LOGGER.info("%d/%d features selected by SelectFromModel.", len(sel_features), X.shape[1])
        joblib.dump(sel_features, out / "selected_features.pkl"); LOGGER.info("Selected features list saved.")

        if hasattr(clf_step, "feature_importances_") and sel_features:
            importances = clf_step.feature_importances_
            if len(importances) == len(sel_features):
                # MODIFICACIÓN: Crear DataFrame inicial con importancias crudas
                imp_df = pd.DataFrame({"Gene": sel_features, "Importance": importances})

                # MODIFICACIÓN: Calcular importancia máxima y escalar a 0-100
                max_importance = imp_df["Importance"].max()
                if max_importance > 0:
                    imp_df["Importance_0_100"] = (imp_df["Importance"] / max_importance) * 100
                else:
                    imp_df["Importance_0_100"] = 0.0 # Handle case where max importance is 0
                    LOGGER.warning("Maximum feature importance is 0. Scaled importances set to 0.")

                # MODIFICACIÓN: Ordenar por la importancia original (o la escalada, ambas darían el mismo orden)
                imp_df = imp_df.sort_values("Importance", ascending=False)

                # MODIFICACIÓN: Guardar CSV con ambas columnas de importancia
                imp_df.to_csv(out / "gene_importances_selected.csv", index=False, float_format='%.5g')
                LOGGER.info("Feature importances (raw and scaled 0-100) saved.")
                shap_ready = True
            else: LOGGER.warning("Importance array length mismatch.")
        else:
            LOGGER.warning("Classifier '%s' lacks 'feature_importances_'.", type(clf_step).__name__)
            if sel_features and isinstance(clf_step, (RandomForestClassifier, LGBMClassifier)): shap_ready = True # SHAP TreeExplainer compatible
    except Exception as e: LOGGER.error("Error getting importances/selection: %s", e, exc_info=True)

    # --- 5. SHAP Analysis (Optional) ---
    LOGGER.info("--- Step 5: SHAP Analysis ---")
    if cfg.skip_shap: LOGGER.info("SHAP analysis skipped (--skip_shap).")
    elif not shap_ready: LOGGER.warning("SHAP prerequisites not met.")
    elif not HAS_SHAP: LOGGER.warning("SHAP library not installed.")
    elif not HAS_MPL: LOGGER.warning("Matplotlib library not installed.")
    else:
        try:
            LOGGER.info("Computing SHAP values for %d selected features...", len(sel_features))
            selector_step=pipe.named_steps["selector"]; clf_step=pipe.named_steps["clf"]
            X_sel = selector_step.transform(X)
            X_sel_df = pd.DataFrame(X_sel, columns=sel_features, index=X.index)
            explainer = shap.TreeExplainer(clf_step); shap_vals = explainer.shap_values(X_sel_df)

            LOGGER.debug("Standardizing SHAP output format...")
            shap_dict: Dict[str, np.ndarray] = {}
            # ... (Logic to populate shap_dict based on shap_vals type - same as previous version) ...
            if isinstance(shap_vals, list):
                if len(shap_vals) == n_classes: shap_dict = dict(zip(le.classes_, shap_vals))
                elif n_classes == 2 and len(shap_vals) == 1: shap_dict = {le.classes_[1]: shap_vals[0]} # Often case for binary RF/LGBM
                # MODIFICACIÓN: Handle binary case where SHAP returns values only for the positive class (1)
                elif n_classes == 2 and len(shap_vals) == 2 and np.allclose(shap_vals[0], -shap_vals[1]):
                    LOGGER.debug("Detected symmetric SHAP values for binary case. Using values for class '%s'.", le.classes_[1])
                    shap_dict = {le.classes_[1]: shap_vals[1]}
                else: LOGGER.warning("SHAP list structure unexpected: length %d for %d classes.", len(shap_vals), n_classes)

            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3 and shap_vals.shape[2] == n_classes:
                    for i,c in enumerate(le.classes_): shap_dict[c] = shap_vals[:,:,i]
                elif shap_vals.ndim == 2 and n_classes == 2: shap_dict = {le.classes_[1]: shap_vals} # SHAP values for class 1
                else: LOGGER.warning("SHAP ndarray structure unexpected: shape %s for %d classes.", shap_vals.shape, n_classes)

            elif HAS_SHAP and isinstance(shap_vals, Explanation):
                arr = shap_vals.values
                base = shap_vals.base_values # Could be useful later if needed
                if arr.ndim == 3 and arr.shape[2] == n_classes: # Multi-output
                    for i,c in enumerate(le.classes_): shap_dict[c] = arr[:,:,i]
                elif arr.ndim == 2 and n_classes == 2: # Binary case (often returns shape (n_samples, n_features))
                    shap_dict = {le.classes_[1]: arr} # Assume these are SHAP values for class 1
                else: LOGGER.warning("SHAP Explanation structure unexpected: shape %s for %d classes.", arr.shape, n_classes)
            if not shap_dict: raise ValueError("Could not process SHAP values into expected class dictionary.")

            # --- 5b. Calculate Gene Presence Contribution & Counts (Binary Case) ---
            if n_classes == 2:
                LOGGER.info("Calculating gene presence contribution and counts...")
                contribution_results = []
                animal_cls, human_cls = le.classes_[0], le.classes_[1] # Assume order [Animal, Human] or [0, 1]
                sv_human = None # Target SHAP values for 'Human' (class 1)

                if human_cls in shap_dict:
                    sv_human = shap_dict[human_cls]
                elif animal_cls in shap_dict and len(shap_dict) == 1:
                    # If only SHAP for class 0 is present, invert it for class 1 contribution
                    LOGGER.warning("Only SHAP values for class '%s' found. Inverting for '%s' contribution analysis.", animal_cls, human_cls)
                    sv_human = shap_dict[animal_cls] * -1
                elif len(shap_dict) > 0:
                     # Fallback if specific class names aren't found but dict isn't empty
                     first_key = list(shap_dict.keys())[0]
                     LOGGER.warning("Could not find SHAP for '%s' or '%s' specifically. Using values from key '%s' (assuming it corresponds to class 1).", human_cls, animal_cls, first_key)
                     sv_human = shap_dict[first_key]

                if sv_human is not None and not X_sel_df.empty and sel_features:
                    if sv_human.shape != X_sel_df.shape:
                         LOGGER.error(f"SHAP values shape {sv_human.shape} mismatch with selected data shape {X_sel_df.shape}.")
                    else:
                        for i, gene in enumerate(sel_features):
                            mask = (X_sel_df.iloc[:, i] == 1); contrib = "Indet."; c_animal, c_human = 0, 0
                            if mask.sum() > 0:
                                mean_shap = np.mean(sv_human[mask, i])
                                if mean_shap > 1e-6: contrib = f"Favors_{human_cls}"
                                elif mean_shap < -1e-6: contrib = f"Favors_{animal_cls}"
                                else: contrib = "Neutral"
                                counts = y.loc[X_sel_df.index[mask]].value_counts()
                                c_animal = counts.get(animal_cls, 0); c_human = counts.get(human_cls, 0)
                            else: contrib = "NotObsPres"
                            contribution_results.append({"Gene": gene, "Contrib": contrib, f"N_{animal_cls}": c_animal, f"N_{human_cls}": c_human})

                        if contribution_results:
                            contrib_df = pd.DataFrame(contribution_results)
                            # MODIFICACIÓN: Definir el orden deseado de columnas, incluyendo la escalada
                            col_order = ['Gene', 'Importance', 'Importance_0_100', f"N_{animal_cls}", f"N_{human_cls}", 'Contrib']

                            if not imp_df.empty and 'Gene' in imp_df.columns and 'Importance' in imp_df.columns and 'Importance_0_100' in imp_df.columns:
                                # MODIFICACIÓN: Hacer merge con imp_df para obtener ambas columnas de importancia
                                contrib_df=pd.merge(contrib_df, imp_df[['Gene','Importance', 'Importance_0_100']], on='Gene', how='left')
                                # MODIFICACIÓN: Ordenar por la importancia original (o escalada)
                                contrib_df = contrib_df.sort_values('Importance', ascending=False)
                            else:
                                # MODIFICACIÓN: Si imp_df está vacío o faltan columnas, añadir columnas vacías para mantener la estructura
                                if 'Importance' not in contrib_df.columns: contrib_df['Importance'] = np.nan
                                if 'Importance_0_100' not in contrib_df.columns: contrib_df['Importance_0_100'] = np.nan

                            # Reordenar y asegurar que todas las columnas esperadas estén presentes
                            final_cols = [c for c in col_order if c in contrib_df.columns] + [c for c in contrib_df.columns if c not in col_order]
                            contrib_df = contrib_df[final_cols]

                            contrib_file = out / "gene_presence_contribution.csv"
                            # MODIFICACIÓN: Asegurar formato correcto para ambas importancias
                            contrib_df.to_csv(contrib_file, index=False, float_format='%.5g')
                            LOGGER.info("✔ Gene contribution file (with scaled importance 0-100) saved: %s", contrib_file) # MODIFICACIÓN: Mensaje actualizado
                        else: LOGGER.warning("Contribution results list is empty.")
                else: LOGGER.warning("Cannot calculate contributions (missing suitable SHAP values/data).")
            else: LOGGER.info("Contribution analysis skipped (not binary).")

            # --- 5c. Save SHAP Importance CSVs and Plots ---
            LOGGER.info("Generating SHAP importance CSVs and plots...")
            gene_to_index = {gene: idx for idx, gene in enumerate(sel_features)} # For subsetting

            for cls_name, sv_array in shap_dict.items():
                    if sv_array.ndim != 2 or sv_array.shape[1] != len(sel_features):
                           LOGGER.error("SHAP array '%s' invalid shape %s.", cls_name, sv_array.shape); continue
                    mean_abs_shap = np.mean(np.abs(sv_array), axis=0)
                    mean_shap = np.mean(sv_array, axis=0) # Mean SHAP value per feature
                    shap_imp_df = pd.DataFrame({"Gene": sel_features, f"MeanAbsSHAP_{cls_name}": mean_abs_shap, f"MeanSHAP_{cls_name}": mean_shap})
                    shap_imp_df = shap_imp_df.sort_values(f"MeanAbsSHAP_{cls_name}", ascending=False)
                    shap_imp_df.to_csv(out / f"shap_importance_{cls_name}.csv", index=False, float_format='%.5g')

                    # --- Generate Standard Plots (Top N overall) ---
                    std_plot_kwargs = {'shap_values': sv_array, 'features': X_sel_df, 'max_display': cfg.top_genes_plot, 'show': False}
                    std_title = f"SHAP Class: {cls_name} (Top {cfg.top_genes_plot} Overall)"
                    try: plt.figure(); shap.summary_plot(**std_plot_kwargs, plot_type="dot"); plt.title(f"{std_title} - Summary"); plt.tight_layout(); plt.savefig(out / f"shap_summary_dot_{cls_name}.png", dpi=150); plt.close()
                    except Exception as e: LOGGER.error("Failed standard dot plot '%s': %s", cls_name, e)
                    try: plt.figure(); shap.summary_plot(**std_plot_kwargs, plot_type="bar"); plt.title(f"{std_title} - Global Importance"); plt.tight_layout(); plt.savefig(out / f"shap_summary_bar_{cls_name}.png", dpi=150); plt.close()
                    except Exception as e: LOGGER.error("Failed standard bar plot '%s': %s", cls_name, e)

                    # --- Generate Filtered Plots (Top N with Positive Mean SHAP) ---
                    positive_contrib_df = shap_imp_df[shap_imp_df[f"MeanSHAP_{cls_name}"] > 1e-6].copy() # Filter by positive mean SHAP
                    n_to_display = min(cfg.top_genes_plot, len(positive_contrib_df)) # How many to show
                    if n_to_display > 0:
                           filtered_top_genes_df = positive_contrib_df.head(n_to_display)
                           filtered_top_genes = filtered_top_genes_df["Gene"].tolist()
                           filtered_indices = [gene_to_index[gene] for gene in filtered_top_genes if gene in gene_to_index] # Check if gene exists
                           if len(filtered_indices) != len(filtered_top_genes):
                               LOGGER.warning("Mismatch finding indices for filtered top genes.")
                               missing_plot_genes = set(filtered_top_genes) - set(gene_to_index.keys())
                               LOGGER.warning("Genes missing from index: %s", missing_plot_genes)
                               # Continue with the indices found
                           if not filtered_indices:
                                LOGGER.warning(f"No valid indices found for positive contributors plot for class '{cls_name}'. Skipping.")
                                continue

                           sv_subset = sv_array[:, filtered_indices]
                           X_sel_subset = X_sel_df.iloc[:, filtered_indices]
                           # Ensure feature names match the subsetted data
                           pos_plot_kwargs = {'shap_values': sv_subset, 'features': X_sel_subset, 'feature_names': X_sel_subset.columns.tolist(), 'show': False}
                           pos_title = f"SHAP Class: {cls_name} (Top {len(filtered_indices)} Positive Mean Contrib.)"
                           LOGGER.info(f"Generating filtered plots for top {len(filtered_indices)} positive contributors for class '{cls_name}'.")
                           try: # Filtered Dot plot
                               plt.figure(); shap.summary_plot(**pos_plot_kwargs, plot_type="dot"); plt.title(f"{pos_title} - Summary"); plt.tight_layout(); plt.savefig(out / f"shap_summary_dot_positive_{cls_name}.png", dpi=150); plt.close()
                           except Exception as e: LOGGER.error("Failed filtered dot plot '%s': %s", cls_name, e, exc_info=cfg.log_level=="DEBUG")
                           try: # Filtered Bar plot
                               plt.figure(); shap.summary_plot(**pos_plot_kwargs, plot_type="bar"); plt.title(f"{pos_title} - Global Importance"); plt.tight_layout(); plt.savefig(out / f"shap_summary_bar_positive_{cls_name}.png", dpi=150); plt.close()
                           except Exception as e: LOGGER.error("Failed filtered bar plot '%s': %s", cls_name, e, exc_info=cfg.log_level=="DEBUG")
                    else:
                           LOGGER.warning(f"No features found with positive mean SHAP contribution for class '{cls_name}'. Skipping positive-only plots.")

                    LOGGER.debug("SHAP outputs processed for class '%s'.", cls_name)
            LOGGER.info("SHAP analysis outputs generated.")

        except Exception as e: LOGGER.error("SHAP analysis failed unexpectedly: %s", e, exc_info=True)

    # --- 6. Completion ---
    LOGGER.info("--- Training Run Completed ---")
    LOGGER.info("Results saved in: %s", out)
    LOGGER.info("="*60)

###############################################################################
# Prediction Function
###############################################################################

def predict(table: Path, model_dir: Path, out_csv: Optional[Path] = None) -> None:
    """Predicts classes for new samples using a previously trained model."""
    setup_logger("INFO")
    LOGGER.info("="*60 + f"\n Starting Prediction\n Model Dir: {model_dir}\n Data File: {table}\n" + "="*60)

    # --- 1. Load Model Artifacts ---
    LOGGER.info("--- Step 1: Loading Model Artifacts ---")
    try:
        model_pkl, le_pkl = model_dir / "model.pkl", model_dir / "label_encoder.pkl"
        sel_feat_pkl, train_feat_pkl = model_dir / "selected_features.pkl", model_dir / "training_features.pkl"
        if not model_pkl.is_file(): raise FileNotFoundError(f"Model file missing: {model_pkl}")
        if not le_pkl.is_file(): raise FileNotFoundError(f"Encoder file missing: {le_pkl}")

        # Determine which feature list to use
        if sel_feat_pkl.is_file():
            features_to_use = joblib.load(sel_feat_pkl)
            LOGGER.info("Using %d selected features from 'selected_features.pkl'.", len(features_to_use))
        elif train_feat_pkl.is_file():
            features_to_use = joblib.load(train_feat_pkl)
            LOGGER.warning("Using all %d training features from 'training_features.pkl' ('selected_features.pkl' missing).", len(features_to_use))
        else: raise FileNotFoundError(f"Feature list file ('selected_features.pkl' or 'training_features.pkl') missing in {model_dir}.")

        pipe: Pipeline = joblib.load(model_pkl); le: LabelEncoder = joblib.load(le_pkl)
        LOGGER.info(f"Model loaded. Classes: {list(le.classes_)}")
    except Exception as e: LOGGER.error("Error loading model: %s", e, exc_info=True); raise

    # --- 2. Load and Process Prediction Data ---
    LOGGER.info("--- Step 2: Loading Prediction Data ---")
    try:
        raw_pred = read_table(table); pred_cols = panaroo_columns(raw_pred)
        if not pred_cols: raise ValueError(f"No sample columns identified in {table}.")
        Xbin_pred = to_binary(raw_pred, pred_cols)
        LOGGER.info("Prediction data binary: %d genes × %d samples", Xbin_pred.shape[0], Xbin_pred.shape[1])
        X_pred_raw = Xbin_pred.T; X_pred_raw.index.name = "Sample"; X_pred_raw.columns.name = "Gene"
    except Exception as e: LOGGER.error("Failed loading prediction data: %s", e); raise

    # --- 3. Align Features and Predict ---
    LOGGER.info("--- Step 3: Aligning Features & Predicting ---")
    try:
        LOGGER.info("Aligning prediction data to %d model features.", len(features_to_use))
        # Ensure features_to_use is a list of strings
        if not isinstance(features_to_use, list) or not all(isinstance(f, str) for f in features_to_use):
             raise TypeError(f"Loaded features ('features_to_use') are not a list of strings. Type: {type(features_to_use)}")

        # Align columns: keep only those in features_to_use, add missing ones with 0
        X_pred_aligned = X_pred_raw.reindex(columns=features_to_use, fill_value=0)

        missing = set(features_to_use)-set(X_pred_raw.columns); extra = set(X_pred_raw.columns)-set(features_to_use)
        if missing: LOGGER.warning("%d expected features missing in input data (filled with 0): %s...", len(missing), list(missing)[:3])
        if extra: LOGGER.info("%d features in input data ignored (not used by model): %s...", len(extra), list(extra)[:3])
        if X_pred_aligned.shape[1] != len(features_to_use):
             raise ValueError(f"Feature alignment failed. Expected {len(features_to_use)} features, got {X_pred_aligned.shape[1]}.")
        if not all(X_pred_aligned.columns == features_to_use):
             LOGGER.warning("Columns after reindexing are not in the exact expected order, but should contain the correct features.")
             X_pred_aligned = X_pred_aligned[features_to_use] # Enforce order strictly


        res = pd.DataFrame(); pred_cols_base = ["Sample", "Prediction"]
        pred_cols_proba = [f"Prob_{c}" for c in le.classes_] if hasattr(pipe, "predict_proba") else []
        if not X_pred_aligned.empty:
            LOGGER.info("Predicting classes for %d samples...", X_pred_aligned.shape[0])

            # Check if the pipeline needs the selected features or all features before prediction
            # The loaded pipe *should* handle the feature selection internally if it was trained with it.
            # We provide the data aligned to the features expected *after* selection if 'selected_features.pkl' was used,
            # or all training features if 'training_features.pkl' was used.
            # If selected_features.pkl was used, the model expects input matching those selected features.
            # If training_features.pkl was used, the model expects all original training features and will apply selection internally.

            # MODIFICACIÓN: Determine which data frame to pass to predict based on which feature list was loaded.
            # The pipeline expects the *input* to the *first* step.
            # If we loaded 'selected_features.pkl', it means the pipeline saved was likely just the classifier step,
            # or the selector was applied *before* saving. This is ambiguous.
            # Safest bet: Re-align the raw prediction data (X_pred_raw) to the *original training* features if available.
            # If only selected features are available, assume the pipeline expects *only* those.

            data_for_prediction = X_pred_aligned # Default: Assume aligned data is correct input

            # Let's rethink: The pipeline object `pipe` loaded from model.pkl contains *all* steps, including the selector.
            # Therefore, it always expects the input data to have the *original* features that the *selector* expects.
            # We need the list of features the pipeline was originally trained on.
            original_training_features = None
            if train_feat_pkl.is_file():
                 original_training_features = joblib.load(train_feat_pkl)
                 LOGGER.debug("Loaded original training feature list from 'training_features.pkl'.")
            else:
                 # If original list is missing, we have a problem if selector is present.
                 # Fallback: Use features_to_use, assuming pipe might be just the clf.
                 LOGGER.warning("Original 'training_features.pkl' not found. Using the loaded 'features_to_use' list for prediction alignment. This might fail if the saved pipeline includes a selector step.")
                 original_training_features = features_to_use # Risky fallback

            if original_training_features:
                 LOGGER.info("Re-aligning prediction data to %d original training features.", len(original_training_features))
                 X_pred_realigned_for_pipe = X_pred_raw.reindex(columns=original_training_features, fill_value=0)
                 if X_pred_realigned_for_pipe.shape[1] != len(original_training_features):
                      raise ValueError("Re-alignment to original training features failed.")
                 data_for_prediction = X_pred_realigned_for_pipe
            else:
                 # Stick with the previous X_pred_aligned if original features couldn't be loaded
                 LOGGER.warning("Proceeding with prediction using data aligned to 'features_to_use'.")


            # Now predict using the correctly aligned data
            pred_enc = pipe.predict(data_for_prediction); pred_labels = le.inverse_transform(pred_enc)
            res = pd.DataFrame({"Sample": data_for_prediction.index, "Prediction": pred_labels})

            if pred_cols_proba: # Check if proba is expected
                try:
                    proba = pipe.predict_proba(data_for_prediction)
                    # Ensure proba aligns with results df index if prediction was successful
                    res = res.join(pd.DataFrame(proba, columns=pred_cols_proba, index=data_for_prediction.index))
                    LOGGER.info("Probabilities calculated.")
                except Exception as e: LOGGER.warning("Could not compute probabilities: %s", e); pred_cols_proba = [] # Reset if failed
        else: LOGGER.warning("Prediction data empty after alignment.")
        # Ensure DataFrame has columns even if empty
        res = res.reindex(columns=pred_cols_base + pred_cols_proba, fill_value=None)

    except Exception as e: LOGGER.error("Prediction error: %s", e, exc_info=True); raise

    # --- 4. Save Results ---
    LOGGER.info("--- Step 4: Saving Predictions ---")
    try:
        out_csv = out_csv.resolve() if out_csv else model_dir / f"predictions_{timestamp()}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False, float_format='%.4f')
        LOGGER.info("✔ Predictions saved to: %s", out_csv)
    except Exception as e: LOGGER.error("Failed saving predictions: %s", e)
    LOGGER.info("="*60 + "\n Prediction Run Completed\n" + "="*60)

###############################################################################
# CLI Definition
###############################################################################

def _build_cli() -> argparse.ArgumentParser:
    """Builds the command-line interface using argparse."""
    parser = argparse.ArgumentParser(description="Panaroo Animal/Human classifier.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, epilog=(f"Example Usage:\n  Train: python %(prog)s train Gpa.tsv --outdir results\n  Predict: python %(prog)s predict NewGpa.tsv results/DATE-TIME/ -o preds.csv\nNote: Sample columns should end with '{SUFFIX}' or use L/A prefix."))
    sub = parser.add_subparsers(dest="cmd", required=True, title="Commands", description="Choose 'train' or 'predict'")
    # Train Sub-parser
    tr = sub.add_parser("train", help="Train a new model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tr.add_argument("table", type=Path, help="Input Panaroo TSV file")
    tr.add_argument("--outdir", type=Path, default=Path("results"), help="Parent output directory")
    tr.add_argument("--cv_folds", type=int, default=5, metavar='N', help="CV folds (>= 2)")
    tr.add_argument("--n_estimators", type=int, default=400, metavar='N', help="Number of trees")
    tr.add_argument("--max_depth", type=int, default=None, metavar='N', help="Max tree depth (None=unlimited)")
    tr.add_argument("--exclude", nargs="*", dest="exclude_lineages", default=[], metavar="P", help="Lineage prefixes to exclude")
    tr.add_argument("--lgbm", action="store_true", dest="use_lgbm", help="Use LightGBM (requires install)")
    tr.add_argument("--skip_shap", action="store_true", help="Skip SHAP analysis")
    tr.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging verbosity")
    tr.add_argument("--top_genes_plot", type=int, default=20, metavar='N', help="Num genes for SHAP plots (> 0)")
    # Predict Sub-parser
    pr = sub.add_parser("predict", help="Predict using trained model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("table", type=Path, help="Input Panaroo TSV to classify")
    pr.add_argument("model_dir", type=Path, help="Directory with saved model artifacts (must contain model.pkl, label_encoder.pkl, and *features.pkl)")
    pr.add_argument("--output", "-o", type=Path, default=None, metavar="FILE.csv", help="Output CSV file path")
    return parser

###############################################################################
# Main Execution Block
###############################################################################

def main() -> None:
    """Parses CLI arguments and executes the chosen command."""
    parser = _build_cli()
    args = parser.parse_args()
    # --- Execute Train Command ---
    if args.cmd == "train":
        errors = []
        if args.cv_folds < 2: errors.append("--cv_folds must be >= 2")
        if args.n_estimators <= 0: errors.append("--n_estimators must be > 0")
        if args.top_genes_plot <= 0: errors.append("--top_genes_plot must be > 0")
        if errors: print("Error(s):\n"+"\n".join(errors), file=sys.stderr); sys.exit(1)
        cfg = TrainCfg(table=args.table, outdir=args.outdir, cv_folds=args.cv_folds,
                       n_estimators=args.n_estimators, max_depth=args.max_depth,
                       exclude_lineages=args.exclude_lineages, use_lgbm=args.use_lgbm,
                       skip_shap=args.skip_shap, log_level=args.log_level,
                       top_genes_plot=args.top_genes_plot)
        try: train(cfg)
        except Exception as e: logging.getLogger("main").error("Training failed: %s", e, exc_info=True); sys.exit(1)
    # --- Execute Predict Command ---
    elif args.cmd == "predict":
        try:
            # Ensure model directory and input table exist before proceeding
            model_dir = args.model_dir.resolve(strict=True) # Error if dir not found or not accessible
            if not model_dir.is_dir(): raise NotADirectoryError(f"Model directory path is not a directory: {model_dir}")

            table = args.table.resolve(strict=True) # Error if file not found or not accessible
            if not table.is_file(): raise FileNotFoundError(f"Input table file not found: {table}")

            # Resolve output path, create parent directory if needed
            output = None
            if args.output:
                output = args.output.resolve()
                try:
                    output.parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logging.getLogger("main").error(f"Could not create output directory {output.parent}: {e}")
                    sys.exit(1)

            predict(table, model_dir, output)
        except FileNotFoundError as e:
             logging.getLogger("main").error(f"Input Error: {e}")
             sys.exit(1)
        except NotADirectoryError as e:
             logging.getLogger("main").error(f"Input Error: {e}")
             sys.exit(1)
        except Exception as e:
            logging.getLogger("main").error("Prediction failed: %s", e, exc_info=True); sys.exit(1)

if __name__ == "__main__":
    # BasicConfig should ideally be called only once. setup_logger handles detailed config later.
    # Set a default level here in case setup_logger isn't called (e.g., before args parsing).
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stderr)
    main()

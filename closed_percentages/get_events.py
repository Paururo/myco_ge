#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara multifastas alineados contra dos referencias
  • RefPrefix  – secuencia cuyo ID empieza por --ref_prefix
  • CommonRef  – haplotipo más frecuente del fichero

Para cada (Gene‑Cluster, Lineage) calcula:
  – nº de secuencias totales
  – Longitud del alineamiento (AlignmentLength)
  – Para Gains y Losses vs RefPrefix y CommonRef:
      – nº y % de secuencias con ≥1 evento (NumEventSeqs, PercEventSeqs)
      – media (MeanPosPct) ± SD (SDPosPct) ± SE (SEPosPct) del % de
        posiciones afectadas en las secuencias que TIENEN el evento.
      – Intervalo de Confianza 95% para MeanPosPct (MeanPosPct_CI95_Lower/Upper) <<< NUEVO
      – Nº TOTAL de bases ganadas/perdidas en el grupo (TotalEventBases).
      – Nº PROMEDIO de bases ganadas/perdidas por secuencia en el grupo (AvgEventBases).
      – % del total de bases ganadas/perdidas respecto al total de
        posiciones en el alineamiento del grupo (PercTotalEventBases).

Salidas
  • <cluster>_event_details.tsv
  • summary_long_format.tsv (con columnas nuevas IC95 y AvgEventBases)
  • summary_majority_events.tsv (Opcional, filtrado del anterior)
"""
import os
import argparse
import warnings
from statistics import mean, stdev
from math import sqrt
from typing import List, Tuple, Optional, Dict, Any
from Bio import SeqIO
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# --- NUEVA DEPENDENCIA ---
try:
    from scipy.stats import t
except ImportError:
    print("[ERROR] La librería 'scipy' es necesaria para calcular intervalos de confianza.")
    print("Por favor, instálala con: pip install scipy")
    sys.exit(1) # Salir si no se encuentra scipy

# --- Ignorar advertencias ---
warnings.filterwarnings("ignore", message="Dataset has just one sample")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ───────────────────────── utilidades básicas ──────────────────────────── #

def compare_alignment(a: str, b: str) -> Tuple[int, int, int, int]:
    """Devuelve (gains, losses, mismatches, total_dist)"""
    g = l = m = 0
    if len(a) != len(b):
        raise ValueError(f"Longitudes desiguales: {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        if x == y: continue
        if x == '-' and y != '-': g += 1
        elif x != '-' and y == '-': l += 1
        else: m += 1
    return g, l, m, g + l + m

def most_common_hap(records: List[SeqIO.SeqRecord]) -> Optional[str]:
    """Secuencia más frecuente; si hay empate → la más larga."""
    if not records: return None
    freq: Dict[str, int] = {}
    for r in records: freq[str(r.seq)] = freq.get(str(r.seq), 0) + 1
    if not freq: return None
    try: top_freq = max(freq.values())
    except ValueError: return None
    candidates = [s for s, count in freq.items() if count == top_freq]
    if not candidates: return None
    return max(candidates, key=len)

def stat_tuple(vals: List[float]) -> Tuple[float, float, float]:
    """Calcula media, sd, se."""
    valid_vals = [v for v in vals if pd.notna(v)]
    n = len(valid_vals)
    if n == 0: return (np.nan, np.nan, np.nan)
    mu = mean(valid_vals)
    sd = stdev(valid_vals) if n > 1 else 0.0
    se = sd / sqrt(n) if n > 1 else 0.0
    return (mu, sd, se)

# ───────────────────── procesado de un FASTA (cluster) ─────────────────── #

def process_fasta(path: str, ref_prefix: str, out_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Procesa un archivo FASTA alineado."""
    cluster = os.path.splitext(os.path.basename(path))[0]
    recs = []
    try:
        recs_gen = SeqIO.parse(path, "fasta")
        first_rec = next(recs_gen, None)
        if first_rec is None: return None, None
        recs = [first_rec] + list(recs_gen)
    except Exception as e: print(f"[ERROR] {cluster} parse: {e}"); return None, None
    if len(recs) < 2: return None, None

    # --- Referencias ---
    pref_recs = [r for r in recs if r.id.startswith(ref_prefix)]
    if not pref_recs: print(f"[WARN] {cluster}: No RefPrefix '{ref_prefix}'."); return None, None
    if len(pref_recs) > 1: print(f"[WARN] {cluster}: Múltiples RefPrefix. Usando: {pref_recs[0].id}")
    ref_seq_rec = pref_recs[0]; ref_seq = str(ref_seq_rec.seq); aln_len = len(ref_seq)
    if aln_len == 0: print(f"[WARN] {cluster}: Aln len 0."); return None, None
    other_recs = [r for r in recs if r.id != ref_seq_rec.id]
    com_seq = most_common_hap(other_recs if other_recs else recs)
    if com_seq is None: print(f"[WARN] {cluster}: No CommonRef."); return None, None
    if len(com_seq) != aln_len: print(f"[WARN] {cluster}: Len CommonRef != RefPrefix."); return None, None

    # --- Calcular eventos ---
    rows = []
    lineage_extraction_failed = False
    for r in recs:
        seq = str(r.seq)
        if len(seq) != aln_len: print(f"[WARN] {cluster}: Seq {r.id} len error."); continue
        try:
            gR, lR, mR, dR = compare_alignment(ref_seq, seq)
            gC, lC, mC, dC = compare_alignment(com_seq, seq)
        except ValueError as e: print(f"[ERROR] {cluster}: Comparando {r.id}: {e}."); continue

        gR_pct = (gR / aln_len * 100) if aln_len > 0 else 0; lR_pct = (lR / aln_len * 100) if aln_len > 0 else 0
        dR_pct = (dR / aln_len * 100) if aln_len > 0 else 0; gC_pct = (gC / aln_len * 100) if aln_len > 0 else 0
        lC_pct = (lC / aln_len * 100) if aln_len > 0 else 0; dC_pct = (dC / aln_len * 100) if aln_len > 0 else 0

        try: lineage = r.id.split('_')[0]
        except IndexError:
            lineage = "Unknown"
            if not lineage_extraction_failed: print(f"[WARN] {cluster}: Error Lineage ID '{r.id}'."); lineage_extraction_failed = True

        rows.append({
            "Cluster": cluster, "SeqID": r.id, "Lineage": lineage, "AlignmentLength": aln_len,
            "Gains_Ref": gR, "Losses_Ref": lR, "Mismatches_Ref": mR, "Dist_Ref": dR,
            "Gains_Ref_pct": gR_pct, "Losses_Ref_pct": lR_pct, "Dist_Ref_pct": dR_pct,
            "Gains_Com": gC, "Losses_Com": lC, "Mismatches_Com": mC, "Dist_Com": dC,
            "Gains_Com_pct": gC_pct, "Losses_Com_pct": lC_pct, "Dist_Com_pct": dC_pct
        })

    if not rows:
        empty_df_cols = ["Cluster", "SeqID", "Lineage", "AlignmentLength", "Gains_Ref", "Losses_Ref", "Mismatches_Ref", "Dist_Ref", "Gains_Ref_pct", "Losses_Ref_pct", "Dist_Ref_pct", "Gains_Com", "Losses_Com", "Mismatches_Com", "Dist_Com", "Gains_Com_pct", "Losses_Com_pct", "Dist_Com_pct"]
        return pd.DataFrame(columns=empty_df_cols), None

    df = pd.DataFrame(rows)

    # --- Comprobar eventos y guardar ---
    has_any_event = False
    if not df.empty:
        event_cols = ["Gains_Ref", "Losses_Ref", "Gains_Com", "Losses_Com"]
        existing_cols = [c for c in event_cols if c in df.columns]
        if existing_cols:
            for col in existing_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            has_any_event = (df[existing_cols] > 0).any().any()

    if not has_any_event: return df, None # Devolver df pero sin path
    tsv_path = os.path.join(out_dir, f"{cluster}_event_details.tsv")
    try:
        df.to_csv(tsv_path, sep='\t', index=False, float_format='%.6f')
        return df, tsv_path
    except Exception as e: print(f"[ERROR] {cluster} save details: {e}"); return df, None

# ────────────────── Resumen Largo (Gene‑Cluster × Lineage) ───────────────── #

def build_summary_long(dfs: List[pd.DataFrame], out_dir: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Construye resumen largo, incluyendo AvgEventBases e Intervalos de Confianza 95%.
    """
    if not dfs: return None, None
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs: return None, None

    try: big_df = pd.concat([df for df in valid_dfs if isinstance(df, pd.DataFrame)], ignore_index=True)
    except Exception as e: print(f"[ERROR] Concat: {e}"); return None, None
    if big_df.empty: return None, None

    # --- Comprobación y conversión ---
    required_cols = ['Gains_Ref', 'Losses_Ref', 'Gains_Com', 'Losses_Com', 'Gains_Ref_pct', 'Losses_Ref_pct', 'Gains_Com_pct', 'Losses_Com_pct', 'AlignmentLength']
    missing_cols = [c for c in required_cols if c not in big_df.columns]
    if missing_cols: print(f"[ERROR] Faltan cols {missing_cols}"); return None, None
    for col in required_cols: big_df[col] = pd.to_numeric(big_df[col], errors='coerce')
    if any(big_df[col].isnull().all() for col in required_cols): print(f"[WARN] Alguna col numérica es NaN.")

    # --- Columnas booleanas ---
    big_df["HasGain_Ref"] = (big_df["Gains_Ref"] > 0) & pd.notna(big_df["Gains_Ref"])
    big_df["HasLoss_Ref"] = (big_df["Losses_Ref"] > 0) & pd.notna(big_df["Losses_Ref"])
    big_df["HasGain_Com"] = (big_df["Gains_Com"] > 0) & pd.notna(big_df["Gains_Com"])
    big_df["HasLoss_Com"] = (big_df["Losses_Com"] > 0) & pd.notna(big_df["Losses_Com"])

    summary_rows = []
    confidence_level = 0.95 # Para IC 95%
    # --- Agrupación y Cálculo ---
    for (cluster, lineage), group in big_df.groupby(["Cluster", "Lineage"]):
        total_seqs_in_group = len(group)
        if total_seqs_in_group == 0: continue

        # Longitud del alineamiento
        aln_len_series = group['AlignmentLength'].dropna(); aln_len = 0; total_aligned_positions = 0
        if not aln_len_series.empty:
            try: aln_len = int(aln_len_series.mode()[0])
            except (IndexError, ValueError): aln_len = 0
            if aln_len > 0: total_aligned_positions = aln_len * total_seqs_in_group

        # Sumas y promedios totales de bases
        total_gained_ref = group['Gains_Ref'].sum(skipna=True); total_lost_ref = group['Losses_Ref'].sum(skipna=True)
        total_gained_com = group['Gains_Com'].sum(skipna=True); total_lost_com = group['Losses_Com'].sum(skipna=True)
        avg_gained_ref = total_gained_ref / total_seqs_in_group if total_seqs_in_group > 0 else 0
        avg_lost_ref = total_lost_ref / total_seqs_in_group if total_seqs_in_group > 0 else 0
        avg_gained_com = total_gained_com / total_seqs_in_group if total_seqs_in_group > 0 else 0
        avg_lost_com = total_lost_com / total_seqs_in_group if total_seqs_in_group > 0 else 0

        # Porcentajes totales
        perc_total_gained_ref = (total_gained_ref / total_aligned_positions * 100) if total_aligned_positions > 0 else 0
        perc_total_lost_ref = (total_lost_ref / total_aligned_positions * 100) if total_aligned_positions > 0 else 0
        perc_total_gained_com = (total_gained_com / total_aligned_positions * 100) if total_aligned_positions > 0 else 0
        perc_total_lost_com = (total_lost_com / total_aligned_positions * 100) if total_aligned_positions > 0 else 0

        # --- Métricas por Evento/Referencia (incluyendo IC) ---

        def calculate_ci(data_pct_list, n_event_seqs, mean_pct, se_pct):
            """Calcula IC 95% para la media."""
            ci_lower, ci_upper = np.nan, np.nan
            # Necesitamos n>=2 y un error estándar válido y positivo para calcular IC
            if n_event_seqs >= 2 and pd.notna(se_pct) and se_pct > 0:
                df = n_event_seqs - 1
                try:
                    # Valor crítico t para IC bilateral
                    t_crit = t.ppf((1 + confidence_level) / 2, df)
                    margin_error = t_crit * se_pct
                    # Asegurar que los límites estén entre 0 y 100%
                    ci_lower = max(0.0, mean_pct - margin_error)
                    ci_upper = min(100.0, mean_pct + margin_error)
                except Exception as e: # Capturar errores de t.ppf (e.g., df=0)
                    # print(f"[WARN] No se pudo calcular IC (n={n_event_seqs}, se={se_pct}): {e}") # Opcional
                    pass # Dejar CIs como NaN
            return ci_lower, ci_upper

        # RefPrefix / Gain
        gain_ref_group = group.loc[group["HasGain_Ref"] & pd.notna(group["Gains_Ref_pct"])]
        num_gain_ref = len(gain_ref_group)
        perc_gain_ref = (num_gain_ref / total_seqs_in_group) * 100 if total_seqs_in_group > 0 else 0
        mu_gR, sd_gR, se_gR = stat_tuple(gain_ref_group["Gains_Ref_pct"].tolist())
        ci_l_gR, ci_u_gR = calculate_ci(gain_ref_group["Gains_Ref_pct"].tolist(), num_gain_ref, mu_gR, se_gR)
        summary_rows.append({
            "Cluster": cluster, "Lineage": lineage, "TotalSeqs": total_seqs_in_group, "AlignmentLength": aln_len if aln_len > 0 else np.nan,
            "ReferenceType": "RefPrefix", "EventType": "Gain", "NumEventSeqs": num_gain_ref, "PercEventSeqs": perc_gain_ref,
            "MeanPosPct": mu_gR, "SDPosPct": sd_gR, "SEPosPct": se_gR,
            "MeanPosPct_CI95_Lower": ci_l_gR, "MeanPosPct_CI95_Upper": ci_u_gR, # <<< IC
            "TotalEventBases": total_gained_ref, "AvgEventBases": avg_gained_ref, "PercTotalEventBases": perc_total_gained_ref
        })

        # RefPrefix / Loss
        loss_ref_group = group.loc[group["HasLoss_Ref"] & pd.notna(group["Losses_Ref_pct"])]
        num_loss_ref = len(loss_ref_group)
        perc_loss_ref = (num_loss_ref / total_seqs_in_group) * 100 if total_seqs_in_group > 0 else 0
        mu_lR, sd_lR, se_lR = stat_tuple(loss_ref_group["Losses_Ref_pct"].tolist())
        ci_l_lR, ci_u_lR = calculate_ci(loss_ref_group["Losses_Ref_pct"].tolist(), num_loss_ref, mu_lR, se_lR)
        summary_rows.append({
            "Cluster": cluster, "Lineage": lineage, "TotalSeqs": total_seqs_in_group, "AlignmentLength": aln_len if aln_len > 0 else np.nan,
            "ReferenceType": "RefPrefix", "EventType": "Loss", "NumEventSeqs": num_loss_ref, "PercEventSeqs": perc_loss_ref,
            "MeanPosPct": mu_lR, "SDPosPct": sd_lR, "SEPosPct": se_lR,
            "MeanPosPct_CI95_Lower": ci_l_lR, "MeanPosPct_CI95_Upper": ci_u_lR, # <<< IC
            "TotalEventBases": total_lost_ref, "AvgEventBases": avg_lost_ref, "PercTotalEventBases": perc_total_lost_ref
        })

        # CommonRef / Gain
        gain_com_group = group.loc[group["HasGain_Com"] & pd.notna(group["Gains_Com_pct"])]
        num_gain_com = len(gain_com_group)
        perc_gain_com = (num_gain_com / total_seqs_in_group) * 100 if total_seqs_in_group > 0 else 0
        mu_gC, sd_gC, se_gC = stat_tuple(gain_com_group["Gains_Com_pct"].tolist())
        ci_l_gC, ci_u_gC = calculate_ci(gain_com_group["Gains_Com_pct"].tolist(), num_gain_com, mu_gC, se_gC)
        summary_rows.append({
            "Cluster": cluster, "Lineage": lineage, "TotalSeqs": total_seqs_in_group, "AlignmentLength": aln_len if aln_len > 0 else np.nan,
            "ReferenceType": "CommonRef", "EventType": "Gain", "NumEventSeqs": num_gain_com, "PercEventSeqs": perc_gain_com,
            "MeanPosPct": mu_gC, "SDPosPct": sd_gC, "SEPosPct": se_gC,
            "MeanPosPct_CI95_Lower": ci_l_gC, "MeanPosPct_CI95_Upper": ci_u_gC, # <<< IC
            "TotalEventBases": total_gained_com, "AvgEventBases": avg_gained_com, "PercTotalEventBases": perc_total_gained_com
        })

        # CommonRef / Loss
        loss_com_group = group.loc[group["HasLoss_Com"] & pd.notna(group["Losses_Com_pct"])]
        num_loss_com = len(loss_com_group)
        perc_loss_com = (num_loss_com / total_seqs_in_group) * 100 if total_seqs_in_group > 0 else 0
        mu_lC, sd_lC, se_lC = stat_tuple(loss_com_group["Losses_Com_pct"].tolist())
        ci_l_lC, ci_u_lC = calculate_ci(loss_com_group["Losses_Com_pct"].tolist(), num_loss_com, mu_lC, se_lC)
        summary_rows.append({
            "Cluster": cluster, "Lineage": lineage, "TotalSeqs": total_seqs_in_group, "AlignmentLength": aln_len if aln_len > 0 else np.nan,
            "ReferenceType": "CommonRef", "EventType": "Loss", "NumEventSeqs": num_loss_com, "PercEventSeqs": perc_loss_com,
            "MeanPosPct": mu_lC, "SDPosPct": sd_lC, "SEPosPct": se_lC,
            "MeanPosPct_CI95_Lower": ci_l_lC, "MeanPosPct_CI95_Upper": ci_u_lC, # <<< IC
            "TotalEventBases": total_lost_com, "AvgEventBases": avg_lost_com, "PercTotalEventBases": perc_total_lost_com
        })

    if not summary_rows: print("[INFO] No rows for summary."); return None, None
    summary_df = pd.DataFrame(summary_rows)

    # --- Reordenar, Redondear, Guardar ---
    col_order = [ # Definir orden deseado con ICs
        "Cluster", "Lineage", "TotalSeqs", "AlignmentLength", "ReferenceType", "EventType",
        "NumEventSeqs", "PercEventSeqs",
        "MeanPosPct", "MeanPosPct_CI95_Lower", "MeanPosPct_CI95_Upper", # ICs aquí
        "SDPosPct", "SEPosPct",
        "TotalEventBases", "AvgEventBases", "PercTotalEventBases"
    ]
    existing_cols = [col for col in col_order if col in summary_df.columns]
    summary_df = summary_df[existing_cols]

    cols_to_round = { # Añadir redondeo para ICs
        "PercEventSeqs": 2, "MeanPosPct": 4, "SDPosPct": 4, "SEPosPct": 4,
        "MeanPosPct_CI95_Lower": 4, "MeanPosPct_CI95_Upper": 4, # Redondear ICs
        "AvgEventBases": 2, "PercTotalEventBases": 4
    }
    for col, decimals in cols_to_round.items():
         if col in summary_df.columns:
             summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').round(decimals)

    sort_cols = ["Cluster", "Lineage", "ReferenceType", "EventType"]
    existing_sort_cols = [col for col in sort_cols if col in summary_df.columns]
    if existing_sort_cols: summary_df = summary_df.sort_values(by=existing_sort_cols)

    out_tsv = os.path.join(out_dir, "summary_long_format.tsv")
    try:
        count_cols_to_int = ['TotalSeqs', 'AlignmentLength', 'NumEventSeqs', 'TotalEventBases']
        for col in count_cols_to_int:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').astype('Int64')
        summary_df.to_csv(out_tsv, sep='\t', index=False, na_rep='NaN')
        print(f"[INFO] Resumen largo guardado en: {out_tsv}")
        return out_tsv, summary_df
    except Exception as e:
        print(f"[ERROR] No se pudo guardar resumen: {e}")
        return None, summary_df if 'summary_df' in locals() else None

# ────────────────────────────────── main ────────────────────────────────── #
def main():
    start_time = time.time()
    p = argparse.ArgumentParser(description="Calcula eventos Gain/Loss y genera resúmenes.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input_dir", required=True, help="Dir con FASTAs alineados.")
    p.add_argument("-o", "--output_dir", required=True, help="Dir de salida para TSV.")
    p.add_argument("-r", "--ref_prefix", default="L4_G48915", help="Prefijo ID RefPrefix.")
    p.add_argument("--parallel", type=int, default=os.cpu_count(), help="Procesos paralelos.")
    p.add_argument("--majority_threshold", type=float, default=50.0, help="Umbral % 'PercEventSeqs' para filtrar majority_events.")
    args = p.parse_args()

    # --- Validaciones y Preparación ---
    if not os.path.isdir(args.input_dir): print(f"[ERROR] Input dir no existe: {args.input_dir}"); return 1
    if not 0 <= args.majority_threshold <= 100: print(f"[WARN] Umbral inválido. Usando 50.0."); args.majority_threshold = 50.0
    if args.parallel < 1: args.parallel = 1
    try: os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e: print(f"[ERROR] No se pudo crear output dir '{args.output_dir}': {e}"); return 1

    fasta_files = []
    print(f"[INFO] Buscando FASTA en: {args.input_dir}")
    try:
        for fname in os.listdir(args.input_dir):
            fpath = os.path.join(args.input_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".fasta", ".fa", ".fna")): fasta_files.append(fpath)
    except Exception as e: print(f"[ERROR] Error listando '{args.input_dir}': {e}"); return 1
    if not fasta_files: print(f"[ERROR] No FASTAs en: {args.input_dir}"); return 1

    total_files = len(fasta_files); print(f"[INFO] {total_files} FASTA encontrados."); print(f"[INFO] RefPrefix: '{args.ref_prefix}'")
    print(f"[INFO] Paralelismo: {args.parallel}"); print(f"[INFO] Umbral Mayoría: > {args.majority_threshold}%")

    # --- Procesamiento ---
    processed_data_frames: List[Optional[pd.DataFrame]] = []
    print(f"[INFO] Iniciando procesamiento...")
    processed_count, error_count, skipped_count, event_file_count = 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_path = { executor.submit(process_fasta, f, args.ref_prefix, args.output_dir): f for f in fasta_files }
        for future in as_completed(future_to_path):
            processed_count += 1; f_path = future_to_path[future]; f_name = os.path.basename(f_path)
            if processed_count % 100 == 0 or processed_count == total_files: print(f"    Progreso: {processed_count}/{total_files}...", end='\r')
            try:
                df_result, tsv_path_result = future.result()
                if df_result is not None: processed_data_frames.append(df_result)
                else: skipped_count += 1
                if tsv_path_result: event_file_count += 1
            except Exception as exc: print(f"\n[ERROR] Excepción procesando {f_name}: {exc}"); error_count += 1
    print(f"\n[INFO] Proc: {event_file_count} detalles guardados, {skipped_count} saltados, {error_count} errores graves.")

    # --- Resumen y Filtrado ---
    print("[INFO] Generando resumen agregado...")
    final_dfs_for_summary = [df for df in processed_data_frames if df is not None]
    if final_dfs_for_summary:
        summary_path, summary_df_generated = build_summary_long(final_dfs_for_summary, args.output_dir)
        if summary_df_generated is not None and not summary_df_generated.empty:
            if summary_path: print(f"[INFO] Resumen completo: {summary_path}")
            else: print("[WARN] Resumen generado pero no guardado.")

            # Filtrado Mayoritario
            print(f"[INFO] Filtrando por mayoría (> {args.majority_threshold}%)...")
            try:
                if 'PercEventSeqs' not in summary_df_generated.columns: print("[ERROR] Falta 'PercEventSeqs' para filtrar.")
                else:
                    summary_df_generated['PercEventSeqs'] = pd.to_numeric(summary_df_generated['PercEventSeqs'], errors='coerce')
                    filtered_df = summary_df_generated[summary_df_generated['PercEventSeqs'] > args.majority_threshold].copy()
                    if not filtered_df.empty:
                        filtered_tsv_path = os.path.join(args.output_dir, "summary_majority_events.tsv")
                        try:
                            for col in filtered_df.select_dtypes(include=['Int64']).columns: filtered_df[col] = filtered_df[col].astype(float)
                            filtered_df.to_csv(filtered_tsv_path, sep='\t', index=False, na_rep='NaN')
                            print(f"[INFO] {len(filtered_df)} eventos mayoritarios encontrados.")
                            print(f"[INFO] Resumen filtrado: {filtered_tsv_path}")
                        except Exception as e: print(f"[ERROR] No se pudo guardar resumen filtrado: {e}")
                    else: print(f"[INFO] No se encontraron eventos mayoritarios.")
            except Exception as e: print(f"[ERROR] Error durante filtrado: {e}")
        elif summary_path: print("[WARN] Resumen guardado, pero DF inválido para filtrar.")
        else: print("[WARN] No se pudo generar resumen principal ni filtrar.")
    else: print("[INFO] No hay datos válidos para generar resumen.")

    end_time = time.time(); print(f"[SUCCESS] Proceso completado en {end_time - start_time:.2f} segundos."); return 0

if __name__ == "__main__":
    import sys
    # Comprobar si scipy está disponible al inicio del script principal también
    try: from scipy.stats import t
    except ImportError: print("[ERROR] Dependencia 'scipy' no encontrada. Ejecuta: pip install scipy"); sys.exit(1)
    exit_code = main()
    sys.exit(exit_code)

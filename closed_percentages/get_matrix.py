#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_matrix.py — v3.0
===========================

• Compara multifastas alineados frente a:
    1) secuencia cuyo ID empieza por --ref_prefix (RefPrefix)
    2) haplotipo más frecuente del fichero (CommonRef)

• Calcula estadísticas de Gain/Loss (igual que versiones previas).

• NUEVO: genera 4 matrices (%Gain/%Loss × RefPrefix/CommonRef), filas=Cluster
  y columnas=SeqID recortado (primeros dos campos del ID).

  ─ gain_ref_matrix.tsv
  ─ loss_ref_matrix.tsv
  ─ gain_common_matrix.tsv
  ─ loss_common_matrix.tsv
  
python3 ref_event.py     -i aln_files/     -o test/     --ref_prefix L4_G48915 --parallel 8
"""
from __future__ import annotations
import os, argparse, warnings, time, sys
from statistics import mean, stdev, median
from math import sqrt
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np, pandas as pd
from Bio import SeqIO
try:
    from scipy.stats import t
except ImportError:
    print("[ERROR] scipy necesario. Instala con:  pip install scipy")
    sys.exit(1)

warnings.filterwarnings("ignore", message="Dataset has just one sample")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ───────── utilidades básicas ───────── #
def compare_alignment(a: str, b: str) -> Tuple[int, int, int, int]:
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
    if not records: return None
    freq: Dict[str, int] = {}
    for r in records: freq[str(r.seq)] = freq.get(str(r.seq), 0) + 1
    if not freq: return None
    top = max(freq.values()); cand = [s for s,c in freq.items() if c==top]
    return max(cand, key=len)

def stat_tuple(vals: List[float]) -> Tuple[float, float, float]:
    vals = [v for v in vals if pd.notna(v)]
    n = len(vals)
    if n == 0: return (np.nan, np.nan, np.nan)
    mu = mean(vals); sd = stdev(vals) if n>1 else 0.0; se = sd/sqrt(n) if n>1 else 0.0
    return (mu, sd, se)

# ───────── procesar un FASTA ───────── #
def process_fasta(path: str, ref_prefix: str, out_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    cluster = os.path.splitext(os.path.basename(path))[0]
    try:
        recs = list(SeqIO.parse(path,"fasta"))
    except Exception as e: print(f"[ERROR] {cluster} parse: {e}"); return None,None
    if len(recs)<2: return None,None

    pref = [r for r in recs if r.id.startswith(ref_prefix)]
    if not pref: print(f"[WARN] {cluster}: sin RefPrefix."); return None,None
    if len(pref)>1: print(f"[WARN] {cluster}: múltiples RefPrefix, uso {pref[0].id}")
    ref_seq = str(pref[0].seq); aln_len = len(ref_seq)
    others=[r for r in recs if r.id!=pref[0].id]
    com_seq = most_common_hap(others or recs)
    if com_seq is None or len(com_seq)!=aln_len: return None,None

    rows=[]
    for r in recs:
        seq=str(r.seq);  trim="_".join(r.id.split("_")[:2])
        if len(seq)!=aln_len: continue
        gR,lR,_,_ = compare_alignment(ref_seq,seq)
        gC,lC,_,_ = compare_alignment(com_seq,seq)
        rows.append({
            "Cluster":cluster,
            "SeqID":r.id,
            "SeqID_trim":trim,
            "Lineage":trim.split("_")[0],
            "AlignmentLength":aln_len,
            "Gains_Ref_pct":gR/aln_len*100,
            "Losses_Ref_pct":lR/aln_len*100,
            "Gains_Com_pct":gC/aln_len*100,
            "Losses_Com_pct":lC/aln_len*100})
    df=pd.DataFrame(rows)
    tsv=os.path.join(out_dir,f"{cluster}_event_details.tsv")
    df.to_csv(tsv,sep="\t",index=False,float_format="%.4f")
    return df, tsv

# ───────── matrices Gain/Loss (4 archivos) ───────── #
def build_matrices(dfs: List[pd.DataFrame], out_dir: str)->None:
    if not dfs: return
    big=pd.concat(dfs, ignore_index=True)
    def pivot(col:str,fname:str):
        mat=big.pivot_table(index="Cluster",
                            columns="SeqID_trim",
                            values=col,
                            aggfunc="first")
        mat.to_csv(os.path.join(out_dir,fname),sep="\t",float_format="%.4f")
        print(f"[INFO] Matriz → {fname}  ({mat.shape[0]}×{mat.shape[1]})")
    pivot("Gains_Ref_pct",   "gain_ref_matrix.tsv")
    pivot("Losses_Ref_pct",  "loss_ref_matrix.tsv")
    pivot("Gains_Com_pct",   "gain_common_matrix.tsv")
    pivot("Losses_Com_pct",  "loss_common_matrix.tsv")

# ───────── main ───────── #
def main():
    p=argparse.ArgumentParser("Gain/Loss matrices",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i","--input_dir",required=True)
    p.add_argument("-o","--output_dir",required=True)
    p.add_argument("-r","--ref_prefix",default="L4_G48915")
    p.add_argument("--parallel",type=int,default=os.cpu_count())
    args=p.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)
    fasta=[os.path.join(args.input_dir,f) for f in os.listdir(args.input_dir)
           if f.lower().endswith((".fasta",".fa",".fna"))]
    if not fasta: print("[ERROR] No FASTA found."); return

    dfs=[]
    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        fut={ex.submit(process_fasta,f,args.ref_prefix,args.output_dir):f for f in fasta}
        for i,done in enumerate(as_completed(fut),1):
            df,_=done.result();  print(f"  {i}/{len(fasta)}",end="\r")
            if df is not None: dfs.append(df)
    print()

    if dfs:
        print("[INFO] Generando matrices …")
        build_matrices(dfs,args.output_dir)
    else:
        print("[WARN] No se generaron matrices (sin datos válidos).")

if __name__=="__main__":
    sys.exit(main())

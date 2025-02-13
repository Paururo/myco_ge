#!/usr/bin/env python3
import argparse
import logging
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

def perform_enrichment(database_file, loci_file, output_file):
    # Load the mycobrowser database (TSV file)
    logging.info("Loading mycobrowser database from '%s'", database_file)
    try:
        db_df = pd.read_csv(database_file, sep='\t')
    except Exception as e:
        logging.error("Error reading the database file: %s", e)
        raise

    # Check that required columns exist
    if "Locus" not in db_df.columns or "Functional_Category" not in db_df.columns:
        logging.error("Database file must contain 'Locus' and 'Functional_Category' columns.")
        return

    # Load the list of loci (TSV file with one locus per line)
    logging.info("Loading loci list from '%s'", loci_file)
    try:
        with open(loci_file, "r") as f:
            loci_list = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error("Error reading the loci file: %s", e)
        raise

    logging.info("Number of loci in input list: %d", len(loci_list))
    
    # Define the background from the database
    background_genes = set(db_df["Locus"].unique())
    N = len(background_genes)
    logging.info("Total number of genes in the background: %d", N)
    
    # Intersect the input loci with the background
    input_loci = set(loci_list).intersection(background_genes)
    n = len(input_loci)
    logging.info("Number of input loci present in the background: %d", n)
    if n == 0:
        logging.error("None of the input loci were found in the background. Exiting.")
        return

    # Perform enrichment analysis for each functional category
    results = []
    grouped = db_df.groupby("Functional_Category")
    for func_cat, group in grouped:
        genes_in_category = set(group["Locus"])
        M = len(genes_in_category)  # Total genes in this functional category in the background
        k = len(input_loci.intersection(genes_in_category))  # Overlap with input loci

        # Calculate the right-tailed hypergeometric p-value:
        # hypergeom.sf(k-1, N, M, n) gives the probability of k or more successes.
        p_value = hypergeom.sf(k - 1, N, M, n)
        results.append({
            "Functional_Category": func_cat,
            "k": k,
            "M": M,
            "n": n,
            "N": N,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)
    
    # Apply Benjamini-Hochberg correction
    logging.info("Performing BH correction on the p-values")
    rejected, pvals_corrected, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
    results_df["p_value_adj"] = pvals_corrected
    results_df["significant"] = rejected

    # Filter for significant enrichment (adjusted p-value < 0.05)
    sig_results = results_df[results_df["p_value_adj"] < 0.05]
    logging.info("Number of significant functional categories: %d", sig_results.shape[0])
    
    # Write significant results to the output TSV file
    try:
        sig_results.to_csv(output_file, sep='\t', index=False)
        logging.info("Significant enrichment results written to '%s'", output_file)
    except Exception as e:
        logging.error("Error writing the output file: %s", e)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Gene Enrichment Analysis Script using mycobrowser database (TSV files)"
    )
    parser.add_argument("-d", "--database", required=True,
                        help="Path to the mycobrowser database TSV file containing 'Locus' and 'Functional_Category' columns")
    parser.add_argument("-l", "--loci", required=True,
                        help="Path to the TSV file containing a list of loci (one per line) for enrichment analysis")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the output TSV file for significant enrichment results (adjusted p-value < 0.05)")
    args = parser.parse_args()

    # Set up logging with timestamp and log level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    perform_enrichment(args.database, args.loci, args.output)

if __name__ == "__main__":
    main()

"""
Exploratory Data Analysis (EDA) on LSD600 Dataset.

Generates comprehensive statistics on:
- Relation distributions across Train, Dev, Test splits.
- Cross-sentence vs Intra-sentence relation proportions.
- Entity frequencies (LSFs and Diseases) and co-occurrence patterns.
- Sequence length statistics.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter


def run_eda():
    meta_path = "data/LSD600_metadata.tsv"
    rel_path = "data/Consolidated_Relations_Dataset.tsv"

    if not os.path.exists(meta_path) or not os.path.exists(rel_path):
        print("Data files not found. Ensure Zenodo datasets are downloaded to data/ directory.")
        return

    df_meta = pd.read_csv(meta_path, sep="\t")
    df_rel = pd.read_csv(rel_path, sep="\t")

    print("=" * 60)
    print("LSD600 CORPUS EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n1. Metadata Summary:")
    print(f"Total Abstracts: {len(df_meta)}")
    print(f"Abstracts per Split:\n{df_meta['Data_Set'].value_counts().to_string()}")
    print(f"Total Lifestyle Factor mentions: {df_meta['Lifestyle_Factor_Count'].sum()}")
    print(f"Total Disease mentions: {df_meta['Disease_Count'].sum()}")
    print(f"Total Annotated Relations: {df_meta['Relations_Count'].sum()}")

    print("\n2. Relation Types Distribution:")
    rel_counts = df_rel["Relationship_Type"].value_counts()
    for rtype, count in rel_counts.items():
        pct = (count / len(df_rel)) * 100
        print(f"  {rtype:<35}: {count:5d} ({pct:5.2f}%)")

    print("\n3. Relation Types by Split:")
    crosstab = pd.crosstab(df_rel["Relationship_Type"], df_rel["Data_Set"])
    print(crosstab.to_string())

    print("\n4. Top 10 Most Frequent Disease Entities:")
    top_dis = df_rel["Disease_Entity"].value_counts().head(10)
    for ent, cnt in top_dis.items():
        print(f"  {ent:<30}: {cnt}")

    print("\n5. Top 10 Most Frequent Lifestyle Factor Entities:")
    top_lsf = df_rel["LSF_Entity"].value_counts().head(10)
    for ent, cnt in top_lsf.items():
        print(f"  {ent:<30}: {cnt}")

    print("\n6. Abstract Text Length Statistics (Word count):")
    word_counts = df_meta["Abstract_Text"].dropna().apply(lambda x: len(x.split()))
    print(f"  Mean: {word_counts.mean():.1f} words")
    print(f"  Median: {word_counts.median():.1f} words")
    print(f"  Min: {word_counts.min()} words | Max: {word_counts.max()} words")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()

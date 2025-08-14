"""Merge satellite embeddings with IPC annual labels to create training dataset.

This script:
1. Loads the satellite embeddings CSV (64-D vectors + Area_ID + year)
2. Loads the IPC annual labels CSV (5 population percentages + Area_ID + Start_year)
3. Merges them on Area_ID and year to create the final training dataset
4. Saves the result as training_dataset.csv

Run:
    python merge_datasets.py
"""
import pathlib
import pandas as pd
import numpy as np


def merge_training_data(
    embeddings_csv: pathlib.Path,
    labels_csv: pathlib.Path,
    output_csv: pathlib.Path,
) -> None:
    """Merge satellite embeddings with IPC labels to create training dataset.
    
    Parameters
    ----------
    embeddings_csv : pathlib.Path
        Path to satellite embeddings CSV file.
    labels_csv : pathlib.Path
        Path to IPC annual labels CSV file.
    output_csv : pathlib.Path
        Path for output training dataset CSV.
    """
    
    print(f"Loading satellite embeddings from: {embeddings_csv}")
    if not embeddings_csv.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_csv}")
    
    embeddings_df = pd.read_csv(embeddings_csv)
    print(f"Embeddings dataset: {len(embeddings_df):,} rows, {len(embeddings_df.columns)} columns")
    
    print(f"Loading IPC labels from: {labels_csv}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    
    labels_df = pd.read_csv(labels_csv)
    print(f"Labels dataset: {len(labels_df):,} rows, {len(labels_df.columns)} columns")
    
    # Merge on Area_ID and year (embeddings) with Start_year (labels)
    print("Merging datasets on Area_ID and year...")
    merged_df = pd.merge(
        embeddings_df,
        labels_df,
        left_on=["Area_ID", "year"],
        right_on=["Area_ID", "Start_year"],
        how="inner"  # Only keep rows that exist in both datasets
    )
    
    print(f"Merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
    
    # Drop redundant Start_year column (we have year)
    if "Start_year" in merged_df.columns:
        merged_df = merged_df.drop("Start_year", axis=1)
    
    # Verify we have the expected structure
    band_cols = [col for col in merged_df.columns if col.startswith("band_") or col.startswith("embedding_")]
    phase_cols = [col for col in merged_df.columns if col.startswith("avg_pop_phase_")]
    
    print(f"\nDataset structure:")
    print(f"  Satellite embedding features: {len(band_cols)} columns")
    print(f"  IPC phase labels: {len(phase_cols)} columns")
    print(f"  Metadata columns: {len(merged_df.columns) - len(band_cols) - len(phase_cols)} columns")
    
    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"  Unique areas: {merged_df['Area_ID'].nunique():,}")
    print(f"  Years covered: {sorted(merged_df['year'].unique())}")
    print(f"  Countries: {merged_df['Country'].nunique()} unique")
    
    # Show top countries
    print(f"\nTop 5 countries by data points:")
    country_counts = merged_df['Country'].value_counts().head(5)
    for country, count in country_counts.items():
        percentage = (count / len(merged_df)) * 100
        print(f"  {country}: {count:,} rows ({percentage:.1f}%)")
    
    # Verify phase percentages
    if phase_cols:
        total_phase_pct = merged_df[phase_cols].sum(axis=1)
        print(f"\nPhase percentage validation:")
        print(f"  Mean total percentage: {total_phase_pct.mean():.3f}")
        print(f"  Min total percentage: {total_phase_pct.min():.3f}")
        print(f"  Max total percentage: {total_phase_pct.max():.3f}")
        
        # Show average phase distribution
        print(f"\nAverage population distribution by IPC phase:")
        for col in sorted(phase_cols):
            avg_pct = merged_df[col].mean()
            print(f"  {col}: {avg_pct:.3f} ({avg_pct*100:.1f}%)")
    
    # Check for any missing values
    missing_counts = merged_df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nMissing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing values")
    else:
        print(f"\nNo missing values found ✓")
    
    # Save the merged dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    
    print(f"\nTraining dataset saved to: {output_csv}")
    print(f"Dataset is ready for machine learning model training! 🚀")


def main():
    src_dir = pathlib.Path(__file__).resolve().parent
    
    embeddings_path = src_dir / "data" / "satellite_embeddings_level1_FAST.csv"
    labels_path = src_dir / "data" / "ipc_annual_dataset.csv"
    output_path = src_dir / "data" / "training_dataset.csv"
    
    merge_training_data(embeddings_path, labels_path, output_path)


if __name__ == "__main__":
    main() 
import pathlib
import pandas as pd
import numpy as np


def create_annual_ipc_dataset(
    input_csv: pathlib.Path,
    output_csv: pathlib.Path,
    area_column: str = "Area",
    country_column: str = "Country", 
    level1_column: str = "Level 1",
    start_date_column: str = "From",
    phase_column: str = "Phase",
    percentage_column: str = "Percentage",
) -> None:
    """Convert IPC data to annual averages grouped by area and start year.
    
    Parameters
    ----------
    input_csv : pathlib.Path
        Path to the filtered IPC CSV file.
    output_csv : pathlib.Path
        Path for the output annual dataset.
    area_column : str, optional
        Name of the administrative area column.
    country_column : str, optional
        Name of the country column.
    level1_column : str, optional
        Name of the level 1 administrative division column.
    start_date_column : str, optional
        Name of the start date column.
    phase_column : str, optional
        Name of the IPC phase column.
    percentage_column : str, optional
        Name of the population percentage column.
    """
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    print(f"Loading IPC data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Convert start date to datetime and extract year
    df[start_date_column] = pd.to_datetime(df[start_date_column])
    df['Start_year'] = df[start_date_column].dt.year
    
    print(f"Original dataset: {len(df):,} rows")
    print(f"Date range: {df['Start_year'].min()} - {df['Start_year'].max()}")
    
    # Create unique area identifier combining country, level1, and area
    df['Area_ID'] = df[country_column].astype(str) + "_" + \
                    df[level1_column].fillna("").astype(str) + "_" + \
                    df[area_column].astype(str)
    
    # Group by Level 1, Start_year, and Phase, then calculate mean percentage
    print("Grouping and aggregating data by Level 1, year, and phase...")
    grouped = df.groupby([level1_column, 'Start_year', phase_column])[percentage_column].mean().reset_index()
    
    # Pivot to create columns for each phase
    print("Pivoting data to create phase columns...")
    pivoted = grouped.pivot_table(
        index=[level1_column, 'Start_year'],
        columns=phase_column,
        values=percentage_column,
        fill_value=0.0
    ).reset_index()
    
    # Flatten column names
    pivoted.columns.name = None
    
    # Ensure all 5 phases are present as columns
    phase_columns = []
    for phase in [1, 2, 3, 4, 5]:
        col_name = f'avg_pop_phase_{phase}'
        if phase in pivoted.columns:
            pivoted.rename(columns={phase: col_name}, inplace=True)
        else:
            pivoted[col_name] = 0.0
        phase_columns.append(col_name)
    
    # Reorder columns for clarity
    final_columns = [level1_column, 'Start_year'] + phase_columns
    annual_df = pivoted[final_columns].copy()
    
    # Verify percentages sum to reasonable values (should be close to 1.0 for complete data)
    annual_df['total_percentage'] = annual_df[phase_columns].sum(axis=1)
    
    print(f"\nAnnual dataset created: {len(annual_df):,} rows")
    print(f"Unique areas: {annual_df[level1_column].nunique():,}")
    print(f"Years covered: {annual_df['Start_year'].nunique()}")
    print(f"Year range: {annual_df['Start_year'].min()} - {annual_df['Start_year'].max()}")
    
    # Show percentage sum statistics
    print(f"\nPercentage sum statistics:")
    print(f"Mean total percentage: {annual_df['total_percentage'].mean():.3f}")
    print(f"Median total percentage: {annual_df['total_percentage'].median():.3f}")
    print(f"Min total percentage: {annual_df['total_percentage'].min():.3f}")
    print(f"Max total percentage: {annual_df['total_percentage'].max():.3f}")
    
    # Show sample of phase distributions
    print(f"\nSample phase distributions:")
    print("Average population percentages by phase:")
    for col in phase_columns:
        avg_pct = annual_df[col].mean()
        print(f"  {col}: {avg_pct:.3f} ({avg_pct*100:.1f}%)")
    
    # Drop the temporary total_percentage column before saving
    annual_df = annual_df.drop('total_percentage', axis=1)
    
    # Save the annual dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(output_csv, index=False)
    
    print(f"\nAnnual dataset saved to: {output_csv}")
    print(f"Ready for machine learning model training!")


if __name__ == "__main__":
    src_dir = pathlib.Path(__file__).resolve().parent
    
    input_path = src_dir / "data" / "ipc_global_area_long_current_only.csv"
    output_path = src_dir / "data" / "ipc_annual_dataset.csv"
    
    create_annual_ipc_dataset(input_path, output_path) 
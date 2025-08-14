import pathlib
import pandas as pd
from datetime import datetime
import numpy as np


def analyze_ipc_dataset(csv_path: pathlib.Path) -> None:
    """Analyze the filtered IPC dataset and provide comprehensive statistics.
    
    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the filtered IPC CSV file.
    """
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert date columns to datetime
    df['From'] = pd.to_datetime(df['From'])
    df['To'] = pd.to_datetime(df['To'])
    df['Date of analysis'] = pd.to_datetime(df['Date of analysis'], format='%b %Y')
    
    # Calculate duration in days
    df['Duration_days'] = (df['To'] - df['From']).dt.days
    
    print("=" * 60)
    print("IPC DATASET ANALYSIS")
    print("=" * 60)
    
    # Basic dataset info
    total_rows = len(df)
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total data rows: {total_rows:,}")
    print(f"Date range: {df['From'].min().strftime('%Y-%m-%d')} to {df['To'].max().strftime('%Y-%m-%d')}")
    print(f"Analysis periods: {df['Date of analysis'].min().strftime('%b %Y')} to {df['Date of analysis'].max().strftime('%b %Y')}")
    
    # Country analysis
    print(f"\n🌍 COUNTRY DISTRIBUTION")
    country_counts = df['Country'].value_counts()
    total_countries = len(country_counts)
    print(f"Total unique countries: {total_countries}")
    print(f"Top 10 countries by observation count:")
    
    for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
        percentage = (count / total_rows) * 100
        print(f"  {i:2d}. {country}: {count:,} rows ({percentage:.1f}%)")
    
    # Year analysis
    print(f"\n📅 YEAR DISTRIBUTION")
    df['Year'] = df['From'].dt.year
    year_counts = df['Year'].value_counts().sort_index()
    total_years = len(year_counts)
    print(f"Total unique years: {total_years}")
    print(f"Year breakdown:")
    
    for year, count in year_counts.items():
        percentage = (count / total_rows) * 100
        print(f"  {year}: {count:,} rows ({percentage:.1f}%)")
    
    # Duration analysis
    print(f"\n⏱️  DURATION ANALYSIS")
    avg_duration = df['Duration_days'].mean()
    median_duration = df['Duration_days'].median()
    min_duration = df['Duration_days'].min()
    max_duration = df['Duration_days'].max()
    
    print(f"Average duration: {avg_duration:.1f} days ({avg_duration/30.44:.1f} months)")
    print(f"Median duration: {median_duration:.0f} days ({median_duration/30.44:.1f} months)")
    print(f"Min duration: {min_duration} days")
    print(f"Max duration: {max_duration} days ({max_duration/365.25:.1f} years)")
    
    # Duration distribution
    duration_ranges = [
        (0, 30, "≤ 1 month"),
        (31, 90, "1-3 months"),
        (91, 180, "3-6 months"),
        (181, 365, "6-12 months"),
        (366, float('inf'), "> 1 year")
    ]
    
    print(f"\nDuration distribution:")
    for min_days, max_days, label in duration_ranges:
        if max_days == float('inf'):
            mask = df['Duration_days'] >= min_days
        else:
            mask = (df['Duration_days'] >= min_days) & (df['Duration_days'] <= max_days)
        
        count = mask.sum()
        percentage = (count / total_rows) * 100
        print(f"  {label}: {count:,} rows ({percentage:.1f}%)")
    
    # Seasonal analysis
    print(f"\n🌱 SEASONAL ANALYSIS (by start month)")
    df['Start_month'] = df['From'].dt.month
    df['Season'] = df['Start_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    season_counts = df['Season'].value_counts()
    print(f"Observations by season:")
    for season in ['Spring', 'Summer', 'Fall', 'Winter']:
        if season in season_counts:
            count = season_counts[season]
            percentage = (count / total_rows) * 100
            print(f"  {season}: {count:,} rows ({percentage:.1f}%)")
    
    # Monthly breakdown
    print(f"\nMonthly breakdown:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_counts = df['Start_month'].value_counts().sort_index()
    for month_num, count in monthly_counts.items():
        percentage = (count / total_rows) * 100
        month_name = month_names[month_num - 1]
        print(f"  {month_name}: {count:,} rows ({percentage:.1f}%)")
    
    # Phase distribution
    print(f"\n🚨 PHASE DISTRIBUTION")
    phase_counts = df['Phase'].value_counts().sort_index()
    print(f"IPC Phase breakdown:")
    for phase, count in phase_counts.items():
        percentage = (count / total_rows) * 100
        phase_desc = {
            '1': 'Minimal/None',
            '2': 'Stressed', 
            '3': 'Crisis',
            '4': 'Emergency',
            '5': 'Catastrophe/Famine'
        }.get(str(phase), f'Phase {phase}')
        print(f"  Phase {phase} ({phase_desc}): {count:,} rows ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    src_dir = pathlib.Path(__file__).resolve().parent
    dataset_path = src_dir / "data" / "ipc_global_area_long_current_only.csv"
    
    analyze_ipc_dataset(dataset_path) 
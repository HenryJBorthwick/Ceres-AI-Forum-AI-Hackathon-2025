import pandas as pd
import numpy as np
import pathlib

def create_temporal_pairs(csv_file_path, output_file='train_predict.csv'):
    """
    Transform data into temporal pairs: year t features → year t+1 targets
    
    Parameters:
    csv_file_path: path to the input CSV file
    output_file: name of output CSV file with temporal pairs
    
    Returns:
    temporal_df: DataFrame with temporal pairs
    """
    
    # Load the data
    df = pd.read_csv(csv_file_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Years in data: {sorted(df['year'].unique())}")
    print(f"Areas in data: {df['Area_ID'].nunique()}")
    
    # Sort by Area_ID and year to ensure proper temporal ordering
    df_sorted = df.sort_values(['Area_ID', 'year'])
    
    # Identify feature and target columns
    band_cols = [col for col in df.columns if col.startswith('band_')]
    phase_cols = [col for col in df.columns if col.startswith('avg_pop_phase_')]
    
    print(f"Feature columns: {len(band_cols)} bands")
    print(f"Target columns: {len(phase_cols)} population phases")
    
    temporal_pairs_data = []
    
    # Group by Area_ID to handle temporal relationships within each area
    for area_id in df_sorted['Area_ID'].unique():
        area_data = df_sorted[df_sorted['Area_ID'] == area_id].copy()
        
        # For each consecutive year pair
        for i in range(len(area_data) - 1):
            current_year_data = area_data.iloc[i]
            next_year_data = area_data.iloc[i + 1]
            
            # Check if next year is actually t+1 (consecutive years)
            if next_year_data['year'] == current_year_data['year'] + 1:
                
                # Create row with metadata
                row_data = {
                    'Area_ID': area_id,
                    'year_t': current_year_data['year'],
                    'year_t_plus_1': next_year_data['year'],
                    'Country': current_year_data['Country'],
                    'Level1': current_year_data['Level1'],
                    'Area': current_year_data['Area']
                }
                
                # Add features from year t (with prefix)
                for col in band_cols:
                    row_data[f'feature_{col}'] = current_year_data[col]
                
                # Add targets from year t+1 (with prefix)
                for col in phase_cols:
                    row_data[f'target_{col}'] = next_year_data[col]
                
                temporal_pairs_data.append(row_data)
    
    # Create DataFrame from temporal pairs
    temporal_df = pd.DataFrame(temporal_pairs_data)
    
    if len(temporal_df) == 0:
        raise ValueError("No valid temporal pairs found. Need consecutive year data for same areas.")
    
    # Save to CSV
    output_path = pathlib.Path(output_file)
    temporal_df.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"Created {len(temporal_df)} temporal pairs")
    print(f"Saved to: {output_path.absolute()}")
    print(f"CSV structure:")
    print(f"  - Metadata: 6 columns (Area_ID, years, Country, Level1, Area)")
    print(f"  - Features: {len(band_cols)} columns (feature_band_0 to feature_band_63)")
    print(f"  - Targets: {len(phase_cols)} columns (target_avg_pop_phase_1 to target_avg_pop_phase_5)")
    print(f"  - Total columns: {len(temporal_df.columns)}")
    
    return temporal_df

if __name__ == "__main__":
    print("Temporal Data Transformer")
    print("=" * 50)
    print("Creates year t → year t+1 training pairs")
    print("Features from year t, targets from year t+1")
    print("=" * 50)
    
    # Transform the data
    try:
        temporal_df = create_temporal_pairs('src/ai/train.csv', 'src/ai/train_predict.csv')
        print(f"\n🎯 Transformation complete!")
        print(f"Your temporal dataset is ready in 'train_predict.csv'")
        
    except FileNotFoundError:
        print("❌ Error: 'train.csv' not found")
        print("Please make sure your input file is named 'train.csv'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
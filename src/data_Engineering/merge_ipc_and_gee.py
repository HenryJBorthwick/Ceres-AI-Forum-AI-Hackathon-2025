import pandas as pd

# Define file paths
satellite_embeddings_path = '/Users/charlie/Desktop/Hackathon/src/data/satellite_embeddings_level1_FAST.csv'
ipc_data_path = '/Users/charlie/Desktop/Hackathon/src/data/ipc_annual_dataset.csv'
output_path = '/Users/charlie/Desktop/Hackathon/src/data/merged_ipc_and_gee.csv'

# Load datasets
satellite_df = pd.read_csv(satellite_embeddings_path)
ipc_df = pd.read_csv(ipc_data_path)

# Ensure necessary columns are present
required_columns_satellite = ['Level1', 'year']
required_columns_ipc = ['Level 1', 'Start_year']

for col in required_columns_satellite:
    if col not in satellite_df.columns:
        raise KeyError(f"Column '{col}' not found in satellite embeddings dataset.")

for col in required_columns_ipc:
    if col not in ipc_df.columns:
        raise KeyError(f"Column '{col}' not found in IPC dataset.")

# Merge datasets on 'Level1' and 'Start_year'
merged_df = pd.merge(satellite_df, ipc_df.rename(columns={'Level 1': 'Level1', 'Start_year': 'year'}), on=['Level1', 'year'], how='inner')

# Save the merged dataset
merged_df.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}") 
import pandas as pd
import numpy as np
import os

path = os.path.join(os.path.dirname(__file__), 'dummy_data', 'ipc_predictions_2026.csv')
df = pd.read_csv(path)

# Convert total population to int
df['Total country population'] = df['Total country population'].astype(int)

# Calculate number of unique areas per country
df['unique_area'] = df['Level 1'].astype(str) + '_' + df['Area'].astype(str)
num_areas = df.groupby('Country')['unique_area'].nunique()

# Scale Numbers by dividing by num_areas for the country
df['Number'] = df.apply(lambda row: int(round(row['Number'] / num_areas[row['Country']])), axis=1)

# Round Percentages to 2 decimals
df['Percentage'] = df['Percentage'].round(2)

# Create a copy for current
current_df = df.copy()

# Create projection_df by copying and varying
projection_df = current_df.copy()
projection_df['Validity period'] = 'first projection'
projection_df['From'] = '2026-05-01'
projection_df['To'] = '2026-10-31'

# Vary percentages for projection
def vary_percentages(group):
    pcts = group['Percentage'].values + np.random.uniform(-0.05, 0.05, len(group))
    pcts = np.maximum(pcts, 0)
    total_pct = pcts.sum()
    if total_pct > 0:
        pcts = pcts / total_pct
    group['Percentage'] = pcts.round(2)
    area_pop = group['Number'].sum()
    group['Number'] = (pcts * area_pop).round(0).astype(int)
    return group

# Apply variation to groups of same area etc., but since Phase different, group by all except Phase
group_keys = ['Date of analysis', 'Country', 'Total country population', 'Level 1', 'Area', 'Validity period', 'From', 'To']
projection_df = projection_df.groupby(group_keys[:-3], as_index=False).apply(vary_percentages).reset_index(drop=True)

# Concat current and projection
df = pd.concat([current_df, projection_df], ignore_index=True)

# Now add 'all' and '3+' rows for all groups
groupby_keys = ['Date of analysis', 'Country', 'Total country population', 'Level 1', 'Area', 'Validity period', 'From', 'To']
grouped = df[df['Phase'].isin([1,2,3,4,5])].groupby(groupby_keys)

new_rows = []
for name, group in grouped:
    sum_number = group['Number'].sum()
    if sum_number == 0:
        continue
        
    number_3plus = group[group['Phase'].isin([3,4,5])]['Number'].sum()
    pct_3plus = group[group['Phase'].isin([3,4,5])]['Percentage'].sum()
    
    # 'all' row
    all_row = group.iloc[0].copy()
    all_row['Phase'] = 'all'
    all_row['Number'] = sum_number
    all_row['Percentage'] = 1.0
    new_rows.append(all_row)
    
    # '3+' row
    three_row = group.iloc[0].copy()
    three_row['Phase'] = '3+'
    three_row['Number'] = number_3plus
    three_row['Percentage'] = round(pct_3plus, 2)
    new_rows.append(three_row)

# Add new rows
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Sort
phase_order = {'all': 0, '3+': 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
df['phase_order'] = df['Phase'].map(phase_order)
df = df.sort_values(groupby_keys + ['phase_order']).drop('phase_order', axis=1)
df = df.drop('unique_area', axis=1)  # Clean up

# Save back
df.to_csv(path, index=False)
print("Predictions file updated successfully.")

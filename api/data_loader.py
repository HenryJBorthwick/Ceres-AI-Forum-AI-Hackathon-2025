import pandas as pd
import random
import os
from datetime import datetime

# Country code to name mapping
COUNTRY_MAPPING = {
    'AFG': 'Afghanistan',
    'AGO': 'Angola', 
    'BDI': 'Burundi',
    'BEN': 'Benin',
    'BFA': 'Burkina Faso',
    'BGD': 'Bangladesh',
    'CAF': 'Central African Republic',
    'CIV': 'Côte d\'Ivoire',
    'CMR': 'Cameroon',
    'COD': 'Democratic Republic of Congo',
    'CPV': 'Cape Verde',
    'DJI': 'Djibouti',
    'DOM': 'Dominican Republic',
    'ECU': 'Ecuador',
    'ETH': 'Ethiopia',
    'GHA': 'Ghana',
    'GIN': 'Guinea',
    'GMB': 'Gambia',
    'GNB': 'Guinea-Bissau',
    'GTM': 'Guatemala',
    'HND': 'Honduras',
    'HTI': 'Haiti',
    'KEN': 'Kenya',
    'LBN': 'Lebanon',
    'LBR': 'Liberia',
    'LSO': 'Lesotho',
    'MDG': 'Madagascar',
    'MLI': 'Mali',
    'MOZ': 'Mozambique',
    'MRT': 'Mauritania',
    'MWI': 'Malawi',
    'NAM': 'Namibia',
    'NER': 'Niger',
    'NGA': 'Nigeria',
    'PAK': 'Pakistan',
    'PSE': 'Palestine',
    'SDN': 'Sudan',
    'SEN': 'Senegal',
    'SLE': 'Sierra Leone',
    'SLV': 'El Salvador',
    'SOM': 'Somalia',
    'SSD': 'South Sudan',
    'SWZ': 'Eswatini',
    'TCD': 'Chad',
    'TGO': 'Togo',
    'TLS': 'Timor-Leste',
    'TZA': 'Tanzania',
    'UGA': 'Uganda',
    'YEM': 'Yemen',
    'ZAF': 'South Africa',
    'ZMB': 'Zambia',
    'ZWE': 'Zimbabwe'
}

CSV_PATH = '../data/ipc_global_area_long_current_only.csv'
PREDICTIONS_CSV_PATH = './dummy_data/ipc_predictions_2026.csv'

def load_data():
    df = pd.read_csv(CSV_PATH, comment='#')
    df['Year'] = pd.to_datetime(df['From']).dt.year
    df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
    return df

def load_prediction_data():
    """Load the 2026 prediction data from CSV file"""
    try:
        df = pd.read_csv(PREDICTIONS_CSV_PATH, comment='#')
        df['Year'] = pd.to_datetime(df['From']).dt.year
        df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
        df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
        df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The predicted data file '{PREDICTIONS_CSV_PATH}' could not be found.")
    except Exception as e:
        raise Exception(f"Error loading predicted data from '{PREDICTIONS_CSV_PATH}': {str(e)}")

def aggregate_data(df, country, level1=None, area=None):
    # Filter data
    filtered = df[df['Country'] == country]
    if level1:
        filtered = filtered[filtered['Level 1'] == level1]
    if area:
        filtered = filtered[filtered['Area'] == area]
    
    if filtered.empty:
        return []

    # Aggregate by year and phase
    grouped = filtered.groupby(['Year', 'Phase'])['Number'].sum().reset_index()
    years = sorted(grouped['Year'].unique())
    total_pop = filtered['Total country population'].max()  # Assume max is consistent

    # Get historic data per year
    historic_data = []
    for year in years:
        year_data = grouped[grouped['Year'] == year]
        phases = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        affected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for _, row in year_data.iterrows():
            phase = int(row['Phase'])
            num = row['Number']
            phases[phase] = (num / total_pop) * 100 if total_pop else 0
            affected[phase] = num
        historic_data.append(_format_json(country, level1 or area or '', total_pop, phases, affected, year, False))

    # Generate dummy historic if <5 years
    if len(historic_data) < 5:
        earliest = historic_data[0]
        for y in range(len(historic_data), 5):
            # Create varied phase percentages (±10% variation)
            prev_phases_pct = {}
            prev_phases_num = {}
            for phase in [1, 2, 3, 4, 5]:
                base_pct = earliest['ipc_phases'][f'phase_{phase}']['percent_affected']
                # Apply random variation ±10%
                varied_pct = base_pct * random.uniform(0.9, 1.1)
                prev_phases_pct[phase] = max(0, varied_pct)  # Ensure non-negative
                prev_phases_num[phase] = (prev_phases_pct[phase] / 100) * total_pop
            
            # Normalize to ensure total = 100%
            total_pct = sum(prev_phases_pct.values())
            if total_pct > 0:
                for phase in prev_phases_pct:
                    prev_phases_pct[phase] = (prev_phases_pct[phase] / total_pct) * 100
                    prev_phases_num[phase] = (prev_phases_pct[phase] / 100) * total_pop
            
            historic_data.insert(0, _format_json(country, level1 or area or '', total_pop, prev_phases_pct, prev_phases_num, earliest['year'] - (5 - y), False))

    # Generate predicted data from CSV file (will raise error if file not found)
    predicted = []
    prediction_df = load_prediction_data()
    
    # Filter prediction data the same way as historic data
    pred_filtered = prediction_df[prediction_df['Country'] == country]
    if level1:
        pred_filtered = pred_filtered[pred_filtered['Level 1'] == level1]
    if area:
        pred_filtered = pred_filtered[pred_filtered['Area'] == area]
    
    if not pred_filtered.empty:
        # Aggregate prediction data by year and phase
        pred_grouped = pred_filtered.groupby(['Year', 'Phase'])['Number'].sum().reset_index()
        pred_years = sorted(pred_grouped['Year'].unique())
        
        # Generate prediction data for each year found (should be 2026)
        for year in pred_years:
            pred_year_data = pred_grouped[pred_grouped['Year'] == year]
            pred_phases_pct = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            pred_phases_num = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            for _, row in pred_year_data.iterrows():
                phase = int(row['Phase'])
                num = row['Number']
                pred_phases_pct[phase] = (num / total_pop) * 100 if total_pop else 0
                pred_phases_num[phase] = num
            
            predicted.append(_format_json(country, level1 or area or '', total_pop, pred_phases_pct, pred_phases_num, year, True))

    return historic_data + predicted

def _format_json(country, area, total_pop, phases_pct, phases_num, year, is_predicted):
    ipc_phases = {}
    descriptions = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency", 5: "Catastrophe"}
    for p in [1,2,3,4,5]:
        ipc_phases[f'phase_{p}'] = {
            "description": descriptions[p],
            "affected_population": int(round(phases_num.get(p, 0))),  # Convert to Python int
            "percent_affected": float(round(phases_pct.get(p, 0), 1)),  # Convert to Python float
            # Add chart-ready data with prediction flag
            "chart_value": float(round(phases_pct.get(p, 0), 1)),
            "is_predicted": bool(is_predicted)
        }
    return {
        "location": {"country": str(country), "area": str(area), "total_population": int(total_pop)},
        "ipc_phases": ipc_phases,
        "summary": {"total_affected": int(total_pop), "total_percentage": 100.0},
        "year": int(year),  # Convert to Python int
        "is_predicted": bool(is_predicted),  # Convert to Python bool
        # Add chart-ready format
        "chart_data": {
            "year": int(year),
            "phase1": float(round(phases_pct.get(1, 0), 1)),
            "phase2": float(round(phases_pct.get(2, 0), 1)),
            "phase3": float(round(phases_pct.get(3, 0), 1)),
            "phase4": float(round(phases_pct.get(4, 0), 1)),
            "phase5": float(round(phases_pct.get(5, 0), 1)),
            "is_predicted": bool(is_predicted)
        }
    }

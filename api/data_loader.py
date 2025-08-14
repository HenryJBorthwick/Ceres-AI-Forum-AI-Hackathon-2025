import pandas as pd
import random
import os
from datetime import datetime
import numpy as np

from model.inference_example import predict_by_level1_or_area, model_available

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

# Import the comprehensive region mapping
from comprehensive_region_mapping import get_satellite_region_name

CSV_PATH = '../data/ipc_global_area_long_current_only.csv'

def load_data():
    df = pd.read_csv(CSV_PATH, comment='#')
    df['Year'] = pd.to_datetime(df['From']).dt.year
    df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
    return df

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
    
    # Get historic data per year (up to 2024)
    historic_data = []
    historic_total_pops = []
    for year in years:
        if year >= 2025:
            continue
        year_data = grouped[grouped['Year'] == year]
        total_pop_year = year_data['Number'].sum()
        if total_pop_year == 0:
            continue
        phases = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        affected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for _, row in year_data.iterrows():
            phase = int(row['Phase'])
            num = row['Number']
            phases[phase] = (num / total_pop_year) * 100 if total_pop_year else 0
            affected[phase] = num
        historic_data.append(_format_json(country, level1 or area or '', total_pop_year, phases, affected, year, False))
        historic_total_pops.append(total_pop_year)
    
    # Calculate average total pop from historical years
    average_total_pop = np.mean(historic_total_pops) if historic_total_pops else 0
    
    # Generate AI predicted data for 2025 (only if we have historical data for population baseline)
    predicted = []
    if model_available and average_total_pop > 0:
        pred_years = [2025]
        for pred_year in pred_years:
            try:
                if level1:
                    # Use mapped region name for satellite embeddings
                    satellite_region = get_satellite_region_name(level1)
                    pred = predict_by_level1_or_area(level1=satellite_region, year=pred_year)
                elif area:
                    pred = predict_by_level1_or_area(area_id=f"{country}_{area.replace(' ', '_')}", year=pred_year)
                else:
                    pred = predict_by_level1_or_area(country=country, year=pred_year)
                
                phases_pct = {int(k.split()[-1]): v for k, v in pred['phase_percentages'].items()}
                total_pop = average_total_pop
                phases_num = {p: (phases_pct[p] / 100.0) * total_pop for p in range(1,6)}
                predicted.append(_format_json(country, level1 or area or '', total_pop, phases_pct, phases_num, pred_year, True))
            except ValueError as e:
                print(f"Warning: Could not generate prediction for {country} {level1 or area or ''} in {pred_year}: {str(e)}")
    
    # Combine historical and predicted data
    all_data = historic_data + predicted
    
    # Sort chronologically by year
    all_data_sorted = sorted(all_data, key=lambda x: x['year'])
    
    return all_data_sorted

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
        "location": {"country": str(country), "area": str(area), "total_population": int(round(total_pop))},
        "ipc_phases": ipc_phases,
        "summary": {"total_affected": int(round(total_pop)), "total_percentage": 100.0},
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

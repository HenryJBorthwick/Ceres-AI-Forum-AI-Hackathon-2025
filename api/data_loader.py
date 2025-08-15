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

# Countries that have satellite embeddings available for inference
COUNTRIES_WITH_EMBEDDINGS = {
    'AFG', 'AGO', 'BEN', 'BFA', 'BGD', 'CAF', 'CMR', 'COD', 'DJI', 'ETH',
    'GHA', 'GIN', 'HND', 'HTI', 'KEN', 'LBN', 'MDG', 'MLI', 'MOZ', 'MRT',
    'NER', 'NGA', 'SDN', 'SEN', 'SLE', 'SOM', 'SSD', 'TCD', 'TGO', 'TZA',
    'UGA', 'YEM', 'ZAF', 'ZMB', 'ZWE'
}

CSV_PATH = '../data/ipc_global_area_long_current_only.csv'

# Global data validator instance
_data_validator = None

def get_data_validator():
    """Get or create the global data validator instance"""
    global _data_validator
    if _data_validator is None:
        from data_validator import DataValidator
        ipc_data = load_data()
        satellite_embeddings = load_satellite_embeddings_data()
        _data_validator = DataValidator(ipc_data, satellite_embeddings)
    return _data_validator

def load_satellite_embeddings_data():
    """Load satellite embeddings data for validation"""
    try:
        sat_data_path = os.path.join(os.path.dirname(__file__), 'model', 'sat_embeddings', 'sat_embeddings_recent.csv')
        if os.path.exists(sat_data_path):
            return pd.read_csv(sat_data_path)
        else:
            print(f"Warning: Satellite embeddings file not found at {sat_data_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load satellite embeddings: {e}")
        return pd.DataFrame()

def load_data():
    df = pd.read_csv(CSV_PATH, comment='#')
    df['Year'] = pd.to_datetime(df['From']).dt.year
    df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
    return df

def get_valid_countries():
    """Get countries that have both IPC data and satellite embeddings"""
    validator = get_data_validator()
    return validator.get_valid_countries()

def get_valid_regions_for_country(country_code):
    """Get valid regions for a country that exist in both datasets"""
    validator = get_data_validator()
    return validator.get_valid_regions_for_country(country_code)

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
                # Only generate predictions for countries with satellite embeddings
                if country not in COUNTRIES_WITH_EMBEDDINGS:
                    continue
                
                # Validate that the region can be used for inference
                validator = get_data_validator()
                if level1 and not validator.validate_region_for_inference(country, level1):
                    print(f"Warning: Region {level1} in {country} cannot be used for inference")
                    continue
                
                if level1:
                    # Use mapped region name for satellite embeddings
                    satellite_region = validator.get_satellite_region_name_for_inference(country, level1)
                    if satellite_region:
                        pred = predict_by_level1_or_area(level1=satellite_region, year=pred_year)
                    else:
                        print(f"Warning: Could not map region {level1} to satellite data for {country}")
                        continue
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

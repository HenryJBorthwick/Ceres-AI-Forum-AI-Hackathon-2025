import pandas as pd
import random
import os
from datetime import datetime

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

    # Generate predicted (2026, vary crisis phases +5%)
    last_historic = historic_data[-1]
    pred_phases = {}
    shift = 5  # % increase for phase 3-5
    for phase in [1,2,3,4,5]:
        last_pct = last_historic['ipc_phases'][f'phase_{phase}']['percent_affected']
        if phase >= 3:
            # Increase crisis phases by 5%
            pred_phases[phase] = max(0, last_pct + shift)
        else:
            # Reduce lower phases, but ensure they don't go below 0
            pred_phases[phase] = max(0, last_pct - (shift / 2))
    
    # Ensure total doesn't exceed 100% and redistribute if needed
    total_pct = sum(pred_phases.values())
    if total_pct > 100:
        # Scale down proportionally to fit within 100%
        pred_phases = {k: v * (100 / total_pct) for k, v in pred_phases.items()}
    elif total_pct < 100:
        # Add remaining percentage to phase 1 (safest assumption)
        pred_phases[1] += (100 - total_pct)
    
    pred_affected = {k: (v / 100) * total_pop for k, v in pred_phases.items()}
    predicted = [_format_json(country, level1 or area or '', total_pop, pred_phases, pred_affected, 2026, True)]

    return historic_data + predicted

def _format_json(country, area, total_pop, phases_pct, phases_num, year, is_predicted):
    ipc_phases = {}
    descriptions = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency", 5: "Catastrophe"}
    for p in [1,2,3,4,5]:
        ipc_phases[f'phase_{p}'] = {
            "description": descriptions[p],
            "affected_population": int(round(phases_num.get(p, 0))),  # Convert to Python int
            "percent_affected": float(round(phases_pct.get(p, 0), 1))  # Convert to Python float
        }
    return {
        "location": {"country": str(country), "area": str(area), "total_population": int(total_pop)},
        "ipc_phases": ipc_phases,
        "summary": {"total_affected": int(total_pop), "total_percentage": 100.0},
        "year": int(year),  # Convert to Python int
        "is_predicted": bool(is_predicted)  # Convert to Python bool
    }

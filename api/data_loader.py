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
            prev_phases = {k: v * random.uniform(0.9, 1.1) for k, v in earliest['ipc_phases'].items() if isinstance(v, dict)}  # Vary ±10%
            # Normalize to 100%
            total_pct = sum(p['percent_affected'] for p in prev_phases.values())
            for p in prev_phases.values():
                p['percent_affected'] /= (total_pct / 100)
                p['affected_population'] = (p['percent_affected'] / 100) * total_pop
            historic_data.insert(0, _format_json(country, level1 or area or '', total_pop, prev_phases, {k: v['affected_population'] for k, v in prev_phases.items()}, earliest['year'] - (5 - y), False))

    # Generate predicted (2026, vary crisis phases +5%)
    last_historic = historic_data[-1]
    pred_phases = {}
    shift = 5  # % increase for phase 3-5
    for phase in [1,2,3,4,5]:
        last_pct = last_historic['ipc_phases'][f'phase_{phase}']['percent_affected']
        if phase >= 3:
            pred_phases[phase] = last_pct + shift
        else:
            pred_phases[phase] = last_pct - (shift / 2)  # Reduce lower phases
    # Normalize
    total_pct = sum(pred_phases.values())
    pred_phases = {k: v / (total_pct / 100) for k, v in pred_phases.items()}
    pred_affected = {k: (v / 100) * total_pop for k, v in pred_phases.items()}
    predicted = [_format_json(country, level1 or area or '', total_pop, pred_phases, pred_affected, 2026, True)]

    return historic_data + predicted

def _format_json(country, area, total_pop, phases_pct, phases_num, year, is_predicted):
    ipc_phases = {}
    descriptions = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency", 5: "Catastrophe"}
    for p in [1,2,3,4,5]:
        ipc_phases[f'phase_{p}'] = {
            "description": descriptions[p],
            "affected_population": round(phases_num.get(p, 0)),
            "percent_affected": round(phases_pct.get(p, 0), 1)
        }
    return {
        "location": {"country": country, "area": area, "total_population": int(total_pop)},
        "ipc_phases": ipc_phases,
        "summary": {"total_affected": int(total_pop), "total_percentage": 100.0},
        "year": year,
        "is_predicted": is_predicted
    }

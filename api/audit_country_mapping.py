#!/usr/bin/env python3
"""
Country Mapping Audit Script
Audits the mapping between countries in the IPC dataset and the embeddings dataset
to ensure inference always works when selecting a country.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import COUNTRY_MAPPING, load_data
from model.inference_example import load_satellite_embeddings, model_available

def audit_country_mapping():
    """Audit the mapping between IPC countries and satellite embeddings"""
    
    print("🔍 AUDITING COUNTRY MAPPING BETWEEN IPC DATASET AND EMBEDDINGS DATASET")
    print("=" * 80)
    
    # Check if model is available
    if not model_available:
        print("❌ ML model not available - cannot perform audit")
        return
    
    # Load IPC data
    print("\n📊 Loading IPC dataset...")
    try:
        ipc_df = load_data()
        print(f"✅ IPC dataset loaded: {len(ipc_df)} rows")
    except Exception as e:
        print(f"❌ Failed to load IPC dataset: {e}")
        return
    
    # Load satellite embeddings
    print("\n🛰️ Loading satellite embeddings...")
    try:
        sat_df = load_satellite_embeddings()
        if sat_df is not None:
            print(f"✅ Satellite embeddings loaded: {len(sat_df)} rows")
        else:
            print("❌ Failed to load satellite embeddings")
            return
    except Exception as e:
        print(f"❌ Failed to load satellite embeddings: {e}")
        return
    
    # Get unique countries from IPC dataset
    ipc_countries = set(ipc_df['Country'].unique())
    print(f"\n🌍 IPC Dataset Countries ({len(ipc_countries)}):")
    for country in sorted(ipc_countries):
        print(f"  - {country}")
    
    # Get unique countries from satellite embeddings
    sat_countries = set()
    for area_id in sat_df['Area_ID'].dropna().unique():
        if '_' in area_id:
            country_code = area_id.split('_')[0]
            sat_countries.add(country_code)
    
    print(f"\n🛰️ Satellite Embeddings Countries ({len(sat_countries)}):")
    for country in sorted(sat_countries):
        print(f"  - {country}")
    
    # Analyze mapping
    print(f"\n🔗 MAPPING ANALYSIS:")
    print("=" * 50)
    
    # Countries in IPC but not in embeddings
    missing_in_embeddings = ipc_countries - sat_countries
    if missing_in_embeddings:
        print(f"\n❌ COUNTRIES IN IPC BUT MISSING FROM EMBEDDINGS ({len(missing_in_embeddings)}):")
        for country in sorted(missing_in_embeddings):
            country_name = COUNTRY_MAPPING.get(country, country)
            print(f"  - {country} ({country_name}) - NO INFERENCE POSSIBLE")
    
    # Countries in embeddings but not in IPC
    missing_in_ipc = sat_countries - ipc_countries
    if missing_in_ipc:
        print(f"\n⚠️ COUNTRIES IN EMBEDDINGS BUT NOT IN IPC ({len(missing_in_ipc)}):")
        for country in sorted(missing_in_ipc):
            print(f"  - {country} - Has satellite data but no IPC history")
    
    # Countries with both datasets
    common_countries = ipc_countries & sat_countries
    if common_countries:
        print(f"\n✅ COUNTRIES WITH BOTH DATASETS ({len(common_countries)}):")
        for country in sorted(common_countries):
            country_name = COUNTRY_MAPPING.get(country, country)
            # Count regions for this country
            country_regions = sat_df[sat_df['Area_ID'].str.startswith(f"{country}_", na=False)]
            region_count = len(country_regions['Area_ID'].unique())
            print(f"  - {country} ({country_name}) - {region_count} regions available for inference")
    
    # Test inference for each common country
    print(f"\n🧪 TESTING INFERENCE FOR COMMON COUNTRIES:")
    print("=" * 50)
    
    from model.inference_example import predict_by_level1_or_area
    
    inference_results = {}
    for country in sorted(common_countries):
        try:
            print(f"\n🔮 Testing inference for {country}...")
            result = predict_by_level1_or_area(country=country, year=2025)
            
            if result:
                inference_results[country] = "SUCCESS"
                print(f"  ✅ SUCCESS: {result['dominant_phase']} ({result['confidence']}% confidence)")
                print(f"     Samples used: {result.get('samples_used', 'N/A')}")
                print(f"     Embedding year: {result.get('used_embedding_year', 'N/A')}")
            else:
                inference_results[country] = "FAILED"
                print(f"  ❌ FAILED: No result returned")
                
        except Exception as e:
            inference_results[country] = f"ERROR: {str(e)}"
            print(f"  ❌ ERROR: {str(e)}")
    
    # Summary
    print(f"\n📋 AUDIT SUMMARY:")
    print("=" * 50)
    print(f"Total IPC countries: {len(ipc_countries)}")
    print(f"Total embedding countries: {len(sat_countries)}")
    print(f"Common countries: {len(common_countries)}")
    print(f"Missing from embeddings: {len(missing_in_embeddings)}")
    print(f"Missing from IPC: {len(missing_in_ipc)}")
    
    successful_inference = sum(1 for result in inference_results.values() if result == "SUCCESS")
    print(f"Successful inference tests: {successful_inference}/{len(common_countries)}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    if missing_in_embeddings:
        print(f"1. Add satellite embeddings for {len(missing_in_embeddings)} missing countries:")
        for country in sorted(missing_in_embeddings):
            country_name = COUNTRY_MAPPING.get(country, country)
            print(f"   - {country} ({country_name})")
    
    if missing_in_ipc:
        print(f"2. Consider adding IPC data for {len(missing_in_ipc)} countries with satellite data")
    
    if successful_inference < len(common_countries):
        print(f"3. Fix inference errors for countries with both datasets")
    
    print(f"4. Ensure region mapping is complete for all countries")
    
    return {
        'ipc_countries': ipc_countries,
        'sat_countries': sat_countries,
        'common_countries': common_countries,
        'missing_in_embeddings': missing_in_embeddings,
        'missing_in_ipc': missing_in_ipc,
        'inference_results': inference_results
    }

if __name__ == "__main__":
    try:
        results = audit_country_mapping()
        print(f"\n🎯 Audit completed successfully!")
    except Exception as e:
        print(f"\n💥 Audit failed with error: {e}")
        import traceback
        traceback.print_exc()

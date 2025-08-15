#!/usr/bin/env python3
"""
Analyze mapping gaps between IPC data and satellite embeddings
This script helps identify regions that need manual mapping
"""

import sys
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def analyze_mapping_gaps():
    """Analyze gaps in region mapping between IPC and satellite data"""
    print("🔍 Analyzing Mapping Gaps")
    print("=" * 50)
    
    try:
        # Import modules
        from data_loader import load_data, load_satellite_embeddings_data
        
        # Load data
        print("📊 Loading data...")
        ipc_data = load_data()
        satellite_data = load_satellite_embeddings_data()
        
        print(f"✓ IPC data: {len(ipc_data)} rows")
        print(f"✓ Satellite data: {len(satellite_data)} rows")
        
        # Get unique countries from both datasets
        ipc_countries = set(ipc_data['Country'].unique())
        sat_countries = set(satellite_data['Country'].unique())
        
        print(f"\n🌍 Countries in IPC data: {len(ipc_countries)}")
        print(f"🌍 Countries in satellite data: {len(sat_countries)}")
        
        # Find common countries
        common_countries = ipc_countries.intersection(sat_countries)
        print(f"🌍 Common countries: {len(common_countries)}")
        
        # Analyze each common country
        mapping_analysis = {}
        
        for country_code in sorted(common_countries):
            print(f"\n📍 Analyzing {country_code}...")
            
            # Get regions from both datasets
            ipc_regions = set(ipc_data[ipc_data['Country'] == country_code]['Level 1'].dropna().unique())
            sat_regions = set(satellite_data[satellite_data['Country'] == country_code]['Level1'].unique())
            
            # Find exact matches
            exact_matches = ipc_regions.intersection(sat_regions)
            
            # Find regions that need mapping
            ipc_only = ipc_regions - sat_regions
            sat_only = sat_regions - ipc_regions
            
            # Store analysis
            mapping_analysis[country_code] = {
                'ipc_regions': ipc_regions,
                'sat_regions': sat_regions,
                'exact_matches': exact_matches,
                'ipc_only': ipc_only,
                'sat_only': sat_only,
                'total_ipc': len(ipc_regions),
                'total_sat': len(sat_regions),
                'exact_matches_count': len(exact_matches),
                'ipc_only_count': len(ipc_only),
                'sat_only_count': len(sat_only)
            }
            
            print(f"  IPC regions: {len(ipc_regions)}")
            print(f"  Satellite regions: {len(sat_regions)}")
            print(f"  Exact matches: {len(exact_matches)}")
            print(f"  IPC only: {len(ipc_only)}")
            print(f"  Satellite only: {len(sat_only)}")
        
        # Generate comprehensive report
        print("\n📋 Generating comprehensive mapping report...")
        generate_mapping_report(mapping_analysis, ipc_data, satellite_data)
        
        # Suggest mappings for IPC-only regions
        print("\n💡 Suggesting potential mappings...")
        suggest_potential_mappings(mapping_analysis)
        
        print("\n✅ Analysis completed!")
        return mapping_analysis
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_mapping_report(mapping_analysis, ipc_data, satellite_data):
    """Generate a comprehensive mapping report"""
    
    report_lines = []
    report_lines.append("COMPREHENSIVE MAPPING ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary statistics
    total_countries = len(mapping_analysis)
    total_exact_matches = sum(data['exact_matches_count'] for data in mapping_analysis.values())
    total_ipc_only = sum(data['ipc_only_count'] for data in mapping_analysis.values())
    total_sat_only = sum(data['sat_only_count'] for data in mapping_analysis.values())
    
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Total countries analyzed: {total_countries}")
    report_lines.append(f"Total exact matches: {total_exact_matches}")
    report_lines.append(f"Total IPC-only regions: {total_ipc_only}")
    report_lines.append(f"Total satellite-only regions: {total_sat_only}")
    report_lines.append("")
    
    # Country-by-country breakdown
    report_lines.append("COUNTRY-BY-COUNTRY BREAKDOWN")
    report_lines.append("=" * 60)
    
    for country_code, data in mapping_analysis.items():
        country_name = get_country_name(country_code)
        report_lines.append(f"\n{country_name} ({country_code})")
        report_lines.append("-" * (len(country_name) + len(country_code) + 3))
        
        report_lines.append(f"IPC regions: {data['total_ipc']}")
        report_lines.append(f"Satellite regions: {data['total_sat']}")
        report_lines.append(f"Exact matches: {data['exact_matches_count']}")
        report_lines.append(f"IPC only: {data['ipc_only_count']}")
        report_lines.append(f"Satellite only: {data['sat_only_count']}")
        
        # Show exact matches
        if data['exact_matches']:
            report_lines.append("  Exact matches:")
            for region in sorted(data['exact_matches']):
                report_lines.append(f"    ✓ {region}")
        
        # Show IPC-only regions (need mapping)
        if data['ipc_only']:
            report_lines.append("  IPC-only regions (need mapping):")
            for region in sorted(data['ipc_only']):
                report_lines.append(f"    ❌ {region}")
        
        # Show satellite-only regions
        if data['sat_only']:
            report_lines.append("  Satellite-only regions:")
            for region in sorted(data['sat_only']):
                report_lines.append(f"    🛰️  {region}")
    
    # Priority regions for mapping
    report_lines.append("\n" + "=" * 60)
    report_lines.append("PRIORITY REGIONS FOR MAPPING")
    report_lines.append("=" * 60)
    
    # Sort countries by number of unmapped regions
    priority_countries = sorted(
        mapping_analysis.items(), 
        key=lambda x: x[1]['ipc_only_count'], 
        reverse=True
    )
    
    for country_code, data in priority_countries:
        if data['ipc_only_count'] > 0:
            country_name = get_country_name(country_code)
            report_lines.append(f"\n{country_name} ({country_code}) - {data['ipc_only_count']} unmapped regions:")
            for region in sorted(data['ipc_only']):
                report_lines.append(f"  - {region}")
    
    # Write report to file
    report_path = "mapping_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Comprehensive report written to {report_path}")

def suggest_potential_mappings(mapping_analysis):
    """Suggest potential mappings based on similarity"""
    
    print("\n🔍 Analyzing potential mappings...")
    
    # Common naming patterns
    common_patterns = {
        'province': ['Province', 'Province', 'Prov'],
        'north': ['North', 'Northern', 'Nord', 'N'],
        'south': ['South', 'Southern', 'Sud', 'S'],
        'east': ['East', 'Eastern', 'Est', 'E'],
        'west': ['West', 'Western', 'Ouest', 'W'],
        'central': ['Central', 'Centre', 'Centro'],
        'north_west': ['North-West', 'North West', 'Nord-Ouest', 'Northwest'],
        'south_west': ['South-West', 'South West', 'Sud-Ouest', 'Southwest'],
        'north_east': ['North-East', 'North East', 'Nord-Est', 'Northeast'],
        'south_east': ['South-East', 'South East', 'Sud-Est', 'Southeast']
    }
    
    potential_mappings = []
    
    for country_code, data in mapping_analysis.items():
        if data['ipc_only_count'] > 0:
            country_name = get_country_name(country_code)
            
            for ipc_region in data['ipc_only']:
                # Look for similar satellite regions
                for sat_region in data['sat_regions']:
                    similarity_score = calculate_similarity(ipc_region, sat_region)
                    
                    if similarity_score > 0.7:  # High similarity threshold
                        potential_mappings.append({
                            'country_code': country_code,
                            'country_name': country_name,
                            'ipc_region': ipc_region,
                            'satellite_region': sat_region,
                            'similarity_score': similarity_score
                        })
    
    # Sort by similarity score
    potential_mappings.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    if potential_mappings:
        print(f"\n💡 Found {len(potential_mappings)} potential mappings:")
        print("\nHigh-confidence suggestions:")
        
        for i, mapping in enumerate(potential_mappings[:10]):  # Show top 10
            print(f"  {i+1}. {mapping['country_name']} - {mapping['ipc_region']} -> {mapping['satellite_region']} (score: {mapping['similarity_score']:.2f})")
        
        # Save suggestions to file
        suggestions_path = "potential_mappings.txt"
        with open(suggestions_path, 'w', encoding='utf-8') as f:
            f.write("POTENTIAL REGION MAPPINGS\n")
            f.write("=" * 40 + "\n\n")
            
            for mapping in potential_mappings:
                f.write(f"{mapping['country_name']} ({mapping['country_code']})\n")
                f.write(f"  {mapping['ipc_region']} -> {mapping['satellite_region']}\n")
                f.write(f"  Similarity: {mapping['similarity_score']:.2f}\n\n")
        
        print(f"\n✓ Potential mappings saved to {suggestions_path}")
    else:
        print("  No high-confidence mappings found")

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings"""
    if not str1 or not str2:
        return 0.0
    
    # Convert to lowercase and remove common words
    str1_clean = str1.lower().replace('province', '').replace('region', '').strip()
    str2_clean = str2.lower().replace('province', '').replace('region', '').strip()
    
    # Simple similarity based on common characters
    if str1_clean == str2_clean:
        return 1.0
    
    # Check if one is contained in the other
    if str1_clean in str2_clean or str2_clean in str1_clean:
        return 0.8
    
    # Check for common words
    words1 = set(str1_clean.split())
    words2 = set(str2_clean.split())
    
    if words1 and words2:
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        if total_words:
            return len(common_words) / len(total_words)
    
    return 0.0

def get_country_name(country_code):
    """Get country name from country code"""
    country_names = {
        'AFG': 'Afghanistan', 'AGO': 'Angola', 'BEN': 'Benin', 'BFA': 'Burkina Faso',
        'BGD': 'Bangladesh', 'CAF': 'Central African Republic', 'CMR': 'Cameroon',
        'COD': 'Democratic Republic of Congo', 'DJI': 'Djibouti', 'ETH': 'Ethiopia',
        'GHA': 'Ghana', 'GIN': 'Guinea', 'HND': 'Honduras', 'HTI': 'Haiti',
        'KEN': 'Kenya', 'LBN': 'Lebanon', 'MDG': 'Madagascar', 'MLI': 'Mali',
        'MOZ': 'Mozambique', 'MRT': 'Mauritania', 'NER': 'Niger', 'NGA': 'Nigeria',
        'SDN': 'Sudan', 'SEN': 'Senegal', 'SLE': 'Sierra Leone', 'SOM': 'Somalia',
        'SSD': 'South Sudan', 'TCD': 'Chad', 'TGO': 'Togo', 'TZA': 'Tanzania',
        'UGA': 'Uganda', 'YEM': 'Yemen', 'ZAF': 'South Africa', 'ZMB': 'Zambia',
        'ZWE': 'Zimbabwe'
    }
    return country_names.get(country_code, country_code)

def main():
    """Main function"""
    print("🚀 Starting Mapping Gap Analysis")
    print("=" * 60)
    
    analysis_result = analyze_mapping_gaps()
    
    if analysis_result:
        print("\n🎉 Analysis completed successfully!")
        print("Check the generated reports for detailed information:")
        print("  - mapping_analysis_report.txt: Comprehensive analysis")
        print("  - potential_mappings.txt: Suggested mappings")
        return 0
    else:
        print("\n❌ Analysis failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

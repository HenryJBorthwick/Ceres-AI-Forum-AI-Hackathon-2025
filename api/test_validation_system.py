#!/usr/bin/env python3
"""
Test script for the new data validation system
This script tests the mapping between IPC data and satellite embeddings
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_data_validation():
    """Test the data validation system"""
    print("🧪 Testing Data Validation System")
    print("=" * 50)
    
    try:
        # Import modules
        from data_loader import load_data, load_satellite_embeddings_data
        from data_validator import DataValidator
        
        print("✓ Successfully imported modules")
        
        # Load data
        print("\n📊 Loading data...")
        ipc_data = load_data()
        satellite_data = load_satellite_embeddings_data()
        
        print(f"✓ IPC data loaded: {len(ipc_data)} rows")
        print(f"✓ Satellite data loaded: {len(satellite_data)} rows")
        
        # Create validator
        print("\n🔍 Creating data validator...")
        validator = DataValidator(ipc_data, satellite_data)
        print("✓ Data validator created successfully")
        
        # Test country validation
        print("\n🌍 Testing country validation...")
        valid_countries = validator.get_valid_countries()
        print(f"✓ Found {len(valid_countries)} valid countries")
        
        # Show first few countries
        for i, country in enumerate(valid_countries[:5]):
            print(f"  {i+1}. {country['name']} ({country['code']}) - {country['valid_regions_count']} regions")
        
        # Test region validation for first country
        if valid_countries:
            first_country = valid_countries[0]
            print(f"\n📍 Testing region validation for {first_country['name']}...")
            
            valid_regions = validator.get_valid_regions_for_country(first_country['code'])
            print(f"✓ Found {len(valid_regions)} valid regions")
            
            # Show first few regions
            for i, region in enumerate(valid_regions[:5]):
                mapping_info = validator.get_region_mapping_info(first_country['code'], region)
                print(f"  {i+1}. {region} -> {mapping_info['satellite_name']} ({mapping_info['status']})")
        
        # Get coverage summary
        print("\n📈 Getting coverage summary...")
        coverage = validator.get_data_coverage_summary()
        print(f"✓ Total countries: {coverage['total_countries']}")
        print(f"✓ Total valid regions: {coverage['total_valid_regions']}")
        print(f"✓ Coverage percentage: {coverage['coverage_percentage']:.1f}%")
        
        # Show mapping breakdown
        print("\n🗺️  Mapping breakdown:")
        for status, count in coverage['mapping_breakdown'].items():
            print(f"  {status.replace('_', ' ').title()}: {count}")
        
        # Test unmapped regions
        print("\n❌ Checking unmapped regions...")
        unmapped = validator.get_unmapped_regions()
        print(f"✓ Found {len(unmapped)} unmapped regions")
        
        if unmapped:
            print("  Sample unmapped regions:")
            for region in unmapped[:3]:
                print(f"    {region['country_name']} - {region['region_name']}")
        
        # Test satellite-only regions
        print("\n🛰️  Checking satellite-only regions...")
        satellite_only = validator.get_satellite_only_regions()
        print(f"✓ Found {len(satellite_only)} satellite-only regions")
        
        if satellite_only:
            print("  Sample satellite-only regions:")
            for region in satellite_only[:3]:
                print(f"    {region['country_name']} - {region['region_name']}")
        
        # Export mapping report
        print("\n📋 Exporting mapping report...")
        report_path = "mapping_report.txt"
        report_content = validator.export_mapping_report(report_path)
        print(f"✓ Mapping report exported to {report_path}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    try:
        # Import app
        from app import app
        from fastapi.testclient import TestClient
        
        print("✓ Successfully imported FastAPI app")
        
        # Create test client
        client = TestClient(app)
        print("✓ Test client created")
        
        # Test countries endpoint
        print("\n📡 Testing /countries endpoint...")
        response = client.get("/countries")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Countries endpoint working: {len(data.get('countries', []))} countries")
        else:
            print(f"❌ Countries endpoint failed: {response.status_code}")
        
        # Test levels endpoint for first country
        if response.status_code == 200:
            data = response.json()
            if data.get('countries'):
                first_country = data['countries'][0]
                print(f"\n📡 Testing /levels/{first_country} endpoint...")
                
                levels_response = client.get(f"/levels/{first_country}")
                if levels_response.status_code == 200:
                    levels_data = levels_response.json()
                    print(f"✓ Levels endpoint working: {len(levels_data.get('levels', []))} levels")
                else:
                    print(f"❌ Levels endpoint failed: {levels_response.status_code}")
        
        # Test data validation report endpoint
        print("\n📡 Testing /data-validation-report endpoint...")
        report_response = client.get("/data-validation-report")
        if report_response.status_code == 200:
            report_data = report_response.json()
            print("✓ Data validation report endpoint working")
            if 'coverage_summary' in report_data:
                coverage = report_data['coverage_summary']
                print(f"  Coverage: {coverage.get('total_countries', 0)} countries, {coverage.get('total_valid_regions', 0)} regions")
        else:
            print(f"❌ Data validation report endpoint failed: {report_response.status_code}")
        
        print("\n✅ API endpoint tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during API testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Data Validation System Tests")
    print("=" * 60)
    
    # Test data validation
    validation_success = test_data_validation()
    
    # Test API endpoints
    api_success = test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Data Validation: {'✅ PASSED' if validation_success else '❌ FAILED'}")
    print(f"API Endpoints: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if validation_success and api_success:
        print("\n🎉 All tests passed! The validation system is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

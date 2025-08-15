import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from comprehensive_region_mapping import (
    get_satellite_region_name, 
    get_ipc_region_name,
    get_available_regions_for_country,
    get_region_mapping_status
)

class DataValidator:
    """
    Validates and maps data between IPC datasets and satellite embeddings
    to ensure only valid regions are displayed in the graph configurator
    """
    
    def __init__(self, ipc_data: pd.DataFrame, satellite_embeddings: pd.DataFrame):
        self.ipc_data = ipc_data
        self.satellite_embeddings = satellite_embeddings
        self._build_validation_cache()
    
    def _build_validation_cache(self):
        """Build cache of valid regions and mappings for performance"""
        self._valid_regions_cache = {}
        self._mapping_status_cache = {}
        
        # Get unique countries from both datasets
        ipc_countries = set(self.ipc_data['Country'].unique())
        sat_countries = set(self.satellite_embeddings['Country'].unique())
        
        # Only process countries that exist in both datasets
        common_countries = ipc_countries.intersection(sat_countries)
        
        for country_code in common_countries:
            self._valid_regions_cache[country_code] = get_available_regions_for_country(
                country_code, self.ipc_data, self.satellite_embeddings
            )
            self._mapping_status_cache[country_code] = get_region_mapping_status(
                country_code, self.ipc_data, self.satellite_embeddings
            )
    
    def get_valid_countries(self) -> List[Dict]:
        """
        Get list of countries that have both IPC data and satellite embeddings
        
        Returns:
            List of country dictionaries with validation info
        """
        valid_countries = []
        
        for country_code in self._valid_regions_cache.keys():
            if len(self._valid_regions_cache[country_code]) > 0:
                valid_countries.append({
                    'code': country_code,
                    'name': self._get_country_name(country_code),
                    'valid_regions_count': len(self._valid_regions_cache[country_code]),
                    'inference_status': 'satellite_ai',
                    'inference_note': 'Full AI prediction using satellite data'
                })
        
        return sorted(valid_countries, key=lambda x: x['name'])
    
    def get_valid_regions_for_country(self, country_code: str) -> List[str]:
        """
        Get valid regions for a country that exist in both datasets
        
        Args:
            country_code: Country code (e.g., 'AFG')
            
        Returns:
            List of valid region names
        """
        return self._valid_regions_cache.get(country_code, [])
    
    def get_region_mapping_info(self, country_code: str, region_name: str) -> Dict:
        """
        Get detailed mapping information for a specific region
        
        Args:
            country_code: Country code
            region_name: Region name from IPC data
            
        Returns:
            Dictionary with mapping details
        """
        mapping_status = self._mapping_status_cache.get(country_code, {})
        return mapping_status.get(region_name, {
            'status': 'unknown',
            'satellite_name': None,
            'mapping_type': 'unknown'
        })
    
    def validate_region_for_inference(self, country_code: str, region_name: str) -> bool:
        """
        Check if a region can be used for AI inference
        
        Args:
            country_code: Country code
            region_name: Region name
            
        Returns:
            True if region can be used for inference
        """
        valid_regions = self.get_valid_regions_for_country(country_code)
        return region_name in valid_regions
    
    def get_satellite_region_name_for_inference(self, country_code: str, region_name: str) -> Optional[str]:
        """
        Get the satellite region name to use for inference
        
        Args:
            country_code: Country code
            region_name: Region name from IPC data
            
        Returns:
            Satellite region name to use for inference, or None if not available
        """
        mapping_info = self.get_region_mapping_info(country_code, region_name)
        
        if mapping_info['status'] in ['exact_match', 'mapped', 'reverse_mapped']:
            return mapping_info['satellite_name']
        
        return None
    
    def get_data_coverage_summary(self) -> Dict:
        """
        Get summary of data coverage and mapping status
        
        Returns:
            Dictionary with coverage statistics
        """
        total_countries = len(self._valid_regions_cache)
        total_valid_regions = sum(len(regions) for regions in self._valid_regions_cache.values())
        
        # Count mapping types
        mapping_counts = {'exact_match': 0, 'mapped': 0, 'reverse_mapped': 0, 'unmapped': 0}
        
        for country_mappings in self._mapping_status_cache.values():
            for region_mapping in country_mappings.values():
                status = region_mapping['status']
                if status in mapping_counts:
                    mapping_counts[status] += 1
        
        return {
            'total_countries': total_countries,
            'total_valid_regions': total_valid_regions,
            'mapping_breakdown': mapping_counts,
            'coverage_percentage': (total_valid_regions / max(1, len(self.ipc_data['Level 1'].dropna().unique()))) * 100
        }
    
    def get_unmapped_regions(self) -> List[Dict]:
        """
        Get list of regions that couldn't be mapped
        
        Returns:
            List of unmapped regions with details
        """
        unmapped = []
        
        for country_code, mappings in self._mapping_status_cache.items():
            for region_name, mapping_info in mappings.items():
                if mapping_info['status'] == 'unmapped':
                    unmapped.append({
                        'country_code': country_code,
                        'country_name': self._get_country_name(country_code),
                        'region_name': region_name,
                        'ipc_data_available': True,
                        'satellite_data_available': False
                    })
        
        return unmapped
    
    def get_satellite_only_regions(self) -> List[Dict]:
        """
        Get regions that exist in satellite data but not in IPC data
        
        Returns:
            List of satellite-only regions
        """
        satellite_only = []
        
        for country_code in self._valid_regions_cache.keys():
            ipc_regions = set(self.ipc_data[self.ipc_data['Country'] == country_code]['Level 1'].dropna().unique())
            sat_regions = set(self.satellite_embeddings[self.satellite_embeddings['Country'] == country_code]['Level1'].unique())
            
            # Find regions that only exist in satellite data
            sat_only = sat_regions - ipc_regions
            
            for region_name in sat_only:
                satellite_only.append({
                    'country_code': country_code,
                    'country_name': self._get_country_name(country_code),
                    'region_name': region_name,
                    'ipc_data_available': False,
                    'satellite_data_available': True
                })
        
        return satellite_only
    
    def _get_country_name(self, country_code: str) -> str:
        """Get country name from country code"""
        # This should be imported from data_loader, but for now we'll use a simple mapping
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
    
    def export_mapping_report(self, filepath: str = None) -> str:
        """
        Export a comprehensive mapping report
        
        Args:
            filepath: Optional filepath to save report
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("IPC Data to Satellite Embeddings Mapping Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary statistics
        coverage = self.get_data_coverage_summary()
        report_lines.append(f"Total Countries: {coverage['total_countries']}")
        report_lines.append(f"Total Valid Regions: {coverage['total_valid_regions']}")
        report_lines.append(f"Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
        report_lines.append("")
        
        # Mapping breakdown
        report_lines.append("Mapping Status Breakdown:")
        for status, count in coverage['mapping_breakdown'].items():
            report_lines.append(f"  {status.replace('_', ' ').title()}: {count}")
        report_lines.append("")
        
        # Country-by-country breakdown
        report_lines.append("Country-by-Country Breakdown:")
        report_lines.append("-" * 40)
        
        for country_code in sorted(self._valid_regions_cache.keys()):
            country_name = self._get_country_name(country_code)
            valid_regions = self._valid_regions_cache[country_code]
            mapping_status = self._mapping_status_cache[country_code]
            
            report_lines.append(f"\n{country_name} ({country_code}):")
            report_lines.append(f"  Valid Regions: {len(valid_regions)}")
            
            # Show mapping status for each region
            for region_name in sorted(valid_regions):
                status_info = mapping_status.get(region_name, {})
                status = status_info.get('status', 'unknown')
                sat_name = status_info.get('satellite_name', 'N/A')
                report_lines.append(f"    {region_name} -> {sat_name} ({status})")
        
        # Unmapped regions
        unmapped = self.get_unmapped_regions()
        if unmapped:
            report_lines.append(f"\nUnmapped Regions ({len(unmapped)}):")
            report_lines.append("-" * 30)
            for region in unmapped[:20]:  # Limit to first 20
                report_lines.append(f"  {region['country_name']} - {region['region_name']}")
            if len(unmapped) > 20:
                report_lines.append(f"  ... and {len(unmapped) - 20} more")
        
        report_content = "\n".join(report_lines)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content

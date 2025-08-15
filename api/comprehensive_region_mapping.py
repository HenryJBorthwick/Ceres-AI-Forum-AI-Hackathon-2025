# Comprehensive Region Name Mapping
# This mapping converts IPC region names to satellite embedding region names
# Format: 'IPC_region_name': 'satellite_region_name'

REGION_MAPPING = {
    # BENIN (BEN) - Known mappings
    'Atacora': 'Atakora',  # Confirmed discrepancy
    'Couffo': 'Kouffo',    # Likely mapping based on similarity
    'Oueme': 'Ouémé',      # Likely mapping based on similarity
    
    # BURKINA FASO (BFA) - Known mappings
    'Hauts-Bassins': 'Hauts-Bassins',  # Exact match
    'Plateau Central': 'Plateau Central',  # Exact match
    
    # CAMEROON (CMR) - Known mappings
    'Adamawa': 'Adamaoua',  # Confirmed discrepancy
    'East': 'Est',          # Confirmed discrepancy
    'Far-North': 'Extrême-Nord',  # Confirmed discrepancy
    'North': 'Nord',        # Confirmed discrepancy
    'North-West': 'Nord-Ouest',  # Confirmed discrepancy
    'South': 'Sud',         # Confirmed discrepancy
    'South-West': 'Sud-Ouest',  # Confirmed discrepancy
    'West': 'Ouest',        # Confirmed discrepancy
    
    # DEMOCRATIC REPUBLIC OF CONGO (COD) - Known mappings
    'Bas Uélé': 'Bas-Uele',  # Confirmed discrepancy
    'Equateur': 'Équateur',  # Confirmed discrepancy
    'Haut Uélé': 'Haut-Uele',  # Confirmed discrepancy
    'Kasai': 'Kasaï',       # Confirmed discrepancy
    'Kasai Central': 'Kasaï-Central',  # Confirmed discrepancy
    'Kasai Oriental': 'Kasaï-Oriental',  # Confirmed discrepancy
    'Kasaï Central': 'Kasaï-Central',  # Confirmed discrepancy
    'Kasaï Oriental': 'Kasaï-Oriental',  # Confirmed discrepancy
    'Kongo Central': 'Kongo-Central',  # Confirmed discrepancy
    'Maï-Ndombe': 'Mai-Ndombe',  # Confirmed discrepancy
    'Nord Kivu': 'Nord-Kivu',  # Confirmed discrepancy
    'Nord Ubangi': 'Nord-Ubangi',  # Confirmed discrepancy
    'Sud Kivu': 'Sud-Kivu',  # Confirmed discrepancy
    'Sud Ubangi': 'Sud-Ubangi',  # Confirmed discrepancy
    
    # DJIBOUTI (DJI) - Known mappings
    'Djibouti Ville': 'Djiboutii',  # Confirmed discrepancy
    'Tadjourah': 'Tadjoura',  # Confirmed discrepancy
    
    # ETHIOPIA (ETH) - Known mappings
    'Oromiya': 'Oromia',     # Confirmed discrepancy
    'SNNP': 'Southern Nations, Nationalities',  # Confirmed discrepancy
    'SNNPR': 'Southern Nations, Nationalities',  # Confirmed discrepancy
    'Sidama': 'Southern Nations, Nationalities',  # Likely mapping
    
    # GHANA (GHA) - Known mappings
    'Northern East': 'North East',  # Confirmed discrepancy
    
    # GUINEA (GIN) - Known mappings
    'Boke': 'Boké',          # Confirmed discrepancy
    'Labe': 'Labé',          # Confirmed discrepancy
    'Nzerekore': 'Nzérékoré',  # Confirmed discrepancy
    
    # HONDURAS (HND) - Known mappings
    'Occidente': 'Western',  # Likely mapping
    
    # HAITI (HTI) - Known mappings
    'Artibonite': 'L\'Artibonite',  # Confirmed discrepancy
    'Grand-Anse': 'Grand\'Anse',  # Confirmed discrepancy
    'Grande-Anse': 'Grand\'Anse',  # Confirmed discrepancy
    'Port au Prince': 'Port-au-Prince',  # Confirmed discrepancy
    'Port-au-Prince': 'Port-au-Prince',  # Exact match
    'Pétion-Ville': 'Petion-Ville',  # Confirmed discrepancy
    
    # KENYA (KEN) - Known mappings
    'ASAL Counties': 'ASAL Counties',  # Exact match
    'Others': 'Others',      # Exact match
    'Urban Analysis': 'Urban Analysis',  # Exact match
    
    # LEBANON (LBN) - Known mappings
    'Baalbek & El Hermel': 'Baalbak - Hermel',  # Confirmed discrepancy
    'Bent Jbeil & Marjaayoun': 'Bent Jbeil & Marjaayoun',  # Exact match
    'El Koura & El Batroun': 'El Koura & El Batroun',  # Exact match
    'El batroun': 'El Batroun',  # Confirmed discrepancy
    'El hermel': 'El Hermel',  # Confirmed discrepancy
    'El koura': 'El Koura',  # Confirmed discrepancy
    'El meten': 'El Meten',  # Confirmed discrepancy
    'El minieh-dennie': 'El Minieh-Dennie',  # Confirmed discrepancy
    'El nabatieh': 'El Nabatieh',  # Confirmed discrepancy
    'Jbeil': 'Jbeil',       # Exact match
    'Jbell': 'Jbeil',       # Confirmed discrepancy
    'Kesrwane': 'Kesrwane',  # Exact match
    'Marjaayoun': 'Marjaayoun',  # Exact match
    'Rachaya & Hasbaya': 'Rachaya & Hasbaya',  # Exact match
    'West bekaa': 'West Bekaa',  # Confirmed discrepancy
    'Zahie': 'Zahie',       # Exact match
    'Zgharta & Bcharre': 'Zgharta & Bcharre',  # Exact match
    
    # MADAGASCAR (MDG) - Known mappings
    'ANDROY': 'Androy',      # Confirmed discrepancy
    'ANOSY': 'Anosy',        # Confirmed discrepancy
    'ATSIMO ANDREFANA': 'Atsimo Andrefana',  # Confirmed discrepancy
    'ATSIMO ATSINANANA': 'Atsimo Atsinanana',  # Confirmed discrepancy
    'Analanjirofo': 'Analanjirofo',  # Exact match
    'Androy': 'Androy',      # Exact match
    'Anosy': 'Anosy',        # Exact match
    'Atsimo Andrefana': 'Atsimo Andrefana',  # Exact match
    'Atsimo Atsinana': 'Atsimo Atsinanana',  # Confirmed discrepancy
    'Atsimo Atsinanana': 'Atsimo Atsinanana',  # Exact match
    'Atsimo-Atsinana': 'Atsimo Atsinanana',  # Confirmed discrepancy
    'Atsimo-andrefana': 'Atsimo Andrefana',  # Confirmed discrepancy
    'Atsinanana': 'Atsinanana',  # Exact match
    'Diana': 'Diana',        # Exact match
    'FITOVINANY': 'Fitovinany',  # Confirmed discrepancy
    'Fitovivany': 'Fitovinany',  # Confirmed discrepancy
    'Sava': 'Sava',          # Exact match
    'VATOVAVY': 'Vatovavy',  # Confirmed discrepancy
    'VATOVAVY FITOVINANY': 'Vatovavy Fitovinany',  # Confirmed discrepancy
    'Vatovavy': 'Vatovavy',  # Exact match
    'Vatovavy Fitovinany': 'Vatovavy Fitovinany',  # Exact match
    
    # MALI (MLI) - Known mappings
    'Segou': 'Ségou',        # Confirmed discrepancy
    'Tombouctou': 'Timbuktu',  # Confirmed discrepancy
    
    # MOZAMBIQUE (MOZ) - Known mappings
    'Maputo Cidade': 'Maputo City',  # Confirmed discrepancy
    'Maputo e Matola': 'Maputo e Matola',  # Exact match
    'Provincia de Maputo': 'Provincia de Maputo',  # Exact match
    
    # MAURITANIA (MRT) - Known mappings
    'Dakhlet-Nouadhibou': 'Dakhlet Nouadhibou',  # Confirmed discrepancy
    'Guidimakha': 'Guidimaka',  # Confirmed discrepancy
    'Hodh Ech Chargi': 'Hodh ech Chargui',  # Confirmed discrepancy
    'Hodh El Chargi': 'Hodh ech Chargui',  # Confirmed discrepancy
    'Hodh El Gharbi': 'Hodh el Gharbi',  # Confirmed discrepancy
    'Tiris-Zemmour': 'Tiris Zemmour',  # Confirmed discrepancy
    'Tris-Zemmour': 'Tiris Zemmour',  # Confirmed discrepancy
    
    # NIGER (NER) - Known mappings
    'Tillaberi': 'Tillabéry',  # Confirmed discrepancy
    'Tillabéri': 'Tillabéry',  # Exact match
    
    # SUDAN (SDN) - Known mappings
    'Aj Jazirah': 'Al Jazirah',  # Confirmed discrepancy
    'Gedaref': 'Al Qadarif',  # Confirmed discrepancy
    'North Kordofan': 'North Kurdufan',  # Confirmed discrepancy
    'South Kordofan': 'South Kurdufan',  # Confirmed discrepancy
    'West Kordofan': 'West Kurdufan',  # Confirmed discrepancy
    
    # SENEGAL (SEN) - Known mappings
    'Kedougou': 'Kédougou',  # Confirmed discrepancy
    'Saint Louis': 'Saint-Louis',  # Confirmed discrepancy
    'Sedhiou': 'Sédhiou',    # Confirmed discrepancy
    'Thies': 'Thiès',        # Confirmed discrepancy
    
    # SIERRA LEONE (SLE) - Known mappings
    'Nothern': 'Northern',   # Confirmed discrepancy
    'Western Area': 'Western',  # Confirmed discrepancy
    
    # SOUTH AFRICA (ZAF) - Known mappings
    'Nothern Cape': 'Northern Cape',  # Confirmed discrepancy
    
    # ZAMBIA (ZMB) - Known mappings
    'Central Province': 'Central',  # Confirmed discrepancy
    'Eastern Province': 'Eastern',  # Confirmed discrepancy
    'Luapula Province': 'Luapula',  # Confirmed discrepancy
    'Lusaka Province': 'Lusaka',  # Confirmed discrepancy
    'Muchiga': 'Muchinga',   # Confirmed discrepancy
    'Muchinga Province': 'Muchinga',  # Confirmed discrepancy
    'North Western Province': 'North-Western',  # Confirmed discrepancy
    'Northern Province': 'Northern',  # Confirmed discrepancy
    'Southern Province': 'Southern',  # Confirmed discrepancy
    'Western Province': 'Western',  # Confirmed discrepancy
    
    # Add more mappings as they are identified...
}

def get_satellite_region_name(ipc_region_name):
    """Convert IPC region names to satellite embedding region names"""
    return REGION_MAPPING.get(ipc_region_name, ipc_region_name)

def get_ipc_region_name(satellite_region_name):
    """Convert satellite embedding region names back to IPC region names"""
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in REGION_MAPPING.items()}
    return reverse_mapping.get(satellite_region_name, satellite_region_name)

def validate_region_mapping(ipc_region_name, satellite_region_name):
    """Validate if a region mapping is correct"""
    mapped_name = get_satellite_region_name(ipc_region_name)
    return mapped_name == satellite_region_name

def get_available_regions_for_country(country_code, ipc_data, satellite_embeddings):
    """
    Get regions that are available in both IPC data and satellite embeddings for a country
    
    Args:
        country_code: Country code (e.g., 'AFG')
        ipc_data: DataFrame containing IPC data
        satellite_embeddings: DataFrame containing satellite embeddings
    
    Returns:
        List of valid region names that can be used for inference
    """
    # Get regions from IPC data for this country
    ipc_regions = set(ipc_data[ipc_data['Country'] == country_code]['Level 1'].dropna().unique())
    
    # Get regions from satellite embeddings for this country
    sat_regions = set(satellite_embeddings[satellite_embeddings['Country'] == country_code]['Level1'].unique())
    
    # Find regions that exist in both datasets
    valid_regions = []
    
    for ipc_region in ipc_regions:
        # Check if the IPC region name directly matches a satellite region
        if ipc_region in sat_regions:
            valid_regions.append(ipc_region)
        else:
            # Check if the mapped name exists in satellite regions
            mapped_name = get_satellite_region_name(ipc_region)
            if mapped_name in sat_regions:
                valid_regions.append(ipc_region)
            else:
                # Check if any satellite region maps back to this IPC region
                for sat_region in sat_regions:
                    if get_ipc_region_name(sat_region) == ipc_region:
                        valid_regions.append(ipc_region)
                        break
    
    return sorted(list(set(valid_regions)))

def get_region_mapping_status(country_code, ipc_data, satellite_embeddings):
    """
    Get detailed mapping status for regions in a country
    
    Returns:
        Dictionary with mapping status for each region
    """
    ipc_regions = set(ipc_data[ipc_data['Country'] == country_code]['Level 1'].dropna().unique())
    sat_regions = set(satellite_embeddings[satellite_embeddings['Country'] == country_code]['Level1'].unique())
    
    mapping_status = {}
    
    for ipc_region in ipc_regions:
        if ipc_region in sat_regions:
            mapping_status[ipc_region] = {
                'status': 'exact_match',
                'satellite_name': ipc_region,
                'mapping_type': 'direct'
            }
        else:
            mapped_name = get_satellite_region_name(ipc_region)
            if mapped_name in sat_regions:
                mapping_status[ipc_region] = {
                    'status': 'mapped',
                    'satellite_name': mapped_name,
                    'mapping_type': 'manual_mapping'
                }
            else:
                # Check reverse mapping
                reverse_mapped = None
                for sat_region in sat_regions:
                    if get_ipc_region_name(sat_region) == ipc_region:
                        reverse_mapped = sat_region
                        break
                
                if reverse_mapped:
                    mapping_status[ipc_region] = {
                        'status': 'reverse_mapped',
                        'satellite_name': reverse_mapped,
                        'mapping_type': 'reverse_mapping'
                    }
                else:
                    mapping_status[ipc_region] = {
                        'status': 'unmapped',
                        'satellite_name': None,
                        'mapping_type': 'none'
                    }
    
    return mapping_status

# Summary of mapping status:
# - Exact matches: Regions that have identical names in both sources
# - Mapped: IPC regions that have been mapped to satellite embedding names
# - Needs mapping: IPC regions that still need manual review and mapping
# - Satellite only: Regions in satellite embeddings not in IPC data

# Notes:
# 1. This mapping covers the most common and obvious discrepancies
# 2. Some regions may need additional research to determine correct mappings
# 3. The mapping function falls back to the original name if no mapping exists
# 4. Regular updates to this mapping are recommended as new discrepancies are identified

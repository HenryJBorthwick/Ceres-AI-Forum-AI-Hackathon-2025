from __future__ import annotations

import pathlib
from typing import List, Set
import json
import time

import ee
import pandas as pd
import geopandas as gpd
import pygadm


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def initialize_gee() -> None:
    """Initialize Google Earth Engine with existing credentials."""
    try:
        # Use the high-volume endpoint for better performance with many small requests
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("✅ Google Earth Engine initialized successfully")
        return
    except Exception as e:
        print(f"⚠️ Initial initialization failed: {str(e)}")
        fallback_project = 'gee-hackathon-project'  # A common fallback project
        try:
            print(f"🔄 Trying with project: {fallback_project}")
            ee.Initialize(project=fallback_project, opt_url='https://earthengine-highvolume.googleapis.com')
            print("✅ Google Earth Engine initialized successfully")
            return
        except Exception as fallback_error:
            print(f"   ❌ Failed: {str(fallback_error)}")

        print("❌ All initialization attempts failed!")
        print("💡 Try one of these solutions:")
        print("   1. Authenticate with `earthengine authenticate` in your terminal.")
        print("   2. Set a default project: `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`")
        raise Exception("Could not initialize Google Earth Engine")


def get_countries_and_years_from_ipc(ipc_csv: pathlib.Path) -> tuple[Set[str], List[int]]:
    """Extract unique countries and years from IPC dataset."""
    if not ipc_csv.exists():
        raise FileNotFoundError(f"IPC dataset not found: {ipc_csv}")
    
    df = pd.read_csv(ipc_csv)
    countries = set(df['Country'].unique())
    years = [int(year) for year in sorted(df['Start_year'].unique())]
    
    print(f"Found {len(countries)} countries in IPC dataset.")
    print(f"Years to process: {years}")
    
    return countries, years


def country_code_to_name(country_code: str) -> str:
    """Convert 3-letter country codes to full country names for GADM."""
    code_to_name = {
        'AFG': 'Afghanistan', 'AGO': 'Angola', 'BDI': 'Burundi', 'BEN': 'Benin',
        'BFA': 'Burkina Faso', 'BGD': 'Bangladesh', 'CAF': 'Central African Republic',
        'CIV': "Côte d'Ivoire", 'CMR': 'Cameroon', 'COD': 'Democratic Republic of the Congo',
        'COG': 'Republic of the Congo', 'DJI': 'Djibouti', 'ERI': 'Eritrea',
        'ETH': 'Ethiopia', 'GHA': 'Ghana', 'GIN': 'Guinea', 'GMB': 'Gambia',
        'GNB': 'Guinea-Bissau', 'GTM': 'Guatemala', 'HND': 'Honduras',
        'HTI': 'Haiti', 'IRQ': 'Iraq', 'KEN': 'Kenya', 'LBN': 'Lebanon',
        'LBR': 'Liberia', 'LBY': 'Libya', 'LSO': 'Lesotho', 'MDG': 'Madagascar',
        'MLI': 'Mali', 'MOZ': 'Mozambique', 'MRT': 'Mauritania', 'MWI': 'Malawi',
        'NER': 'Niger', 'NGA': 'Nigeria', 'NIC': 'Nicaragua', 'PAK': 'Pakistan',
        'PSE': 'Palestine', 'RWA': 'Rwanda', 'SDN': 'Sudan', 'SEN': 'Senegal',
        'SLE': 'Sierra Leone', 'SLV': 'El Salvador', 'SOM': 'Somalia',
        'SSD': 'South Sudan', 'SWZ': 'Eswatini', 'SYR': 'Syria', 'TCD': 'Chad',
        'TGO': 'Togo', 'TZA': 'Tanzania', 'UGA': 'Uganda', 'YEM': 'Yemen',
        'ZAF': 'South Africa', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
    }
    return code_to_name.get(country_code, country_code)


def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float = 0.005) -> gpd.GeoDataFrame:
    """
    Simplify polygon geometries to reduce payload size.
    Level 1 areas are larger, so we can use smaller tolerance for better precision.
    
    Args:
        gdf: GeoDataFrame with polygon geometries
        tolerance: Simplification tolerance (0.005 = ~500m, good for state level)
    """
    print(f"🔧 Simplifying geometries with tolerance {tolerance}...")
    
    original_size = len(str(gdf.geometry.to_json()))
    gdf_copy = gdf.copy()
    gdf_copy['geometry'] = gdf_copy['geometry'].simplify(tolerance=tolerance, preserve_topology=True)
    simplified_size = len(str(gdf_copy.geometry.to_json()))
    
    reduction_pct = ((original_size - simplified_size) / original_size) * 100
    print(f"    📉 Geometry size reduced by {reduction_pct:.1f}%")
    
    return gdf_copy


def load_gadm_boundaries_level1(countries: Set[str], data_dir: pathlib.Path, 
                               simplify_tolerance: float = 0.005) -> gpd.GeoDataFrame:
    """Load administrative boundaries at Level 1 (states/provinces) - MUCH FASTER!"""
    print("📍 Loading administrative boundaries using PyGADM library...")
    print(f"    🚀 Target: Level 1 (states/provinces) - FAST MODE!")

    all_boundaries = []
    shapefiles_dir = data_dir / "shapefiles"
    shapefiles_dir.mkdir(parents=True, exist_ok=True)
    
    for country_code in sorted(countries):
        try:
            if country_code == 'PAK':
                print(f"  📥 Downloading boundaries for Pakistan (PAK) using specific admin code...")
                # For Pakistan, get level 1 instead of level 2
                gdf = pygadm.Items(admin="PAK.1", content_level=1)
            else:
                country_name = country_code_to_name(country_code)
                print(f"  📥 Downloading boundaries for {country_name} ({country_code})...")
                gdf = pygadm.Items(name=country_name, content_level=1)  # ⚡ LEVEL 1!
            
            # 🔧 Light simplification since Level 1 areas are already larger
            gdf = simplify_geometries(gdf, tolerance=simplify_tolerance)
            
            gdf['Country'] = country_code
            gdf['Level1'] = gdf.get('NAME_1', gdf.get('NAME_0', ''))  # State/Province name
            gdf['Area'] = gdf.get('NAME_1', gdf.get('NAME_0', ''))    # Same as Level1 for consistency
            
            # Simpler Area_ID since we're at state level
            gdf['Area_ID'] = (
                gdf['Country'].astype(str) + "_" +
                gdf['Area'].astype(str).str.replace(' ', '_', regex=False)
            )
            
            shapefile_path = shapefiles_dir / f"{country_code}_adm1.shp"
            gdf.to_file(shapefile_path)
            print(f"    💾 Saved shapefile: {shapefile_path}")
            
            all_boundaries.append(gdf)
            print(f"    ✅ Found {len(gdf)} states/provinces")
            
        except Exception as e:
            country_name = country_code_to_name(country_code)
            print(f"    ⚠️ Failed to download {country_name}: {str(e)}")
            continue
    
    if not all_boundaries:
        raise ValueError("No administrative boundaries could be downloaded")
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_boundaries, ignore_index=True))
    combined_gdf.to_file(data_dir / "shapefiles" / "all_countries_adm1.shp")
    print(f"✅ Total states/provinces loaded: {len(combined_gdf)}")
    print(f"    📊 Reduction from Level 2: ~{4441} districts → {len(combined_gdf)} states/provinces")
    return combined_gdf


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Convert GeoPandas GeoDataFrame to Earth Engine FeatureCollection."""
    print("🔄 Converting boundaries to Earth Engine format...")
    features = []
    for _, row in gdf.iterrows():
        geom_json = row.geometry.__geo_interface__
        properties = {'Area_ID': row['Area_ID'], 'Country': row['Country']}
        feature = ee.Feature(ee.Geometry(geom_json), properties)
        features.append(feature)
    
    fc = ee.FeatureCollection(features)
    print(f"✅ Converted {len(features)} features to Earth Engine format")
    return fc


def extract_embeddings_level1(
    fc: ee.FeatureCollection,
    years: List[int],
    batch_size: int = 20,   # 🔧 Smaller batches due to higher resolution
    scale: int = 100,       # 🔧 Much smaller scale - native is 10m, using 100m for aggregation
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Extract satellite embeddings optimized for Level 1 (state/province) data.
    Much faster than Level 2!
    """
    all_results = []
    
    # DEBUG: Check the collection
    print("🔍 DEBUG: Checking satellite embedding collection...")
    img_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    
    try:
        collection_size = img_collection.size().getInfo()
        print(f"🔍 DEBUG: Collection contains {collection_size} images total")
        
        # Get date range of available data
        date_range = img_collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"]).getInfo()
        print(f"🔍 DEBUG: Collection date range: {date_range}")
        
        # Check band names
        first_img = img_collection.first()
        band_names = first_img.bandNames().getInfo()
        print(f"🔍 DEBUG: Band names: {band_names}")
        print(f"🔍 DEBUG: Number of bands: {len(band_names)}")
        
    except Exception as e:
        print(f"🔍 DEBUG: Error checking collection: {str(e)}")

    for year in years:
        print(f"🛰️  Processing year {year}...")
        
        try:
            # Create proper date range for the year
            start_date = ee.Date.fromYMD(year, 1, 1)
            end_date = start_date.advance(1, 'year')
            
            # DEBUG: Check year filter with bounds
            print(f"🔍 DEBUG: Filtering for year {year} with date range {year}-01-01 to {year+1}-01-01")
            
            # Get the bounds of all features to filter the collection
            fc_bounds = fc.geometry().bounds()
            
            # CRITICAL FIX: Filter by both date AND bounds, then mosaic
            year_filtered = img_collection.filter(ee.Filter.date(start_date, end_date)).filterBounds(fc_bounds)
            year_size = year_filtered.size().getInfo()
            print(f"🔍 DEBUG: Found {year_size} images for year {year} within bounds")
            
            if year_size == 0:
                print(f"  ⚠️ No satellite data found for {year} within the specified bounds. Skipping...")
                continue
            
            # CRITICAL FIX: Create mosaic instead of just taking first image
            print(f"🔍 DEBUG: Creating mosaic from {year_size} tiles...")
            img = year_filtered.mosaic()
            
            # DEBUG: Test the mosaic on a sample area
            test_region = fc.first().geometry()
            test_sample = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=test_region,
                scale=scale,
                maxPixels=1e6,
                bestEffort=True
            ).getInfo()
            
            print(f"🔍 DEBUG: Test sample from mosaic: {test_sample}")
            if test_sample:
                non_null_count = sum(1 for v in test_sample.values() if v is not None)
                print(f"🔍 DEBUG: Non-null bands in mosaic test: {non_null_count}/64")
                if non_null_count > 0:
                    # Show some actual values
                    sample_values = [(k, v) for k, v in test_sample.items() if v is not None][:5]
                    print(f"🔍 DEBUG: Sample values: {sample_values}")
            
            print(f"  📅 Using mosaic for year {year}")

            def extract_for_feature(feature):
                reducer = ee.Reducer.mean()
                embedding = img.reduceRegion(
                    reducer=reducer,
                    geometry=feature.geometry(),
                    scale=scale,           # Much smaller scale for better resolution
                    maxPixels=1e9,         # More pixels allowed 
                    bestEffort=True,
                    tileScale=2            # Higher tile scale for complex geometries
                )
                return feature.set(embedding).set({'year': year})

            total_features = fc.size().getInfo()
            if total_features == 0:
                print("  No features to process. Skipping year.")
                continue
                
            print(f"  ⚙️  Processing {total_features} states/provinces in batches of {batch_size}...")
            print(f"      📏 Using scale: {scale}m (aggregated from native 10m)")
            year_props_list = []

            for i in range(0, total_features, batch_size):
                # Create batch using toList() method
                batch_list = fc.toList(batch_size, i)
                batch_fc = ee.FeatureCollection(batch_list)
                batch_with_embeddings = batch_fc.map(extract_for_feature)
                
                batch_num = (i // batch_size) + 1
                total_batches = -(-total_features // batch_size)  # Ceiling division
                print(f"    📥 Downloading batch {batch_num}/{total_batches} ({min(batch_size, total_features-i)} areas)...")

                # Retry logic for failed batches
                for attempt in range(max_retries):
                    try:
                        batch_info = batch_with_embeddings.getInfo()
                        
                        # DEBUG: Check batch structure
                        print(f"🔍 DEBUG: Batch info type: {type(batch_info)}")
                        if 'features' in batch_info:
                            print(f"🔍 DEBUG: Batch contains {len(batch_info['features'])} features")
                            
                            # Check first feature
                            if batch_info['features']:
                                first_feature = batch_info['features'][0]
                                print(f"🔍 DEBUG: First feature keys: {list(first_feature.get('properties', {}).keys())}")
                                
                                # Count embedding bands - look for A00, A01, A02... pattern
                                props = first_feature.get('properties', {})
                                embedding_keys = [k for k in props.keys() if k.startswith('A') and len(k) == 3 and k[1:].isdigit()]
                                print(f"🔍 DEBUG: Found {len(embedding_keys)} embedding bands")
                                
                                # Check some embedding values
                                for key in embedding_keys[:5]:  # First 5 bands
                                    value = props.get(key)
                                    print(f"🔍 DEBUG: {key}: {value}")
                        
                        props = [f['properties'] for f in batch_info['features']]
                        year_props_list.extend(props)
                        print(f"    ✅ Batch {batch_num} successful ({len(props)} areas)")
                        break
                        
                    except Exception as batch_error:
                        if "payload size exceeds" in str(batch_error).lower():
                            print(f"    ⚠️ Batch {batch_num} payload too large (attempt {attempt+1}/{max_retries})")
                            if attempt < max_retries - 1:
                                print(f"    🔄 Retrying with smaller sub-batch...")
                                time.sleep(2)
                            else:
                                print(f"    ❌ Skipping batch {batch_num} after {max_retries} attempts")
                        else:
                            print(f"    ❌ Batch {batch_num} error: {str(batch_error)}")
                            break

            if year_props_list:
                df_year = pd.DataFrame(year_props_list)
                
                # DEBUG: Check DataFrame structure
                print(f"🔍 DEBUG: DataFrame shape: {df_year.shape}")
                print(f"🔍 DEBUG: DataFrame columns: {list(df_year.columns)}")
                
                # Check for embedding columns - look for A00, A01, A02... pattern
                embedding_cols = [col for col in df_year.columns if col.startswith('A') and len(col) == 3 and col[1:].isdigit()]
                print(f"🔍 DEBUG: Embedding columns found: {len(embedding_cols)}")
                
                if embedding_cols:
                    # Check embedding value statistics
                    for col in embedding_cols[:3]:  # First 3 bands
                        values = df_year[col].dropna()
                        if len(values) > 0:
                            print(f"🔍 DEBUG: {col} - min: {values.min():.4f}, max: {values.max():.4f}, mean: {values.mean():.4f}")
                        else:
                            print(f"🔍 DEBUG: {col} - all values are NaN!")
                
                all_results.append(df_year)
                print(f"  ✅ Successfully processed {len(df_year)} states/provinces for {year}")
            else:
                print(f"  ⚠️ No data was extracted for {year}")

        except Exception as e:
            print(f"  ❌ A critical error occurred while processing {year}: {str(e)}")
            continue
            
    if not all_results:
        raise RuntimeError("No embeddings were extracted. Check GEE permissions and data availability.")

    df_all = pd.concat(all_results, ignore_index=True)
    print(f"🎉 Total embeddings extracted: {len(df_all)} state-year combinations")
    
    # DEBUG: Final data check
    print(f"🔍 DEBUG: Final DataFrame shape: {df_all.shape}")
    print(f"🔍 DEBUG: Final DataFrame columns: {list(df_all.columns)}")
    
    return df_all


def clean_embedding_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the embedding data."""
    print("🧹 Cleaning embedding data...")
    
    # Look for the correct band naming pattern: A00, A01, A02, etc.
    embedding_cols = [col for col in df.columns if col.startswith('A') and len(col) == 3 and col[1:].isdigit()]
    
    if not embedding_cols:
        print("⚠️ No embedding columns (e.g., 'A00', 'A01') found. Check extraction results.")
        print(f"🔍 DEBUG: Available columns: {list(df.columns)}")
        return df

    print(f"  Found {len(embedding_cols)} embedding dimensions.")
    
    # DEBUG: Check raw embedding values before cleaning
    print("🔍 DEBUG: Raw embedding statistics before cleaning:")
    for col in embedding_cols[:5]:  # First 5 bands
        values = df[col].dropna()
        if len(values) > 0:
            print(f"  {col}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}, count={len(values)}")
    
    # Convert A00, A01, A02... to band_0, band_1, band_2...
    rename_dict = {col: f'band_{int(col[1:])}' for col in embedding_cols}
    df = df.rename(columns=rename_dict)
    clean_embedding_cols = list(rename_dict.values())
    
    essential_cols = ['Area_ID', 'year', 'Country', 'Level1', 'Area'] + clean_embedding_cols
    available_cols = [col for col in essential_cols if col in df.columns]
    
    df_clean = df[available_cols].copy()
    
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=clean_embedding_cols)
    after_count = len(df_clean)
    
    if before_count > after_count:
        print(f"  Removed {before_count - after_count} rows with missing embedding data.")
    
    # DEBUG: Check final embedding statistics
    print("🔍 DEBUG: Final embedding statistics after cleaning:")
    for col in clean_embedding_cols[:5]:  # First 5 bands
        if col in df_clean.columns:
            values = df_clean[col].dropna()
            if len(values) > 0:
                print(f"  {col}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    
    print(f"✅ Clean dataset has {len(df_clean)} rows and {len(df_clean.columns)} columns.")
    return df_clean


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main execution function optimized for Level 1 (FAST!)."""
    src_dir = pathlib.Path(__file__).resolve().parent
    data_dir = src_dir / "data"
    
    ipc_csv = data_dir / "ipc_annual_dataset.csv"
    output_csv = data_dir / "satellite_embeddings_level1_FAST.csv"
    
    print("🚀 FAST Admin Level 1 Satellite Embedding Extraction Pipeline")
    print("=" * 65)
    print("⚡ Level 1 = States/Provinces (not districts)")
    print("⚡ Expected runtime: 30-60 minutes (vs 6+ hours for Level 2)")
    print("⚡ ~200-400 areas instead of 4,441")
    print("=" * 65)
    
    initialize_gee()
    
    countries, years = get_countries_and_years_from_ipc(ipc_csv)
    
    # 🚀 Load Level 1 boundaries (states/provinces)
    boundaries_gdf = load_gadm_boundaries_level1(countries, data_dir, simplify_tolerance=0.005)
    
    boundaries_fc = gdf_to_ee_fc(boundaries_gdf)
    
    # 🚀 Extract with Level 1 optimized parameters
    embeddings_df = extract_embeddings_level1(
        boundaries_fc, 
        years, 
        batch_size=100,   # Larger batches OK for Level 1
        scale=2000,       # Better resolution for larger areas
        max_retries=3
    )
    
    full_meta = boundaries_gdf[['Area_ID', 'Level1', 'Area']].drop_duplicates(subset=['Area_ID'])
    merged_df = pd.merge(embeddings_df, full_meta, on='Area_ID', how='left')
    
    clean_df = clean_embedding_data(merged_df)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    
    print(f"\n💾 Level 1 satellite embeddings saved to: {output_csv}")
    print(f"📊 Final dataset: {len(clean_df)} state-year combinations")
    print(f"⚡ Processing complete! Much faster than Level 2.")
    print("🎯 Ready for ML model training!")
    
    # Quick stats
    if len(clean_df) > 0:
        unique_countries = clean_df['Country'].nunique()
        unique_areas = clean_df['Area_ID'].nunique() 
        unique_years = clean_df['year'].nunique()
        print(f"\n📈 Dataset Summary:")
        print(f"   Countries: {unique_countries}")
        print(f"   States/Provinces: {unique_areas}")
        print(f"   Years: {unique_years}")
        print(f"   Total combinations: {len(clean_df)}")


if __name__ == "__main__":
    main()
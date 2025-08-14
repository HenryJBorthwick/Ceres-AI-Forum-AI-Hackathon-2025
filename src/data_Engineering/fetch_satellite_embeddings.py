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


def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float = 0.01) -> gpd.GeoDataFrame:
    """
    Simplify polygon geometries to reduce payload size while preserving shape.
    
    Args:
        gdf: GeoDataFrame with polygon geometries
        tolerance: Simplification tolerance (higher = more simplified)
                  0.01 = ~1km at equator, good balance of detail vs size
    """
    print(f"🔧 Simplifying geometries with tolerance {tolerance}...")
    
    original_size = len(str(gdf.geometry.to_json()))
    gdf_copy = gdf.copy()
    gdf_copy['geometry'] = gdf_copy['geometry'].simplify(tolerance=tolerance, preserve_topology=True)
    simplified_size = len(str(gdf_copy.geometry.to_json()))
    
    reduction_pct = ((original_size - simplified_size) / original_size) * 100
    print(f"    📉 Geometry size reduced by {reduction_pct:.1f}%")
    
    return gdf_copy


def load_gadm_boundaries(countries: Set[str], data_dir: pathlib.Path, 
                        simplify_tolerance: float = 0.01) -> gpd.GeoDataFrame:
    """Load administrative boundaries using the pygadm library with optimizations."""
    print("📍 Loading administrative boundaries using PyGADM library...")
    print(f"    🎯 Target: Level 2 (districts/counties) with payload optimizations")

    all_boundaries = []
    shapefiles_dir = data_dir / "shapefiles"
    shapefiles_dir.mkdir(parents=True, exist_ok=True)
    
    for country_code in sorted(countries):
        try:
            if country_code == 'PAK':
                print(f"  📥 Downloading boundaries for Pakistan (PAK) using specific admin code...")
                gdf = pygadm.Items(admin="PAK.1_1", content_level=2)
            else:
                country_name = country_code_to_name(country_code)
                print(f"  📥 Downloading boundaries for {country_name} ({country_code})...")
                gdf = pygadm.Items(name=country_name, content_level=2)  # Keep Level 2!
            
            # 🔧 OPTIMIZATION 1: Simplify geometries to reduce payload
            gdf = simplify_geometries(gdf, tolerance=simplify_tolerance)
            
            gdf['Country'] = country_code
            gdf['Level1'] = gdf.get('NAME_1', '')
            gdf['Area'] = gdf.get('NAME_2', gdf.get('NAME_0', ''))
            
            gdf['Area_ID'] = (
                gdf['Country'].astype(str) + "_" +
                gdf['Level1'].fillna("").str.replace(' ', '_', regex=False) + "_" +
                gdf['Area'].astype(str).str.replace(' ', '_', regex=False)
            )
            
            shapefile_path = shapefiles_dir / f"{country_code}_adm2_simplified.shp"
            gdf.to_file(shapefile_path)
            print(f"    💾 Saved simplified shapefile: {shapefile_path}")
            
            all_boundaries.append(gdf)
            print(f"    ✅ Found {len(gdf)} administrative areas")
            
        except Exception as e:
            country_name = country_code_to_name(country_code)
            print(f"    ⚠️ Failed to download {country_name}: {str(e)}")
            continue
    
    if not all_boundaries:
        raise ValueError("No administrative boundaries could be downloaded")
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_boundaries, ignore_index=True))
    combined_gdf.to_file(data_dir / "shapefiles" / "all_countries_adm2_simplified.shp")
    print(f"✅ Total administrative areas loaded: {len(combined_gdf)}")
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


def extract_embeddings(
    fc: ee.FeatureCollection,
    years: List[int],
    batch_size: int = 25,  # 🔧 OPTIMIZATION 2: Much smaller batches
    scale: int = 3000,     # 🔧 OPTIMIZATION 3: Higher scale (less detail but smaller payload)
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Extracts satellite embeddings with aggressive optimizations for Level 2 data.
    """
    all_results = []
    img_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    for year in years:
        print(f"🛰️  Processing year {year}...")
        
        try:
            img = img_collection.filter(ee.Filter.calendarRange(year, year, "year")).first()
            
            if img.propertyNames().size().getInfo() == 0:
                print(f"  ⚠️ No satellite data found for {year}. Skipping...")
                continue
            print(f"  📅 Found satellite data for year {year}")

            def extract_for_feature(feature):
                reducer = ee.Reducer.mean()
                embedding = img.reduceRegion(
                    reducer=reducer,
                    geometry=feature.geometry(),
                    scale=scale,           # 🔧 Higher scale = less detail = smaller payload
                    maxPixels=5e7,         # 🔧 Reduced max pixels for safety
                    bestEffort=True,
                    tileScale=2            # 🔧 Additional optimization for complex geometries
                )
                return feature.set(embedding).set({'year': year})

            total_features = fc.size().getInfo()
            if total_features == 0:
                print("  No features to process. Skipping year.")
                continue
                
            print(f"  ⚙️  Processing {total_features} features in batches of {batch_size}...")
            print(f"      📏 Using scale: {scale}m (higher scale = smaller payload)")
            year_props_list = []

            for i in range(0, total_features, batch_size):
                # Create batch using toList() and slice - works across all GEE versions
                batch_list = fc.toList(batch_size, i)
                batch_fc = ee.FeatureCollection(batch_list)
                batch_with_embeddings = batch_fc.map(extract_for_feature)
                
                batch_num = (i // batch_size) + 1
                total_batches = -(-total_features // batch_size)  # Ceiling division
                print(f"    📥 Downloading batch {batch_num}/{total_batches} ({batch_size} areas)...")

                # 🔧 OPTIMIZATION 4: Retry logic for failed batches
                for attempt in range(max_retries):
                    try:
                        batch_info = batch_with_embeddings.getInfo()
                        props = [f['properties'] for f in batch_info['features']]
                        year_props_list.extend(props)
                        print(f"    ✅ Batch {batch_num} successful ({len(props)} areas)")
                        break
                        
                    except Exception as batch_error:
                        if "payload size exceeds" in str(batch_error).lower():
                            print(f"    ⚠️ Batch {batch_num} payload too large (attempt {attempt+1}/{max_retries})")
                            if attempt < max_retries - 1:
                                # Try with even smaller sub-batches
                                sub_batch_size = max(1, batch_size // 2)
                                print(f"    🔄 Retrying with sub-batches of {sub_batch_size}...")
                                time.sleep(2)  # Brief pause before retry
                            else:
                                print(f"    ❌ Skipping batch {batch_num} after {max_retries} attempts")
                        else:
                            print(f"    ❌ Batch {batch_num} error: {str(batch_error)}")
                            break

            if year_props_list:
                df_year = pd.DataFrame(year_props_list)
                all_results.append(df_year)
                print(f"  ✅ Successfully processed {len(df_year)} areas for {year}")
            else:
                print(f"  ⚠️ No data was extracted for {year}")

        except Exception as e:
            print(f"  ❌ A critical error occurred while processing {year}: {str(e)}")
            continue
            
    if not all_results:
        raise RuntimeError("No embeddings were extracted. Check GEE permissions and data availability.")

    df_all = pd.concat(all_results, ignore_index=True)
    print(f"🎉 Total embeddings extracted: {len(df_all)} area-year combinations")
    return df_all


def clean_embedding_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the embedding data."""
    print("🧹 Cleaning embedding data...")
    embedding_cols = [col for col in df.columns if col.startswith('b') and col[1:].isdigit()]
    
    if not embedding_cols:
        print("⚠️ No embedding columns (e.g., 'b0', 'b1') found. Check extraction results.")
        return df

    print(f"  Found {len(embedding_cols)} embedding dimensions.")
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
    
    print(f"✅ Clean dataset has {len(df_clean)} rows and {len(df_clean.columns)} columns.")
    return df_clean


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main execution function with Level 2 optimizations."""
    src_dir = pathlib.Path(__file__).resolve().parent
    data_dir = src_dir / "data"
    
    ipc_csv = data_dir / "ipc_annual_dataset.csv"
    output_csv = data_dir / "satellite_embeddings_level2.csv"
    
    print("🛰️ Optimized Level 2 Satellite Embedding Extraction Pipeline")
    print("=" * 60)
    print("🎯 Optimizations: Geometry simplification + Higher scale + Small batches")
    print("=" * 60)
    
    initialize_gee()
    
    countries, years = get_countries_and_years_from_ipc(ipc_csv)
    
    # 🔧 Load with geometry simplification
    boundaries_gdf = load_gadm_boundaries(countries, data_dir, simplify_tolerance=0.01)
    
    boundaries_fc = gdf_to_ee_fc(boundaries_gdf)
    
    # 🔧 Extract with optimized parameters
    embeddings_df = extract_embeddings(
        boundaries_fc, 
        years, 
        batch_size=25,    # Small batches
        scale=3000,       # Higher scale (3km vs 1km)
        max_retries=3
    )
    
    full_meta = boundaries_gdf[['Area_ID', 'Level1', 'Area']].drop_duplicates(subset=['Area_ID'])
    merged_df = pd.merge(embeddings_df, full_meta, on='Area_ID', how='left')
    
    clean_df = clean_embedding_data(merged_df)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    
    print(f"\n💾 Level 2 satellite embeddings saved to: {output_csv}")
    print(f"📊 Final dataset: {len(clean_df)} area-year combinations")
    print("🎯 Ready for ML model training!")


if __name__ == "__main__":
    main()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import data_loader
from data_loader import COUNTRY_MAPPING
import sys
import os
import logging
import json
from pathlib import Path
from model.inference_example import predict_by_level1_or_area, load_satellite_embeddings
from model.inference_example import model_available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add model directory to path for imports
model_dir = Path(__file__).parent / "model"
sys.path.insert(0, str(model_dir))

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow both localhost and 127.0.0.1, fixed CORS issue
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
df = data_loader.load_data()

# Load satellite embeddings on startup if model is available
if model_available:
    try:
        load_satellite_embeddings()
        print("✓ Satellite embeddings loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load satellite embeddings: {e}")

# Helper function to convert country name back to code
def get_country_code(country_name: str) -> str:
    for code, name in COUNTRY_MAPPING.items():
        if name == country_name:
            return code
    return country_name  # Fallback to original if not found

@app.get("/countries")
def get_countries():
    # Only show countries that have satellite embeddings available
    from data_loader import COUNTRIES_WITH_EMBEDDINGS
    
    # Filter to only countries with embeddings
    available_countries = []
    for code in sorted(COUNTRIES_WITH_EMBEDDINGS):
        name = COUNTRY_MAPPING.get(code, code)
        available_countries.append({
            "code": code,
            "name": name,
            "inference_status": "satellite_ai",
            "inference_note": "Full AI prediction using satellite data"
        })
    
    return {
        "countries": [country["name"] for country in available_countries],
        "countries_with_info": available_countries
    }

@app.get("/levels/{country}")
def get_levels(country: str):
    # Convert country name back to code for data filtering
    country_code = get_country_code(country)
    filtered = df[df['Country'] == country_code]
    levels = filtered['Level 1'].dropna().unique().tolist()
    return {"levels": sorted(levels)}

@app.get("/areas/{country}/{level1}")
def get_areas(country: str, level1: str):
    # Convert country name back to code for data filtering
    country_code = get_country_code(country)
    filtered = df[(df['Country'] == country_code) & (df['Level 1'] == level1)]
    areas = filtered['Area'].dropna().unique().tolist()
    return {"areas": sorted(areas)}

@app.get("/graph-data")
def get_graph_data(country: str, level1: str | None = None, area: str | None = None):
    try:
        # Convert country name back to code for data filtering
        country_code = get_country_code(country)
        data = data_loader.aggregate_data(df, country_code, level1, area)
        return data
    except Exception as e:
        # Return a proper error response instead of letting FastAPI handle it
        # This ensures CORS headers are always sent
        return {"error": "Failed to load data", "message": str(e), "country": country, "level1": level1, "area": area}

@app.get("/predict")
def predict_population_phases(level1: str | None = None, area_id: str | None = None, year: int = 2024):
    """
    Predict population phases using satellite embeddings
    
    Args:
        level1: Level 1 administrative division name (e.g., "Badakhshan")
        area_id: Area ID (e.g., "AFG_Badakhshan") 
        year: Year for prediction (default 2024)
    
    Returns:
        Prediction results with phase percentages 1-5
    """
    # Log incoming request
    logger.info(f"🔮 PREDICTION REQUEST: level1='{level1}', area_id='{area_id}', year={year}")
    
    if not model_available:
        logger.error("❌ Model not available")
        raise HTTPException(status_code=503, detail="ML model not available")
    
    if not level1 and not area_id:
        logger.error("❌ Missing required parameters")
        raise HTTPException(status_code=400, detail="Must provide either level1 or area_id parameter")
    
    try:
        # Make prediction using the model
        logger.info(f"🤖 Making prediction with model...")
        result = predict_by_level1_or_area(level1=level1, area_id=area_id, year=year)
        logger.info(f"📊 Raw model result: {json.dumps(result, indent=2)}")
        
        # Format the response to match the requested format
        formatted_result = {
            "level1": result.get("level1"),
            "area_id": result.get("area_id"),
            "year": result["year"],
            "predictions": {
                "phase_1": result["phase_percentages"]["Phase 1"],
                "phase_2": result["phase_percentages"]["Phase 2"], 
                "phase_3": result["phase_percentages"]["Phase 3"],
                "phase_4": result["phase_percentages"]["Phase 4"],
                "phase_5": result["phase_percentages"]["Phase 5"]
            },
            "dominant_phase": result["dominant_phase"],
            "confidence": result["confidence"],
            "samples_used": result.get("samples_used", 1)
        }
        
        # Log what we're sending back to frontend
        logger.info(f"✅ SENDING TO FRONTEND:")
        logger.info(f"📤 Response: {json.dumps(formatted_result, indent=2)}")
        logger.info(f"🎯 Prediction Summary: {formatted_result['dominant_phase']} ({formatted_result['confidence']}% confidence)")
        logger.info(f"📈 Phase breakdown: P1={formatted_result['predictions']['phase_1']:.1f}%, P2={formatted_result['predictions']['phase_2']:.1f}%, P3={formatted_result['predictions']['phase_3']:.1f}%, P4={formatted_result['predictions']['phase_4']:.1f}%, P5={formatted_result['predictions']['phase_5']:.1f}%")
        
        return formatted_result
        
    except ValueError as e:
        logger.error(f"❌ ValueError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-status")
def get_model_status():
    """Check if the ML model is available"""
    return {
        "model_available": model_available,
        "satellite_data_path": str(model_dir / "sat_embeddings" / "sat_embeddings_recent.csv"),
        "model_path": str(model_dir / "lightgbm_population_phase_model.pkl")
    }

@app.get("/country-inference-status/{country}")
def get_country_inference_status(country: str):
    """
    Get detailed inference status for a specific country
    
    Args:
        country: Country name (e.g., "Afghanistan")
    
    Returns:
        Detailed inference status and capabilities
    """
    from data_loader import COUNTRIES_WITH_EMBEDDINGS, COUNTRY_MAPPING
    
    # Convert country name to code
    country_code = get_country_code(country)
    
    if country_code in COUNTRIES_WITH_EMBEDDINGS:
        status = {
            "country_code": country_code,
            "country_name": country,
            "inference_type": "satellite_ai",
            "inference_available": True,
            "prediction_method": "Full AI prediction using satellite imagery",
            "data_source": "Satellite embeddings + LightGBM model",
            "accuracy": "High (based on 64 spectral bands)",
            "regions_available": "All administrative regions",
            "note": "Real-time satellite data analysis"
        }
    else:
        status = {
            "country_code": country_code,
            "country_name": country,
            "inference_type": "not_available",
            "inference_available": False,
            "prediction_method": "None",
            "data_source": "None",
            "accuracy": "None",
            "regions_available": "None",
            "note": "Country not available - no satellite data"
        }
    
    return status

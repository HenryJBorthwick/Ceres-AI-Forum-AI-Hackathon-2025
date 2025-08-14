from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import data_loader
from data_loader import COUNTRY_MAPPING

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

# Helper function to convert country name back to code
def get_country_code(country_name: str) -> str:
    for code, name in COUNTRY_MAPPING.items():
        if name == country_name:
            return code
    return country_name  # Fallback to original if not found

@app.get("/countries")
def get_countries():
    country_codes = sorted(df['Country'].unique().tolist())
    # Convert codes to names for frontend display
    country_names = [COUNTRY_MAPPING.get(code, code) for code in country_codes]
    return {"countries": sorted(country_names)}

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

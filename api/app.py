from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import data_loader

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

@app.get("/countries")
def get_countries():
    return {"countries": sorted(df['Country'].unique().tolist())}

@app.get("/levels/{country}")
def get_levels(country: str):
    filtered = df[df['Country'] == country]
    levels = filtered['Level 1'].dropna().unique().tolist()
    return {"levels": sorted(levels)}

@app.get("/areas/{country}/{level1}")
def get_areas(country: str, level1: str):
    filtered = df[(df['Country'] == country) & (df['Level 1'] == level1)]
    areas = filtered['Area'].dropna().unique().tolist()
    return {"areas": sorted(areas)}

@app.get("/graph-data")
def get_graph_data(country: str, level1: str | None = None, area: str | None = None):
    try:
        data = data_loader.aggregate_data(df, country, level1, area)
        return data
    except Exception as e:
        # Return a proper error response instead of letting FastAPI handle it
        # This ensures CORS headers are always sent
        return {"error": "Failed to load data", "message": str(e), "country": country, "level1": level1, "area": area}

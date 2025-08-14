
# Example: How to use the saved LightGBM model for inference
import joblib
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Get the current directory for relative paths
current_dir = Path(__file__).parent

# Initialize model availability status
model_available = False

try:
    # Load the trained model
    model = joblib.load(current_dir / 'lightgbm_population_phase_model.pkl')

    # Load model metadata
    with open(current_dir / 'model_info.json', 'r') as f:
        model_info = json.load(f)

    feature_columns = model_info['feature_columns']
    phase_names = model_info['phase_names']

    # Load satellite embeddings data
    sat_embeddings_path = current_dir / 'sat_embeddings' / 'sat_embeddings_recent.csv'
    sat_embeddings_df = None
    
    model_available = True
    
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    model_available = False
    model = None
    feature_columns = None
    phase_names = None
    sat_embeddings_path = None
    sat_embeddings_df = None


def load_satellite_embeddings():
    """Load satellite embeddings data on startup"""
    global sat_embeddings_df
    if not model_available:
        return None
    if sat_embeddings_df is None and sat_embeddings_path and sat_embeddings_path.exists():
        sat_embeddings_df = pd.read_csv(sat_embeddings_path)
    return sat_embeddings_df

def predict_population_phases(new_data):
    """
    Predict population phases for new spectral data
    
    Args:
        new_data: DataFrame with 64 spectral band features (feature_band_0 to feature_band_63)
                 OR numpy array with shape (n_samples, 64)
    
    Returns:
        dict with predictions, dominant_phase, and confidence
    """
    if not model_available:
        raise ValueError("Model not available")
        
    # Ensure we have the right feature columns
    if isinstance(new_data, pd.DataFrame):
        if not all(col in new_data.columns for col in feature_columns):
            raise ValueError(f"Input data must contain columns: {feature_columns}")
        X = new_data[feature_columns]
    else:
        # Assume numpy array with correct shape
        if new_data.shape[1] != 64:
            raise ValueError("Input array must have 64 features (spectral bands)")
        X = new_data
    
    # Make predictions (returns percentages as decimals)
    predictions = model.predict(X)
    
    # Convert to percentages
    predictions_percent = predictions * 100
    
    # Find dominant phase for each sample
    dominant_phases = np.argmax(predictions, axis=1)
    dominant_phase_names = [phase_names[i] for i in dominant_phases]
    
    # Calculate confidence (max probability)
    confidence_scores = np.max(predictions, axis=1)
    
    results = []
    for i in range(len(predictions)):
        result = {
            'sample_id': i,
            'phase_percentages': {
                phase_names[j]: round(predictions_percent[i][j], 1) 
                for j in range(5)
            },
            'dominant_phase': dominant_phase_names[i],
            'confidence': round(confidence_scores[i] * 100, 1)
        }
        results.append(result)
    
    return results

def predict_by_level1_or_area(level1=None, area_id=None, year=2024):
    """
    Predict population phases by Level1 or Area_ID using satellite embeddings
    
    Args:
        level1: Level 1 administrative division name
        area_id: Area ID (e.g., "AFG_Badakhshan")
        year: Year for prediction (default 2024)
    
    Returns:
        dict with predictions for phase percentages
    """
    if not model_available:
        raise ValueError("Model not available")
        
    # Load satellite embeddings if not already loaded
    embeddings_df = load_satellite_embeddings()
    
    if embeddings_df is None:
        raise ValueError("Satellite embeddings data not found")
    
    # Filter by Level1 or Area_ID
    if area_id:
        filtered_df = embeddings_df[
            (embeddings_df['Area_ID'] == area_id) & 
            (embeddings_df['year'] == year)
        ]
    elif level1:
        filtered_df = embeddings_df[
            (embeddings_df['Level1'] == level1) & 
            (embeddings_df['year'] == year)
        ]
    else:
        raise ValueError("Must provide either level1 or area_id")
    
    if filtered_df.empty:
        raise ValueError(f"No satellite data found for the specified criteria")
    
    # Extract band features (convert from band_0...band_63 to feature_band_0...feature_band_63)
    band_cols = [f'band_{i}' for i in range(64)]
    
    # Check if satellite data has the expected columns
    if not all(col in filtered_df.columns for col in band_cols):
        raise ValueError(f"Satellite data missing expected band columns")
    
    # Create feature DataFrame with the expected column names
    feature_data = pd.DataFrame()
    for i in range(64):
        feature_data[f'feature_band_{i}'] = filtered_df[f'band_{i}']
    
    # Make predictions
    predictions = predict_population_phases(feature_data)
    
    # Aggregate predictions if multiple rows (take average)
    if len(predictions) > 1:
        # Average all phase percentages
        avg_percentages = {}
        for phase in phase_names:
            avg_percentages[phase] = round(
                np.mean([pred['phase_percentages'][phase] for pred in predictions]), 1
            )
        
        # Find dominant phase from averaged percentages
        dominant_phase = max(avg_percentages, key=avg_percentages.get)
        confidence = round(max(avg_percentages.values()), 1)
        
        result = {
            'level1': level1,
            'area_id': area_id,
            'year': year,
            'phase_percentages': avg_percentages,
            'dominant_phase': dominant_phase,
            'confidence': confidence,
            'samples_used': len(predictions)
        }
    else:
        result = predictions[0]
        result.update({
            'level1': level1,
            'area_id': area_id,
            'year': year,
            'samples_used': 1
        })
    
    return result


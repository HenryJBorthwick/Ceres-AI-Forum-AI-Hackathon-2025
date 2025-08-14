
# Example: How to use the saved LightGBM model for inference
import joblib
import pandas as pd
import numpy as np
import json

# Load the trained model
model = joblib.load('lightGBM result/lightgbm_population_phase_model.pkl')

# Load model metadata
with open('lightGBM result/model_info.json', 'r') as f:
    model_info = json.load(f)

feature_columns = model_info['feature_columns']
phase_names = model_info['phase_names']

def predict_population_phases(new_data):
    """
    Predict population phases for new spectral data
    
    Args:
        new_data: DataFrame with 64 spectral band features (feature_band_0 to feature_band_63)
                 OR numpy array with shape (n_samples, 64)
    
    Returns:
        dict with predictions, dominant_phase, and confidence
    """
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

# Example usage:
# new_spectral_data = pd.read_csv('new_spectral_data.csv')
# predictions = predict_population_phases(new_spectral_data)
# print(f"Predicted dominant phase: {predictions[0]['dominant_phase']}")
# print(f"Confidence: {predictions[0]['confidence']}%")
# print(f"All phases: {predictions[0]['phase_percentages']}")

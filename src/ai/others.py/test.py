import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ML Methods
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)

print("🤖 MACHINE LEARNING METHODS COMPARISON")
print("🎯 Population Phase Prediction Challenge")
print("=" * 70)

# Load and prepare data
df = pd.read_csv('src/ai/train_predict.csv')

# Separate features (X) and targets (y)
feature_cols = [col for col in df.columns if col.startswith('feature_band_')]
target_cols = [col for col in df.columns if col.startswith('target_avg_pop_phase_')]

X = df[feature_cols]
y = df[target_cols]
y_floats = y.astype(float)

# Split the data (same as neural network for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y_floats, test_size=0.1, random_state=42)

print(f"📊 Dataset Summary:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Target phases: {y_train.shape[1]}")

# Scale features (some methods benefit from scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_train_data, X_test_data, y_train_data, y_test_data, model_name):
    """Train and evaluate a machine learning model"""
    print(f"\n🔬 Training {model_name}...")
    
    # Train the model
    model.fit(X_train_data, y_train_data)
    
    # Make predictions
    predictions = model.predict(X_test_data)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_data, predictions)
    mse = mean_squared_error(y_test_data, predictions)
    
    # Calculate phase accuracy (dominant phase matching)
    pred_phases = np.argmax(predictions, axis=1)
    actual_phases = np.argmax(y_test_data.values, axis=1)
    phase_accuracy = np.mean(pred_phases == actual_phases)
    
    # Calculate per-phase MAE
    phase_mae = np.mean(np.abs(predictions - y_test_data.values), axis=0)
    
    return {
        'model': model,
        'predictions': predictions,
        'mae': mae,
        'mse': mse,
        'phase_accuracy': phase_accuracy,
        'phase_mae': phase_mae,
        'pred_phases': pred_phases,
        'actual_phases': actual_phases
    }

# Define ML models to test
models = {
    'XGBoost': {
        'model': MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )),
        'use_scaled': False,
        'description': 'Gradient boosting with extreme optimization'
    },
    
    'Random Forest': {
        'model': MultiOutputRegressor(RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        'use_scaled': False,
        'description': 'Ensemble of decision trees with bagging'
    },
    
    'LightGBM': {
        'model': MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )),
        'use_scaled': False,
        'description': 'Microsoft\'s gradient boosting framework'
    },
    
    'Ridge Regression': {
        'model': MultiOutputRegressor(Ridge(
            alpha=1.0,
            random_state=42
        )),
        'use_scaled': True,
        'description': 'Linear regression with L2 regularization'
    },
    
    'Elastic Net': {
        'model': MultiOutputRegressor(ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42,
            max_iter=2000
        )),
        'use_scaled': True,
        'description': 'Linear regression with L1 + L2 regularization'
    },
    
    'Support Vector Regression': {
        'model': MultiOutputRegressor(SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            epsilon=0.01
        )),
        'use_scaled': True,
        'description': 'Non-linear regression using support vectors'
    },
    
    'K-Nearest Neighbors': {
        'model': MultiOutputRegressor(KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='euclidean'
        )),
        'use_scaled': True,
        'description': 'Instance-based learning with local averaging'
    }
}

# Train and evaluate all models
results = {}
phase_names = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']

for name, config in models.items():
    # Choose scaled or unscaled data
    X_train_data = X_train_scaled if config['use_scaled'] else X_train
    X_test_data = X_test_scaled if config['use_scaled'] else X_test
    
    # Evaluate model
    result = evaluate_model(
        config['model'], 
        X_train_data, 
        X_test_data, 
        y_train, 
        y_test, 
        name
    )
    
    # Add configuration info
    result['description'] = config['description']
    result['use_scaled'] = config['use_scaled']
    
    results[name] = result
    
    # Print immediate results
    print(f"   ✅ Phase Accuracy: {result['phase_accuracy']:.1%}")
    print(f"   📊 Overall MAE: {result['mae']:.4f}")

# Create ensemble of top 3 models
print(f"\n🤝 Creating Ensemble Model...")
top_3_models = sorted(results.items(), key=lambda x: x[1]['phase_accuracy'], reverse=True)[:3]
top_3_names = [name for name, _ in top_3_models]

ensemble_estimators = []
for name in top_3_names:
    config = models[name]
    if config['use_scaled']:
        # For scaled models, we need to create a pipeline-like approach
        # For simplicity, we'll use the unscaled version
        if name == 'Ridge Regression':
            ensemble_estimators.append((name.replace(' ', '_').lower(), Ridge(alpha=1.0, random_state=42)))
        elif name == 'Support Vector Regression':
            ensemble_estimators.append((name.replace(' ', '_').lower(), SVR(kernel='linear', C=0.1)))
        elif name == 'K-Nearest Neighbors':
            ensemble_estimators.append((name.replace(' ', '_').lower(), KNeighborsRegressor(n_neighbors=7)))
    else:
        if name == 'XGBoost':
            ensemble_estimators.append((name.lower(), xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
        elif name == 'Random Forest':
            ensemble_estimators.append((name.replace(' ', '_').lower(), RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
        elif name == 'LightGBM':
            ensemble_estimators.append((name.lower(), lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))

if len(ensemble_estimators) >= 2:
    ensemble_model = MultiOutputRegressor(VotingRegressor(
        estimators=ensemble_estimators,
        n_jobs=-1
    ))
    
    ensemble_result = evaluate_model(
        ensemble_model,
        X_train,  # Use unscaled for ensemble
        X_test,
        y_train,
        y_test,
        f"Ensemble ({', '.join(top_3_names)})"
    )
    
    ensemble_result['description'] = f"Voting ensemble of top 3 models: {', '.join(top_3_names)}"
    ensemble_result['use_scaled'] = False
    results['Ensemble'] = ensemble_result
    
    print(f"   ✅ Ensemble Phase Accuracy: {ensemble_result['phase_accuracy']:.1%}")
    print(f"   📊 Ensemble Overall MAE: {ensemble_result['mae']:.4f}")

# Create comprehensive analysis
print(f"\n{'='*80}")
print("📋 COMPREHENSIVE MODEL ANALYSIS")
print(f"{'='*80}")

# Sort results by phase accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['phase_accuracy'], reverse=True)

# Create comparison table
comparison_data = []
for name, result in sorted_results:
    comparison_data.append({
        'Model': name,
        'Phase Accuracy': f"{result['phase_accuracy']:.1%}",
        'Overall MAE': f"{result['mae']:.4f}",
        'Overall MSE': f"{result['mse']:.4f}",
        'Uses Scaling': 'Yes' if result.get('use_scaled', False) else 'No',
        'Description': result['description']
    })

results_df = pd.DataFrame(comparison_data)
print(results_df.to_string(index=False))

# Detailed analysis for each model
print(f"\n{'='*80}")
print("🔍 DETAILED MODEL ANALYSIS")
print(f"{'='*80}")

for name, result in sorted_results:
    print(f"\n📊 {name.upper()}")
    print("-" * 50)
    print(f"Description: {result['description']}")
    print(f"Data Preprocessing: {'Scaled features' if result.get('use_scaled', False) else 'Raw features'}")
    print(f"Phase Accuracy: {result['phase_accuracy']:.1%}")
    print(f"Overall MAE: {result['mae']:.4f}")
    print(f"Overall MSE: {result['mse']:.4f}")
    
    # Per-phase analysis
    print(f"Per-Phase MAE:")
    for i, phase_name in enumerate(phase_names):
        print(f"  {phase_name}: {result['phase_mae'][i]:.4f}")
    
    # Phase distribution analysis
    unique_pred, pred_counts = np.unique(result['pred_phases'], return_counts=True)
    unique_actual, actual_counts = np.unique(result['actual_phases'], return_counts=True)
    
    print(f"Predicted phase distribution:")
    for i, count in enumerate(pred_counts):
        phase_idx = unique_pred[i] if i < len(unique_pred) else i
        print(f"  Phase {phase_idx + 1}: {count} samples ({count/len(result['pred_phases']):.1%})")
    
    # Strengths and weaknesses
    if name == 'XGBoost':
        print("💪 Strengths: Excellent with tabular data, handles complex patterns, built-in regularization")
        print("⚠️ Considerations: Can overfit with very small datasets, requires parameter tuning")
    elif name == 'Random Forest':
        print("💪 Strengths: Robust to overfitting, handles missing values, provides feature importance")
        print("⚠️ Considerations: Can be memory intensive, less interpretable than single trees")
    elif name == 'LightGBM':
        print("💪 Strengths: Fast training, memory efficient, excellent accuracy")
        print("⚠️ Considerations: Can overfit with small datasets, newer algorithm")
    elif name == 'Ridge Regression':
        print("💪 Strengths: Simple, fast, never overfits, highly interpretable")
        print("⚠️ Considerations: Assumes linear relationships, may underfit complex patterns")
    elif name == 'Elastic Net':
        print("💪 Strengths: Automatic feature selection, handles multicollinearity")
        print("⚠️ Considerations: Requires parameter tuning, assumes linear relationships")
    elif name == 'Support Vector Regression':
        print("💪 Strengths: Works well in high dimensions, memory efficient")
        print("⚠️ Considerations: Sensitive to feature scaling, parameter selection important")
    elif name == 'K-Nearest Neighbors':
        print("💪 Strengths: Simple, no assumptions about data distribution, good with local patterns")
        print("⚠️ Considerations: Sensitive to curse of dimensionality, computationally expensive")
    elif name == 'Ensemble':
        print("💪 Strengths: Combines multiple models, typically best performance, reduces overfitting")
        print("⚠️ Considerations: More complex, slower prediction, less interpretable")

# Final recommendations
print(f"\n{'='*80}")
print("🏆 FINAL RECOMMENDATIONS")
print(f"{'='*80}")

best_model = sorted_results[0]
second_best = sorted_results[1] if len(sorted_results) > 1 else None

print(f"🥇 WINNER: {best_model[0]}")
print(f"   Phase Accuracy: {best_model[1]['phase_accuracy']:.1%}")
print(f"   Overall MAE: {best_model[1]['mae']:.4f}")
print(f"   Why it won: {best_model[1]['description']}")

if second_best:
    print(f"\n🥈 RUNNER-UP: {second_best[0]}")
    print(f"   Phase Accuracy: {second_best[1]['phase_accuracy']:.1%}")
    print(f"   Overall MAE: {second_best[1]['mae']:.4f}")

# Compare with Neural Network (assuming ~70% from previous results)
nn_baseline = 0.698  # From your previous results
best_accuracy = best_model[1]['phase_accuracy']

print(f"\n📈 COMPARISON WITH NEURAL NETWORK:")
print(f"   Neural Network: ~{nn_baseline:.1%}")
print(f"   Best ML Method: {best_accuracy:.1%}")
improvement = (best_accuracy - nn_baseline) / nn_baseline * 100
if improvement > 0:
    print(f"   Improvement: +{improvement:.1f}% better! 🚀")
else:
    print(f"   Neural Network performs {abs(improvement):.1f}% better")

print(f"\n🎯 FOR YOUR HACKATHON:")
print(f"   1. Lead with: {best_model[0]} ({best_model[1]['phase_accuracy']:.1%} accuracy)")
print(f"   2. Mention: Tested 8 different ML approaches")
print(f"   3. Highlight: {best_model[1]['description']}")
print(f"   4. Compare: Neural networks vs traditional ML on small data")

print(f"\n🛠️ IMPLEMENTATION RECOMMENDATION:")
if best_model[1]['phase_accuracy'] > 0.75:
    print("   ✅ Excellent performance! Ready for production.")
elif best_model[1]['phase_accuracy'] > 0.65:
    print("   ✅ Good performance! Consider ensemble for production.")
else:
    print("   📊 Moderate performance. Consider more data or feature engineering.")

print(f"\n🚀 Next steps: Focus on {best_model[0]} for your final model!")
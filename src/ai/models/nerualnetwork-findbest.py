import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping #type: ignore

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and prepare data
df = pd.read_csv('src/ai/train_predict.csv')

# Separate features (X) and targets (y)
feature_cols = [col for col in df.columns if col.startswith('feature_band_')]
target_cols = [col for col in df.columns if col.startswith('target_avg_pop_phase_')]

X = df[feature_cols]
y = df[target_cols]

# The targets are already percentages (0-1), but we'll ensure they're floats
y_floats = y.astype(float)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_floats, test_size=0.1, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Scale features for better training
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define different model architectures and configurations
model_configs = [
    {
        'name': 'Tiny (565 params)',
        'layers': [8],
        'dropout': 0.4,
        'l2_reg': 0.01,
        'loss': 'mse',
        'params_est': 565
    },
    {
        'name': 'Small (1125 params)', 
        'layers': [16],
        'dropout': 0.3,
        'l2_reg': 0.01,
        'loss': 'mse',
        'params_est': 1125
    },
    {
        'name': 'Medium (2149 params)',
        'layers': [32],
        'dropout': 0.2,
        'l2_reg': 0.005,
        'loss': 'mse',
        'params_est': 2149
    },
    {
        'name': 'Two Layer (1441 params)',
        'layers': [16, 8],
        'dropout': 0.3,
        'l2_reg': 0.01,
        'loss': 'mse',
        'params_est': 1441
    },
    # Test different loss functions with best architecture
    {
        'name': 'Small + MAE Loss',
        'layers': [16],
        'dropout': 0.3,
        'l2_reg': 0.01,
        'loss': 'mae',
        'params_est': 1125
    },
    {
        'name': 'Small + Huber Loss',
        'layers': [16],
        'dropout': 0.3,
        'l2_reg': 0.01,
        'loss': 'huber',
        'params_est': 1125
    },
    {
        'name': 'Small + Categorical CE',
        'layers': [16],
        'dropout': 0.3,
        'l2_reg': 0.01,
        'loss': 'categorical_crossentropy',
        'params_est': 1125
    }
]

def create_model(config):
    """Create a model based on configuration"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(64,)))
    
    # Add hidden layers
    for i, units in enumerate(config['layers']):
        model.add(tf.keras.layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg'])
        ))
        model.add(tf.keras.layers.Dropout(config['dropout']))
    
    # Output layer
    if config['loss'] == 'categorical_crossentropy':
        activation = 'softmax'
    else:
        activation = 'linear'  # For regression
        
    model.add(tf.keras.layers.Dense(5, activation=activation))
    
    return model

def train_and_evaluate_model_extended(config):
    """Train and evaluate a single model configuration with variable epochs and batch size"""
    
    model = create_model(config)
    
    # Compile model
    if config['loss'] == 'categorical_crossentropy':
        metrics = ['accuracy']
    else:
        metrics = ['mean_absolute_error']
    
    model.compile(
        optimizer='adam',
        loss=config['loss'],
        metrics=metrics
    )
    
    actual_params = model.count_params()
    
    # Early stopping with patience based on epochs
    patience = max(3, config['epochs'] // 10)  # Dynamic patience
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model with specified epochs and batch size
    history = model.fit(
        X_train_scaled, y_train,
        epochs=config['epochs'],
        validation_split=0.15,
        batch_size=config['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    test_loss = test_results[0]
    test_metric = test_results[1] if len(test_results) > 1 else None
    
    # Make predictions
    predictions = model.predict(X_test_scaled, verbose=0)
    
    # Calculate additional metrics
    mae_manual = np.mean(np.abs(predictions - y_test.values))
    mse_manual = np.mean((predictions - y_test.values) ** 2)
    
    # Calculate phase accuracy (how often the highest predicted phase matches actual)
    pred_phases = np.argmax(predictions, axis=1)
    actual_phases = np.argmax(y_test.values, axis=1)
    phase_accuracy = np.mean(pred_phases == actual_phases)
    
    return {
        'config': config,
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_metric': test_metric,
        'mae_manual': mae_manual,
        'mse_manual': mse_manual,
        'phase_accuracy': phase_accuracy,
        'predictions': predictions,
        'actual_params': actual_params,
        'epochs_trained': len(history.history['loss'])
    }

# Define epoch and batch size variations
epoch_options = [5, 20, 50, 100]
batch_size_options = [16, 32, 64, 128, 256]

# Train all model combinations
results = []
total_combinations = len(model_configs) * len(epoch_options) * len(batch_size_options)
current_combo = 0

print(f"\n{'='*80}")
print(f"TRAINING {total_combinations} MODEL COMBINATIONS")
print(f"{'='*80}")

for config in model_configs:
    for epochs in epoch_options:
        for batch_size in batch_size_options:
            current_combo += 1
            
            # Update config with current epoch and batch settings
            config_copy = config.copy()
            config_copy['epochs'] = epochs
            config_copy['batch_size'] = batch_size
            config_copy['combo_name'] = f"{config['name']} | E{epochs} | B{batch_size}"
            
            print(f"\n[{current_combo:3d}/{total_combinations}] Training: {config_copy['combo_name']}")
            
            result = train_and_evaluate_model_extended(config_copy)
            results.append(result)
            
            # Print quick result
            print(f"  ✅ Phase Accuracy: {result['phase_accuracy']:.3f} | MAE: {result['mae_manual']:.4f} | Epochs: {result['epochs_trained']}")

# Create results DataFrame
results_df = pd.DataFrame([
    {
        'Model': r['config']['name'],
        'Epochs': r['config']['epochs'],
        'Batch Size': r['config']['batch_size'],
        'Parameters': r['actual_params'],
        'Test Loss': r['test_loss'],
        'MAE': r['mae_manual'],
        'MSE': r['mse_manual'],
        'Phase Accuracy': r['phase_accuracy'],
        'Epochs Trained': r['epochs_trained'],
        'Loss Function': r['config']['loss'],
        'Dropout': r['config']['dropout'],
        'L2 Reg': r['config']['l2_reg'],
        'Combo Name': r['config']['combo_name']
    } for r in results
])

print(f"\n{'='*80}")
print("ALL COMBINATIONS RESULTS (sorted by Phase Accuracy)")
print(f"{'='*80}")
results_sorted = results_df.sort_values('Phase Accuracy', ascending=False)
print(results_sorted[['Combo Name', 'Phase Accuracy', 'MAE', 'Epochs Trained']].round(4))

print(f"\n{'='*80}")
print("TOP 10 BEST PERFORMING COMBINATIONS")
print(f"{'='*80}")
top_10 = results_sorted.head(10)
for idx, row in top_10.iterrows():
    print(f"{row['Combo Name']}")
    print(f"  Phase Accuracy: {row['Phase Accuracy']:.4f} | MAE: {row['MAE']:.4f} | Epochs Trained: {row['Epochs Trained']}")
    print()

print(f"\n{'='*80}")
print("EPOCH ANALYSIS - Best accuracy by epoch setting")
print(f"{'='*80}")
epoch_analysis = results_df.groupby('Epochs')['Phase Accuracy'].agg(['mean', 'max', 'std']).round(4)
print(epoch_analysis)

print(f"\n{'='*80}")
print("BATCH SIZE ANALYSIS - Best accuracy by batch size")
print(f"{'='*80}")
batch_analysis = results_df.groupby('Batch Size')['Phase Accuracy'].agg(['mean', 'max', 'std']).round(4)
print(batch_analysis)

# Find overall best model
best_model_idx = results_df['Phase Accuracy'].idxmax()
best_model = results[best_model_idx]

print(f"\n{'='*80}")
print("OVERALL BEST MODEL ANALYSIS")
print(f"{'='*80}")
print(f"Best Combination: {best_model['config']['combo_name']}")
print(f"Parameters: {best_model['actual_params']:,}")
print(f"Phase Accuracy: {best_model['phase_accuracy']:.4f}")
print(f"MAE: {best_model['mae_manual']:.4f}")
print(f"MSE: {best_model['mse_manual']:.4f}")
print(f"Epochs Trained: {best_model['epochs_trained']}/{best_model['config']['epochs']}")

print(f"\n{'='*80}")
print("HYPERPARAMETER INSIGHTS")
print(f"{'='*80}")

# Best epoch setting
best_epoch = results_df.loc[results_df['Phase Accuracy'].idxmax(), 'Epochs']
print(f"🎯 Best epoch setting: {best_epoch}")

# Best batch size setting  
best_batch = results_df.loc[results_df['Phase Accuracy'].idxmax(), 'Batch Size']
print(f"🎯 Best batch size: {best_batch}")

# Best model architecture
best_arch = results_df.loc[results_df['Phase Accuracy'].idxmax(), 'Model']
print(f"🎯 Best architecture: {best_arch}")

# Best loss function
best_loss = results_df.loc[results_df['Phase Accuracy'].idxmax(), 'Loss Function']
print(f"🎯 Best loss function: {best_loss}")

print(f"\n{'='*80}")
print("TRAINING EFFICIENCY ANALYSIS")
print(f"{'='*80}")

# Models that achieved >70% accuracy
high_performers = results_df[results_df['Phase Accuracy'] > 0.7]
if len(high_performers) > 0:
    print(f"📈 {len(high_performers)} combinations achieved >70% phase accuracy")
    print(f"   Fastest to train: {high_performers.loc[high_performers['Epochs Trained'].idxmin(), 'Combo Name']}")
    print(f"   (Trained in {high_performers['Epochs Trained'].min()} epochs)")
else:
    print("📊 No combinations achieved >70% phase accuracy")

# Early stopping analysis
early_stopped = results_df[results_df['Epochs Trained'] < results_df['Epochs']]
print(f"⏰ {len(early_stopped)}/{len(results_df)} combinations stopped early")

print(f"\n{'='*80}")
print("FINAL RECOMMENDATIONS")
print(f"{'='*80}")
print(f"🏆 Use: {best_model['config']['combo_name']}")
print(f"📊 Expected Performance: {best_model['phase_accuracy']:.1%} phase accuracy")
print(f"⚡ Training Time: ~{best_model['epochs_trained']} epochs")
print(f"🎛️  Optimal Settings:")
print(f"    - Architecture: {best_arch}")
print(f"    - Loss Function: {best_loss}")
print(f"    - Epochs: {best_epoch}")
print(f"    - Batch Size: {best_batch}")

if best_model['phase_accuracy'] > 0.7:
    print("✅ Excellent performance for 1k dataset!")
elif best_model['phase_accuracy'] > 0.5:
    print("✅ Good performance - model is learning meaningful patterns")
else:
    print("⚠️  Consider alternative approaches (Random Forest, XGBoost)")

print(f"\n{'='*50}")
print("EXPERIMENT COMPLETE")
print(f"{'='*50}")
print(f"Total combinations tested: {len(results_df)}")
print(f"Best phase accuracy achieved: {results_df['Phase Accuracy'].max():.4f}")
print(f"Average phase accuracy: {results_df['Phase Accuracy'].mean():.4f}")
print(f"Standard deviation: {results_df['Phase Accuracy'].std():.4f}")
from training_defs import load_data, train_model


#Loading training, validating, testing data sets
train_loader, val_loader, test_loader, scaler = load_data(
    csv_path="simpler_data_rwc.csv",
    test_size=0.2,    # 20% for validation
    batch_size=32     # Process 32 samples at a time
    )
dataset_size = len(train_loader) + len(val_loader)+len(test_loader)
print(f"Training: {len(train_loader)/dataset_size * 100:.2f}%")
print(f"Validating: {len(val_loader)/dataset_size * 100:.2f}%")
print(f"Testing: {len(test_loader)/dataset_size * 100:.2f}%")

import matplotlib.pyplot as plt
from training_defs import train_model

# Use your current parameters - the logic is now fixed!
model, scaler, losses = train_model(
    input_size=210,      # Match your data
    hidden_size=64,      # Your current size
    learning_rate=0.01, # Your current rate
    num_epochs=200,       # More epochs since learning was broken before
    patience=30
)

plt.plot(losses['train'], label='Training Loss')
plt.plot(losses['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Cell 3: Comprehensive Model Evaluation
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import MLP
from training_defs import load_data

# Load the trained model
model = MLP(input_size=210, hidden_size=64, dropout_rate=0.2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Get test data (IMPORTANT: Use test set, not validation!)
_, _, test_loader, _ = load_data()  # Get test_loader

# Collect all predictions and targets
all_predictions = []
all_targets = []

with torch.no_grad():
    for X, y in test_loader:
        pred = model(X)
        all_predictions.append(pred.numpy())
        all_targets.append(y.numpy())

# Concatenate all batches
y_pred = np.concatenate(all_predictions, axis=0)
y_true = np.concatenate(all_targets, axis=0)

# Calculate metrics for each vegetation fraction
fraction_names = ['Green Vegetation (GV)', 'Non-Photosynthetic Vegetation (NPV)', 'Soil']

print("📊 Model Evaluation Results:")
print("=" * 50)

for i, name in enumerate(fraction_names):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    
    print(f"\n{name}:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

# Overall metrics (averaged across all fractions)
overall_mse = mean_squared_error(y_true, y_pred)
overall_mae = mean_absolute_error(y_true, y_pred)
overall_r2 = r2_score(y_true, y_pred)

print(f"\n{'Overall (All Fractions)':}")
print(f"  MSE:  {overall_mse:.4f}")
print(f"  MAE:  {overall_mae:.4f}")
print(f"  R²:   {overall_r2:.4f}")

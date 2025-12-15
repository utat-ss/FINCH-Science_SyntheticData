import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from model import MLP


class HybridLoss(nn.Module):
    """
    Hybrid loss function combining MSE loss with abundance sum constraint.
    
    Loss = MSE(pred, target) + lambda * |1 - sum(pred)|
    
    Args:
        lambda_weight (float): Weight for the abundance sum constraint term
    """
    def __init__(self, lambda_weight=1.0):
        super(HybridLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # Standard MSE loss
        mse_term = self.mse_loss(pred, target)
        
        # Abundance sum constraint: |1 - sum(pred)|
        abundance_sum = torch.sum(pred, dim=1)  # Sum across the 3 fractions for each sample
        constraint_term = torch.mean(torch.abs(1.0 - abundance_sum))
        
        # Combined loss
        total_loss = mse_term + self.lambda_weight * constraint_term
        
        return total_loss


def load_data(csv_path="simpler_data_rwc.csv", test_size=0.2, batch_size=32):
    """Load and prepare the data for training 
    - CSV file: Path of the csv data file 

    Returns 
    """

    #Load dataset 
    df = pd.read_csv(csv_path)

    metadata_cols = ['Spectra', 'gv_fraction', 'npv_fraction', 'soil_fraction', 
                'RWC index', 'Calculated RWC', 'use']
    
    # Get wavelength columns (everything that's not metadata)
    wavelength_cols = [col for col in df.columns if col not in metadata_cols]

    fraction_cols = ['gv_fraction', 'npv_fraction', 'soil_fraction']

    #Extract features and targets
    x = df[wavelength_cols].values
    y = df[fraction_cols].values

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42) 

    # Normalize spectral features (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader, scaler


def train_model(input_size=210, hidden_sizes=[64], dropout_rate=0.2, learning_rate=0.001, 
                num_epochs=50, patience=10, lambda_weight=1.0):
    """
    Simplified MLP training for vegetation fraction prediction.
    
    Args:
        input_size: Number of spectral bands (default: 210 for 400-2490nm)
        hidden_sizes: List of hidden layer sizes (e.g., [128, 128] for 128-128-3)
        dropout_rate: Dropout regularization
        learning_rate: Adam learning rate
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        lambda_weight: Weight for abundance sum constraint in hybrid loss
    
    Returns:
        model: Trained MLP
        scaler: Fitted StandardScaler
        losses: Training history
    """
    
    # Load and prepare data
    train_loader, val_loader, test_loader, scaler = load_data()
    
    # Create model (matches MLP architecture)
    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HybridLoss(lambda_weight=lambda_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"Training MLP: {model.get_architecture()}")
    
    best_loss = float('inf')
    patience_count = 0
    losses = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for X, y in train_loader:  # ← Iterate through batches properly
            optimizer.zero_grad()  # ← Clear gradients
            pred = model(X)        # ← Forward pass
            loss = criterion(pred, y)  # ← Compute loss
            loss.backward()        # ← Backpropagation 
            optimizer.step()       # ← Update weights
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)  # ← Average over batches
        
        # Validation phase
        model.eval()
        val_loss = sum(criterion(model(X), y) for X, y in val_loader) / len(val_loader)
        
        # To this:
        losses['train'].append(train_loss)         # ✅ Correct (train_loss is float)
        losses['val'].append(val_loss.item()) 
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Progress logging - now shows each epoch
        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(scaler, 'scaler.pth')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, scaler, losses
#!/usr/bin/env python3
"""
Supervised learning trainer using expert demonstration data.
Trains a neural network to mimic successful racing behavior.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ExpertDrivingDataset(Dataset):
    """Dataset for expert driving demonstrations."""
    
    def __init__(self, csv_files, normalize=True):
        """Load and preprocess expert driving data."""
        print(f"Loading data from {len(csv_files)} CSV files...")
        
        # Load all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"Loaded {len(df)} samples from {os.path.basename(csv_file)}")
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError("No valid CSV files found!")
        
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Total samples: {len(self.data)}")
        
        # Extract features and targets
        # Actions (targets): ACCELERATION, BRAKE, STEERING
        self.actions = self.data[['ACCELERATION', 'BRAKE', 'STEERING']].values.astype(np.float32)
        
        # Sensors (features): SPEED, TRACK_POSITION, ANGLE_TO_TRACK_AXIS, 18x TRACK_EDGE_*
        sensor_columns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS'] + \
                        [f'TRACK_EDGE_{i}' for i in range(18)]
        self.sensors = self.data[sensor_columns].values.astype(np.float32)
        
        # Data validation
        print(f"Sensor shape: {self.sensors.shape}")
        print(f"Action shape: {self.actions.shape}")
        
        # Remove invalid samples
        valid_mask = np.isfinite(self.sensors).all(axis=1) & np.isfinite(self.actions).all(axis=1)
        self.sensors = self.sensors[valid_mask]
        self.actions = self.actions[valid_mask]
        print(f"Valid samples after filtering: {len(self.sensors)}")
        
        # Normalize sensors if requested
        if normalize:
            self.scaler = StandardScaler()
            self.sensors = self.scaler.fit_transform(self.sensors)
            print("Sensors normalized using StandardScaler")
        else:
            self.scaler = None
        
        # Data statistics
        print("\n=== Data Statistics ===")
        print(f"Speed range: {self.data['SPEED'].min():.1f} to {self.data['SPEED'].max():.1f}")
        print(f"Track position range: {self.data['TRACK_POSITION'].min():.3f} to {self.data['TRACK_POSITION'].max():.3f}")
        print(f"Steering range: {self.data['STEERING'].min():.3f} to {self.data['STEERING'].max():.3f}")
        print(f"Acceleration mean: {self.data['ACCELERATION'].mean():.3f}")
        print(f"Brake mean: {self.data['BRAKE'].mean():.3f}")
    
    def __len__(self):
        return len(self.sensors)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sensors[idx]), torch.FloatTensor(self.actions[idx])


class ExpertDrivingNet(nn.Module):
    """Neural network for imitating expert driving behavior."""
    
    def __init__(self, input_size=21, hidden_size=256, dropout=0.2):
        super(ExpertDrivingNet, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Separate heads for each action
        self.acceleration_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        self.brake_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output [-1, 1]
        )
    
    def forward(self, x):
        features = self.shared(x)
        
        accel = self.acceleration_head(features)
        brake = self.brake_head(features)
        steering = self.steering_head(features)
        
        return torch.cat([accel, brake, steering], dim=1)


def train_model(train_loader, val_loader, model, num_epochs=1000, lr=0.001):
    """Train the expert driving model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Custom loss function with different weights for each action
    def expert_loss(predictions, targets):
        accel_loss = nn.MSELoss()(predictions[:, 0], targets[:, 0])
        brake_loss = nn.MSELoss()(predictions[:, 1], targets[:, 1])
        steering_loss = nn.MSELoss()(predictions[:, 2], targets[:, 2])
        
        # Weight steering more heavily (most important for control)
        return accel_loss + brake_loss + 2.0 * steering_loss
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_sensors, batch_actions in train_loader:
            batch_sensors, batch_actions = batch_sensors.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_sensors)
            loss = expert_loss(predictions, batch_actions)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_sensors, batch_actions in val_loader:
                batch_sensors, batch_actions = batch_sensors.to(device), batch_actions.to(device)
                predictions = model(batch_sensors)
                loss = expert_loss(predictions, batch_actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/expert_driving_best.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
    
    return train_losses, val_losses


def evaluate_model(model, val_loader, scaler=None):
    """Evaluate the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_sensors, batch_actions in val_loader:
            batch_sensors = batch_sensors.to(device)
            predictions = model(batch_sensors).cpu().numpy()
            targets = batch_actions.numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse_accel = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    mse_brake = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    mse_steering = np.mean((predictions[:, 2] - targets[:, 2]) ** 2)
    
    print(f"\n=== Model Evaluation ===")
    print(f"Acceleration MSE: {mse_accel:.6f}")
    print(f"Brake MSE: {mse_brake:.6f}")
    print(f"Steering MSE: {mse_steering:.6f}")
    
    # Steering accuracy (most important)
    steering_mae = np.mean(np.abs(predictions[:, 2] - targets[:, 2]))
    print(f"Steering MAE: {steering_mae:.6f}")
    
    return {
        'accel_mse': mse_accel,
        'brake_mse': mse_brake,
        'steering_mse': mse_steering,
        'steering_mae': steering_mae
    }


def main():
    """Main training function."""
    print("=== Expert Driving Supervised Learning ===")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Find all CSV files in train_data
    csv_pattern = "train_data/train_data/*.csv"
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")
    
    # Load and prepare data
    dataset = ExpertDrivingDataset(csv_files, normalize=True)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    input_size = dataset.sensors.shape[1]  # Should be 21
    model = ExpertDrivingNet(input_size=input_size)
    
    print(f"\nModel architecture:")
    print(f"Input size: {input_size}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    train_losses, val_losses = train_model(train_loader, val_loader, model, num_epochs=200)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('models/expert_driving_best.pth'))
    metrics = evaluate_model(model, val_loader, dataset.scaler)
    
    # Save final model and scaler
    torch.save(model.state_dict(), 'models/expert_driving_final.pth')
    if dataset.scaler:
        import joblib
        joblib.dump(dataset.scaler, 'models/expert_scaler.pkl')
        print("Saved scaler for inference")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Plot last 50 epochs for detail
    start_idx = max(0, len(train_losses) - 50)
    plt.plot(train_losses[start_idx:], label='Training Loss')
    plt.plot(val_losses[start_idx:], label='Validation Loss')
    plt.title('Training Progress (Last 50 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_progress.png', dpi=150, bbox_inches='tight')
    print("Saved training progress plot to models/training_progress.png")
    
    print(f"\n✅ Training completed!")
    print(f"Best model saved to: models/expert_driving_best.pth")
    print(f"Final model saved to: models/expert_driving_final.pth")


if __name__ == "__main__":
    main()

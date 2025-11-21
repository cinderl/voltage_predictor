import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from math import pi
import matplotlib.pyplot as plt

# 1. Configuration
FILE_NAME = './models/history.csv'
MODEL_PATH = './models/time_value_model.pth'
STATS_PATH = './models/normalization_stats.npz'
BATCH_SIZE = 64
LEARNING_RATE = 0.001 #0.00005 #0.0001 #0.001
EPOCHS = 500
TIME_WINDOW_MINUTES = 30  # Time window for aggregating minimum values
FRAME_SIZE = 5 #10  # Number of best epochs to track
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Must be identical in both scripts) ---
class TimeValuePredictor(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_rate=0.2):
        super(TimeValuePredictor, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, output_size)
        # )
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added Dropout
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added Dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )        
    def forward(self, x):
        return self.net(x)
# -----------------------------------------------------------

# 2. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    print("[INFO] Loading data from:", file_path)
    df = pd.read_csv(file_path, usecols=['state', 'last_changed'])
    print(f"[INFO] Loaded {len(df)} rows.")
    # Remove rows where state is undefined or missing
    df = df[df['state'].notna() & (df['state'] != 'unavailable')].copy()
    print(f"[INFO] Rows after filtering undefined state: {len(df)}")
    df['value'] = df['state'].astype(float)
    print("[INFO] Parsing 'last_changed' timestamps.")
    df['last_changed'] = pd.to_datetime(df['last_changed'], utc=True)
    df = df.dropna()
    print(f"[INFO] Data after dropping NA: {len(df)} rows.")
    # After parsing timestamps and filtering, keep only the lowest value per time window
    df['window_key'] = df['last_changed'].dt.floor(f'{TIME_WINDOW_MINUTES}min').dt.strftime('%Y-%m-%d %H:%M')
    df = df.loc[df.groupby('window_key')['value'].idxmin()].reset_index(drop=True)
    print(f"[INFO] Data after keeping lowest value per {TIME_WINDOW_MINUTES}-minute window: {len(df)} rows.")
    # Extract hour and minute
    print("[INFO] Extracting hour and minute from timestamps.")
    df['hour'] = df['last_changed'].dt.hour
    df['minute'] = df['last_changed'].dt.minute
    # Extract month (datetime .month uses 1..12 where January = 1)
    print("[INFO] Extracting month from timestamps.")
    df['month'] = df['last_changed'].dt.month  # January=1, December=12
    # Extract weekday (Monday=0 .. Sunday=6)
    print("[INFO] Extracting weekday from timestamps.")
    df['weekday'] = df['last_changed'].dt.weekday
    # Cyclical Encoding
    print("[INFO] Applying cyclical encoding to hour and minute.")
    df['hour_sin'] = np.sin(2 * pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * pi * df['minute'] / 60)
    # Cyclical encoding for weekday
    print("[INFO] Applying cyclical encoding to weekday.")
    df['weekday_sin'] = np.sin(2 * pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * pi * df['weekday'] / 7)
    # Cyclical encoding for month (1..12 -> period 12)
    print("[INFO] Applying cyclical encoding to month.")
    df['month_sin'] = np.sin(2 * pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * pi * (df['month'] - 1) / 12)
    X = df[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']].values.astype(np.float32)
    Y = df['value'].values.astype(np.float32).reshape(-1, 1)
    print("[INFO] Calculating normalization statistics.")
    value_mean = Y.mean()
    value_std = Y.std()
    np.savez(STATS_PATH, mean=value_mean, std=value_std)
    print(f"[INFO] Saved normalization stats to {STATS_PATH}.")
    Y_normalized = (Y - value_mean) / value_std
    print("[INFO] Splitting data into train and validation sets.")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_normalized, test_size=0.2, random_state=42)
    print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, Y_train, Y_val

# 3. Custom PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 4. Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device, patience=10):
    print(f"--- Starting Training on {device} ---")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = model.state_dict()
    counter = 0  # Counter for early stopping
    best_epochs_frame = []  # Store the epoch numbers of the 10 best validation losses
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0  # Reset counter
            best_epochs_frame.append(epoch + 1)
            if len(best_epochs_frame) > FRAME_SIZE:
                best_epochs_frame.pop(0)
            print(f'[INFO] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Best)')
        else:
            counter += 1
            if counter >= patience:
                print(f'[INFO] Early stopping triggered after {epoch+1} epochs')
                break
            if (epoch + 1) % 10 == 0:
                print(f'[INFO] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"--- Training Complete (Best Val Loss: {best_val_loss:.4f}) ---")
    print(f"[INFO] Frame of 10 best epochs: {best_epochs_frame}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./models/training_loss.png')
    plt.close()
    print(f"[INFO] Training progress plot saved to ./models/training_loss.png")
    
    return train_losses, val_losses

# --- Main Execution ---
if __name__ == "__main__":
    print("[INFO] Starting model training script.")
    try:
        X_train, X_val, Y_train, Y_val = load_and_preprocess_data(FILE_NAME)
        print("[INFO] Initializing datasets and dataloaders.")
        train_dataset = TimeSeriesDataset(X_train, Y_train)
        val_dataset = TimeSeriesDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        print("[INFO] Initializing model.")
        input_size = X_train.shape[1]
        model = TimeValuePredictor(input_size=input_size).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(train_loader, val_loader, model, criterion, optimizer, EPOCHS, DEVICE)
        print(f"[INFO] Saving trained model to {MODEL_PATH}.")
        torch.save(model.state_dict(), MODEL_PATH)
        print("[INFO] Model training and saving complete.")
    except Exception as e:
        print(f"[ERROR] {e}")
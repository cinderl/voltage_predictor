import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from math import pi
import os
import matplotlib.pyplot as plt

# Configuration
FILE_NAME = './models/history.csv'
PROB_MODEL_PATH = './models/probability_model.pth'
PROB_STATS_PATH = './models/probability_stats.npz'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_WINDOW_MINUTES = 30
LAGS = 0


class TimeValuePredictor(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_rate=0.2):
        super(TimeValuePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.net(x)


def load_and_preprocess_data(file_path, low_threshold=210.0, high_threshold=240.0):
    print('[INFO] Loading data from:', file_path)
    df = pd.read_csv(file_path, usecols=['state', 'last_changed'])
    df = df[df['state'].notna() & (df['state'] != 'unavailable')].copy()
    df['value'] = df['state'].astype(float)
    df['last_changed'] = pd.to_datetime(df['last_changed'], utc=True)
    df = df.dropna()
    # Keep lowest value per window as original script
    df['window_key'] = df['last_changed'].dt.floor(f'{TIME_WINDOW_MINUTES}min').dt.strftime('%Y-%m-%d %H:%M')
    df = df.loc[df.groupby('window_key')['value'].idxmin()].reset_index(drop=True)

    df['hour'] = df['last_changed'].dt.hour
    df['minute'] = df['last_changed'].dt.minute
    # Extract month
    df['month'] = df['last_changed'].dt.month
    df['weekday'] = df['last_changed'].dt.weekday

    # cyclical encodings
    df['hour_sin'] = np.sin(2 * pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * pi * df['minute'] / 60)
    df['weekday_sin'] = np.sin(2 * pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * pi * df['weekday'] / 7)
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * pi * (df['month'] - 1) / 12)

    X = df[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']].values.astype(np.float32)
    # Add lag columns if needed (default 0 features)
    if LAGS > 0:
        lags = np.zeros((len(X), LAGS), dtype=np.float32)
        X = np.concatenate([X, lags], axis=1)

    # Binary label for anomaly: fall (<= low_threshold) or peak (>= high_threshold)
    Y = ((df['value'] <= low_threshold) | (df['value'] >= high_threshold)).astype(np.float32).values.reshape(-1, 1)

    print('[INFO] Saving probability model thresholds to:', PROB_STATS_PATH)
    np.savez(PROB_STATS_PATH, low_threshold=low_threshold, high_threshold=high_threshold)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    print(f'[INFO] Train samples: {len(X_train)}, Val samples: {len(X_val)}')
    return X_train, X_val, Y_train, Y_val


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device, patience=10):
    best_val = float('inf')
    best_state = model.state_dict()
    counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
            counter = 0
            print(f'[INFO] Epoch {epoch+1}/{epochs} - Train: {avg_train:.4f} Val: {avg_val:.4f} (best)')
        else:
            counter += 1
            if counter >= patience:
                print(f'[INFO] Early stopping at epoch {epoch+1}')
                break
            if (epoch+1) % 20 == 0:
                print(f'[INFO] Epoch {epoch+1}/{epochs} - Train: {avg_train:.4f} Val: {avg_val:.4f}')

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


if __name__ == '__main__':
    try:
        X_train, X_val, Y_train, Y_val = load_and_preprocess_data(FILE_NAME)
        train_dataset = TimeSeriesDataset(X_train, Y_train)
        val_dataset = TimeSeriesDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        input_size = X_train.shape[1]
        print('[INFO] Initializing probability model with input size', input_size)
        model = TimeValuePredictor(input_size=input_size, output_size=1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model, train_losses, val_losses = train_model(train_loader, val_loader, model, criterion, optimizer, EPOCHS, DEVICE)
        print('[INFO] Saving probability model to', PROB_MODEL_PATH)
        torch.save(model.state_dict(), PROB_MODEL_PATH)

        # Ensure models directory exists and save training loss plot
        os.makedirs(os.path.dirname(PROB_MODEL_PATH), exist_ok=True)
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss (Probability Model)')
            plt.legend()
            plot_path = './models/training_loss_probability.png'
            plt.savefig(plot_path)
            plt.close()
            print('[INFO] Saved training/validation loss plot to', plot_path)
        except Exception as e:
            print('[WARN] Could not save plot:', e)

        print('[INFO] Done.')
    except Exception as e:
        print('[ERROR]', e)

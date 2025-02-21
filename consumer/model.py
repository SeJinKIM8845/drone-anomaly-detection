#!/usr/bin/env python3
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from filter import FilterPreprocessor  # 전처리 모듈을 통해 데이터를 생성

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, output_dim, output_window_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window_size = output_window_size
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_window_size * output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, input_window_size, hidden_size)
        last_out = out[:, -1, :]         # (batch, hidden_size)
        pred = self.fc(last_out)         # (batch, output_window_size * output_dim)
        pred = pred.view(batch_size, self.output_window_size, -1)
        return pred

def load_config(config_path="model_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (batch_X, batch_Y) in enumerate(train_loader):
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        if torch.isnan(loss):
            print(f"NaN loss encountered in batch {i}. Skipping this batch.")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_samples = 0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            if torch.isnan(loss):
                continue
            running_loss += loss.item() * batch_X.size(0)
            valid_samples += batch_X.size(0)
    epoch_loss = running_loss / valid_samples if valid_samples > 0 else float('nan')
    return epoch_loss

def main():
    config = load_config("model_config.yaml")
    start_time = config["influxdb"]["start_time"]
    stop_time = config["influxdb"]["stop_time"]
    test_split = config["model"]["test_split"]
    batch_size = config["model"]["batch_size"]
    num_epochs = config["model"]["num_epochs"]
    learning_rate = config["model"]["learning_rate"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    model_save_path = config["model"]["model_save_path"]
    input_window_size = config["model"]["input_window_size"]
    output_window_size = config["model"]["output_window_size"]
    
    # 데이터 전처리: FilterPreprocessor를 이용하여 학습용 데이터셋(X, Y) 생성
    preprocessor = FilterPreprocessor(config_path="filter_config.yaml")
    simulation_ids = preprocessor.get_simulation_ids(days=-30)
    if not simulation_ids:
        print("No simulation IDs found.")
        return
    X, Y = preprocessor.filter_batch(simulation_ids, start_time, stop_time,
                                     input_size=input_window_size, output_size=output_window_size)
    if X is None or Y is None:
        print("Failed to load preprocessed data.")
        return
    print(f"Preprocessed training data shapes: X: {X.shape}, Y: {Y.shape}")
    
    # 데이터에 nan 값이 있는지 확인
    print("X 내 nan 개수:", np.isnan(X).sum())
    print("Y 내 nan 개수:", np.isnan(Y).sum())
    
    # 데이터셋 구성 및 Train/Test 분할
    dataset = TimeSeriesDataset(X, Y)
    total_samples = len(dataset)
    test_samples = int(total_samples * test_split)
    train_samples = total_samples - test_samples
    train_dataset, test_dataset = random_split(dataset, [train_samples, test_samples])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 입력 피처 차원 및 타깃 차원 결정
    input_dim = X.shape[2]      # 예: 센서 피처 수 (23)
    target_dim = Y.shape[2]     # 예: 타깃 차원 (3: position_x, position_y, position_z)
    
    model = LSTMModel(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                      dropout=dropout, output_dim=target_dim, output_window_size=output_window_size)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with test loss {test_loss:.6f}")
    
    print("Training complete.")
    
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, test_losses, label="Test Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            predictions = model(batch_X)  
            pred_sample = predictions[0].cpu().numpy()
            target_sample = batch_Y[0].cpu().numpy()
            break
    
    t = np.arange(output_window_size)
    plt.figure(figsize=(12, 8))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.subplot(3, 1, i+1)
        plt.plot(t, target_sample[:, i], label=f"Target {axis}", marker='o')
        plt.plot(t, pred_sample[:, i], label=f"Prediction {axis}", marker='s')
        plt.xlabel("Time Step")
        plt.ylabel(f"{axis} Value")
        plt.title(f"Prediction vs Target for {axis} over {output_window_size} Time Steps")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
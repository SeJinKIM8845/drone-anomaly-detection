import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import logging
from datetime import datetime

class DronePathDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
        # 타겟은 각 시퀀스의 마지막 타임스텝에서 local_pose의 position_x, position_y, position_z (인덱스 10, 11, 12)
        self.targets = self.sequences[:, -1, 10:13]

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class DronePathLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(DronePathLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # 최종 출력: (position_x, position_y, position_z)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 마지막 LSTM 레이어의 마지막 hidden state 사용
        final_output = self.fc(h_n[-1])
        return final_output

class DronePathPredictor:
    def __init__(self, config_path="model_config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self._setup_training_components()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        log_dir = Path(self.config['logging']['dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"model_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
        self.logger = logging.getLogger(__name__)

    def _create_model(self):
        model = DronePathLSTM(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        return model

    def _setup_training_components(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])

    def prepare_data(self):
        data_path = Path(self.config['data']['processed_data_path'])
        sequences_file = data_path / "batch_sequences.npy"
        if not sequences_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {sequences_file}")
        sequences = np.load(sequences_file)
        self.logger.info(f"Loaded sequences shape: {sequences.shape}")
        # Dataset: 각 샘플은 (sequence, target) 형태로 구성
        dataset = DronePathDataset(sequences)
        total_samples = len(dataset)
        train_size = int(0.8 * total_samples)
        val_size = total_samples - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.model.train()
            running_loss = 0.0
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * sequences.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            
            val_loss = self.evaluate(val_loader)
            self.logger.info(f"Epoch {epoch}/{self.config['training']['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * sequences.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        return avg_loss

    def save_model(self):
        save_dir = Path(self.config['model']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")

def main():
    predictor = DronePathPredictor()
    train_loader, val_loader = predictor.prepare_data()
    predictor.train(train_loader, val_loader)

if __name__ == "__main__":
    main()

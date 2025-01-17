#!/usr/bin/env python3
import os
import json
import time  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm 
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from datetime import datetime
import yaml
from filter import DroneDataFilter

class DroneDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: Optional[np.ndarray] = None):
        """Initialize dataset with sequences and optional targets"""
        self.sequences = torch.FloatTensor(sequences)
        if targets is not None:
            self.targets = torch.FloatTensor(targets).reshape(-1, 1)
            print(f"Targets shape: {self.targets.shape}")
        else:
            self.targets = None
        print(f"Dataset initialized - Sequences shape: {self.sequences.shape}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]

class DroneAnomalyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Input shape to forward: {x.shape}")
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context)
        
        return output

class DroneAnomalyDetector:
    def __init__(self, config_path: str = "model_config.yaml"):
        """Initialize the anomaly detector with configuration"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filter = DroneDataFilter()
        self.model = self._create_model()
        self._setup_training_components()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['logging']['dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"model_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_model(self) -> nn.Module:
        """Create and initialize the LSTM model"""
        model = DroneAnomalyLSTM(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        self.logger.info(f"Model created on device: {self.device}")
        return model

    def _setup_training_components(self):
        """Setup loss function and optimizer"""
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training and testing"""
        data_path = Path(self.config['data']['processed_data_path'])
        sequences = np.load(data_path / 'batch_sequences.npy')
        
        labels = self._create_anomaly_labels(sequences)
        
        train_size = int(0.8 * len(sequences))
        test_size = len(sequences) - train_size
        
        train_dataset, test_dataset = random_split(
            DroneDataset(sequences, labels),
            [train_size, test_size]
        )
        
        print(f"Training data shape: {len(train_dataset)}")
        print(f"Test data shape: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_loader, test_loader

    def _create_anomaly_labels(self, sequences: np.ndarray) -> np.ndarray:
        """Create binary labels for anomaly detection"""
        with open(Path(self.config['data']['scaler_params_path']), 'r') as f:
            scaler_params = json.load(f)
        
        labels = np.zeros(len(sequences))
        
        for i, sequence in enumerate(sequences):
            # IMU 데이터의 급격한 변화 확인
            imu_changes = np.abs(np.diff(sequence[:, :6], axis=0))  # First 6 features are IMU
            if np.any(imu_changes > float(self.config['anomaly']['imu_threshold'])):
                labels[i] = 1
                
            # Position 데이터의 급격한 변화 확인
            position_changes = np.abs(np.diff(sequence[:, 10:13], axis=0))  # Position features
            if np.any(position_changes > float(self.config['anomaly']['position_threshold'])):
                labels[i] = 1
            
            # Orientation 데이터의 급격한 변화 확인
            orientation_changes = np.abs(np.diff(sequence[:, 13:16], axis=0))  # Orientation features
            if np.any(orientation_changes > float(self.config['anomaly']['orientation_threshold'])):
                labels[i] = 1
        
        print(f"Created labels - Total sequences: {len(labels)}, Anomalies detected: {np.sum(labels)}")
        return labels

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train the model and return training history"""
        best_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'epoch_times': []
        }
        
        print("\n=== Starting Model Training ===")
        print(f"Total epochs: {self.config['training']['epochs']}")
        print(f"Training device: {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            # Training phase
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print("Training phase:")
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_sequences, batch_targets in progress_bar:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = self.criterion(outputs, batch_targets)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss/batch_count:.4f}'
                })
            
            # Testing phase
            test_loss, accuracy = self.evaluate(test_loader)
            epoch_time = time.time() - epoch_start_time
            
            # Save metrics
            history['train_loss'].append(train_loss/len(train_loader))
            history['test_loss'].append(test_loss)
            history['accuracy'].append(accuracy)
            history['epoch_times'].append(epoch_time)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Average Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")
            
            # Early stopping check
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                self.save_model()
                print("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['patience']:
                    print("\nEarly stopping triggered!")
                    break
        
        print("\n=== Training Complete ===")
        print(f"Best Test Loss: {best_loss:.4f}")
        
        # Save training history
        history_path = Path(self.config['model']['save_dir']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f)
        print(f"Training history saved to {history_path}")
        
        return history

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on test data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                # Sigmoid 적용 후 임계값 비교
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.numel()
        
        return total_loss / len(test_loader), correct / total

    def save_model(self):
        """Save the trained model"""
        save_dir = Path(self.config['model']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_dir / 'best_model.pth')
        
        self.logger.info(f"Model saved to {save_dir / 'best_model.pth'}")

def main():
    """Main function for training the model"""
    try:
        print("\n=== Drone Anomaly Detection Model Training ===")
        
        # Initialize detector
        print("\nInitializing model...")
        detector = DroneAnomalyDetector()
        
        # Print model architecture
        print("\nModel Architecture:")
        print(detector.model)
        
        # Data preparation
        print("\nPreparing data...")
        train_loader, test_loader = detector.prepare_data()
        
        # Print data information
        total_train = len(train_loader.dataset)
        total_test = len(test_loader.dataset)
        print(f"\nData Summary:")
        print(f"Training samples: {total_train}")
        print(f"Test samples: {total_test}")
        print(f"Batch size: {detector.config['training']['batch_size']}")
        print(f"Steps per epoch: {len(train_loader)}")
        
        # Train model
        history = detector.train(train_loader, test_loader)
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Total epochs completed: {len(history['train_loss'])}")
        print(f"Best accuracy: {max(history['accuracy']):.4f}")
        print(f"Final test loss: {history['test_loss'][-1]:.4f}")
        print(f"Average epoch time: {np.mean(history['epoch_times']):.2f} seconds")
        
        # Save results summary
        results_summary = {
            'total_epochs': len(history['train_loss']),
            'best_accuracy': max(history['accuracy']),
            'final_test_loss': history['test_loss'][-1],
            'avg_epoch_time': np.mean(history['epoch_times'])
        }
        
        summary_path = Path(detector.config['model']['save_dir']) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print(f"\nResults saved to {summary_path}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
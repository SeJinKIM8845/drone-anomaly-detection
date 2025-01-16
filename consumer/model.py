import os
import json
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

class DroneDataset(Dataset):
    """드론 데이터를 위한 커스텀 데이터셋"""
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        # 마지막 타임스텝의 값을 정상/비정상 레이블로 사용
        target = sequence[-1]
        return sequence[:-1], target

class DroneLSTM(nn.Module):
    """드론 이상 탐지를 위한 LSTM 모델"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(DroneLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Bidirectional LSTM이므로 hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM 레이어
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 출력만 사용
        last_output = lstm_out[:, -1, :]
        
        # 완전연결 레이어
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
class ModelTrainer:
    """드론 LSTM 모델 학습 및 평가를 위한 클래스"""
    def __init__(self, config: Dict):
        """
        초기화 함수
        Args:
            config: 설정 파라미터가 포함된 딕셔너리
        """
        self.config = config
        self.setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # 모델 하이퍼파라미터 설정
        self.input_size = config['model']['input_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.sequence_length = config['data']['sequence_length']
        self.batch_size = config['training']['batch_size']
        
        # 모델 초기화
        self.model = DroneLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 손실 함수와 옵티마이저 설정
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )

    def setup_logging(self) -> None:
        """로깅 설정"""
        log_dir = self.config['logging']['dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(
            log_dir,
            f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=self.config['logging']['level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, data: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        데이터 준비 및 로더 생성
        Args:
            data: 입력 데이터 배열
        Returns:
            train_loader, test_loader
        """
        # 학습/테스트 데이터 분할
        train_data, test_data = train_test_split(
            data,
            test_size=self.config['data']['test_size'],
            random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = DroneDataset(train_data, self.sequence_length)
        test_dataset = DroneDataset(test_data, self.sequence_length)
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader

    def train_model(self, train_loader: DataLoader) -> List[float]:
        """
        모델 학습
        Args:
            train_loader: 학습 데이터 로더
        Returns:
            학습 손실 리스트
        """
        self.model.train()
        losses = []
        
        for epoch in range(self.config['training']['epochs']):
            epoch_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{self.config["training"]["epochs"]}], Loss: {avg_loss:.4f}')
        
        return losses

    def evaluate_model(self, test_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        모델 평가
        Args:
            test_loader: 테스트 데이터 로더
        Returns:
            테스트 손실, 예측값, 실제값
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        return avg_loss, np.array(predictions), np.array(actuals)

    def save_model(self) -> None:
        """학습된 모델 저장"""
        save_dir = self.config['model']['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        model_path = os.path.join(save_dir, 'drone_lstm_model.pth')
        config_path = os.path.join(save_dir, 'model_config.json')
        
        # 모델 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        
        # 설정 저장
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """
        저장된 모델 로드
        Args:
            model_path: 모델 파일 경로
        """
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Model loaded from {model_path}")
        else:
            self.logger.error(f"No model file found at {model_path}")
            raise FileNotFoundError(f"No model file found at {model_path}")
    
if __name__ == "__main__":
    # 설정 파일 로드
    try:
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

    # 모델 트레이너 초기화 및 실행
    try:
        trainer = ModelTrainer(config)
        
        # 여기에서 데이터 로드 및 전처리 과정이 필요합니다
        # processed_data는 Filter.py에서 전처리된 데이터를 사용
        
        train_loader, test_loader = trainer.prepare_data(processed_data)
        
        # 모델 학습
        losses = trainer.train_model(train_loader)
        
        # 모델 평가
        test_loss, predictions, actuals = trainer.evaluate_model(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        
        # 모델 저장
        trainer.save_model()
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise
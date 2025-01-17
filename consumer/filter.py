#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import MinMaxScaler
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats

class DroneDataFilter:
   def __init__(self, config_path: str = "filter_config.yaml"):
       """Initialize DroneDataFilter with configuration"""
       self.config = self._load_config(config_path)
       self.setup_paths()
       self.setup_logging()
       self.client = self._setup_influxdb()
       self.scalers = {}

   def setup_paths(self):
       """Setup necessary directories"""
       for path_name, path in self.config['paths'].items():
           path_obj = Path(path)
           path_obj.mkdir(parents=True, exist_ok=True)
           setattr(self, f"{path_name}_path", path_obj)

   def setup_logging(self):
       """Setup logging configuration"""
       log_file = Path(self.config['paths']['base_dir']) / f'filter_{datetime.now():%Y%m%d_%H%M%S}.log'
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(levelname)s - %(message)s',
           handlers=[
               logging.StreamHandler(),
               logging.FileHandler(log_file)
           ]
       )
       self.logger = logging.getLogger(__name__)

   def _load_config(self, config_path: str) -> dict:
       """Load configuration from yaml file"""
       try:
           with open(config_path, 'r') as f:
               return yaml.safe_load(f)
       except Exception as e:
           raise RuntimeError(f"Failed to load configuration: {e}")

   def _setup_influxdb(self) -> InfluxDBClient:
       """Setup InfluxDB client"""
       return InfluxDBClient(
           url=self.config['influxdb']['url'],
           token=self.config['influxdb']['token'],
           org=self.config['influxdb']['org']
       )

   def get_sensor_data(self, sensor_type: str, sensor_category: str) -> pd.DataFrame:
        """Get data for specific sensor type"""
        self.logger.info(f"Fetching data for {sensor_type} from category {sensor_category}")
        
        query = f'''
        from(bucket: "{self.config['influxdb']['bucket']}")
            |> range(start: -30d)
            |> filter(fn: (r) => r["_measurement"] == "drone_{sensor_type}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            result = self.client.query_api().query_data_frame(query)
            if isinstance(result, list):
                result = pd.concat(result)
            
            if not result.empty:
                features = self.config['sensors'][sensor_category][sensor_type]['features']
                required_columns = ['_time', 'simulation_id'] + features
                available_columns = [col for col in required_columns if col in result.columns]
                result = result[available_columns]
                
                self.logger.info(f"Loaded {len(result)} rows for {sensor_type}")
                self.logger.info(f"Available columns: {result.columns.tolist()}")
                
                if len(result) == 0:
                    self.logger.warning(f"No data found for {sensor_type}")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error fetching {sensor_type} data: {e}")
            raise

   def get_all_sensor_data(self) -> Dict[str, pd.DataFrame]:
       """Get data for all sensors"""
       all_data = {}
       
       # Primary sensors
       for sensor_type in self.config['sensors']['primary'].keys():
           data = self.get_sensor_data(sensor_type, 'primary')
           if not data.empty:
               all_data[sensor_type] = data
               
       # Secondary sensors
       for sensor_type in self.config['sensors']['secondary'].keys():
           data = self.get_sensor_data(sensor_type, 'secondary')
           if not data.empty:
               all_data[sensor_type] = data
               
       return all_data

   def filter_batch(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
       """Process data for training"""
       self.logger.info("Starting batch data processing")
       
       # 1. 결측치 처리
       clean_data = self._handle_missing_values(data)
       
       # 2. 데이터 정규화
       normalized_data, scaler_params = self._normalize_data(clean_data)
       
       # 3. 시퀀스 생성
       sequences = self._create_sequences(normalized_data)
       
       # 4. 데이터 셔플링
       shuffled_sequences = self._shuffle_data(sequences)
       
       # 5. 저장
       self._save_processed_data(shuffled_sequences, 'batch')
       self._save_scaler_params(scaler_params)
       
       return shuffled_sequences, scaler_params

   def filter_inf(self, data: Dict[str, pd.DataFrame], scaler_params: Dict) -> np.ndarray:
       """Process data for inference"""
       self.logger.info("Starting inference data processing")
       
       # 1. 결측치 처리
       clean_data = self._handle_missing_values(data)
       
       # 2. 저장된 파라미터로 정규화
       normalized_data = self._apply_normalization(clean_data, scaler_params)
       
       # 3. 시퀀스 생성
       sequences = self._create_sequences(normalized_data)
       
       return sequences

   def _handle_missing_values(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
       """Handle missing values in data"""
       clean_data = {}
       max_gap = self.config['preprocessing']['missing_values']['max_gap']
       
       for sensor_type, df in data.items():
           if sensor_type in self.config['sensors']['primary']:
               # 주요 센서는 선형 보간
               df_clean = df.interpolate(
                   method=self.config['preprocessing']['missing_values']['primary_method'],
                   limit=max_gap
               )
           else:
               # 보조 센서는 전방 채우기
               df_clean = df.fillna(
                   method=self.config['preprocessing']['missing_values']['secondary_method'],
                   limit=max_gap
               )
           
           # 남은 결측치는 0으로 채우기
           df_clean = df_clean.fillna(0)
           clean_data[sensor_type] = df_clean
       
       return clean_data

   def _normalize_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Normalize data and store scaling parameters"""
    normalized_data = {}
    scaler_params = {}
    
    # feature_range를 tuple로 변환
    feature_range = tuple(self.config['preprocessing']['normalization']['feature_range'])
    
    for sensor_type, df in data.items():
        features = self.config['sensors']['primary'].get(sensor_type, {}).get('features', []) or \
                  self.config['sensors']['secondary'].get(sensor_type, {}).get('features', [])
        
        if features:
            scaler = MinMaxScaler(feature_range=feature_range)  # tuple로 전달
            normalized_values = scaler.fit_transform(df[features])
            normalized_df = pd.DataFrame(
                normalized_values,
                columns=features,
                index=df.index
            )
            
            normalized_data[sensor_type] = normalized_df
            scaler_params[sensor_type] = {
                'min': scaler.data_min_.tolist(),
                'max': scaler.data_max_.tolist(),
                'feature_names': features
            }
    
    return normalized_data, scaler_params

   def _create_sequences(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create sequences for LSTM"""
        sequence_length = self.config['preprocessing']['sequence_length']
        stride = self.config['preprocessing']['stride']
        
        # 모든 특성 결합
        all_features = []
        for sensor_type in self.config['lstm_data']['feature_order']:
            if sensor_type in data:
                self.logger.info(f"Adding features from {sensor_type}: {data[sensor_type].shape}")
                all_features.append(data[sensor_type])
        
        if not all_features:
            raise ValueError("No features available for sequence creation")
        
        combined_data = pd.concat(all_features, axis=1)
        self.logger.info(f"Combined data shape: {combined_data.shape}")
        
        # 시퀀스 생성
        sequences = []
        for i in range(0, len(combined_data) - sequence_length, stride):
            sequence = combined_data.iloc[i:i + sequence_length].values
            sequences.append(sequence)
            
        sequences = np.array(sequences)
        self.logger.info(f"Created sequences shape: {sequences.shape}")
        
        return sequences

   def _shuffle_data(self, sequences: np.ndarray) -> np.ndarray:
       """Shuffle sequences randomly"""
       indices = np.arange(sequences.shape[0])
       np.random.shuffle(indices)
       shuffled_sequences = sequences[indices]
       
       self.logger.info(f"Data shuffled: {sequences.shape[0]} sequences")
       
       return shuffled_sequences

   def _save_processed_data(self, sequences: np.ndarray, mode: str):
       """Save processed data"""
       save_path = Path(self.config['paths']['base_dir'])
       
       # numpy 형식으로 저장
       np.save(save_path / f'{mode}_sequences.npy', sequences)
       
       # CSV 형식으로 저장 (LSTM용)
       reshaped_data = sequences.reshape(-1, sequences.shape[-1])
       feature_names = self._get_feature_names()
       
       pd.DataFrame(reshaped_data, columns=feature_names).to_csv(
           save_path / f'{mode}_data.csv',
           index=False
       )

   def _save_scaler_params(self, scaler_params: Dict):
       """Save scaling parameters"""
       save_path = Path(self.config['paths']['base_dir']) / self.config['paths']['scaler_params']
       with open(save_path, 'w') as f:
           json.dump(scaler_params, f, indent=2)

   def _get_feature_names(self) -> List[str]:
       """Get list of all feature names"""
       feature_names = []
       for sensor_type in self.config['lstm_data']['feature_order']:
           if sensor_type.endswith('_features'):
               sensor_type = sensor_type[:-9]  # Remove '_features' suffix
           if sensor_type in self.config['sensors']['primary']:
               feature_names.extend(self.config['sensors']['primary'][sensor_type]['features'])
           elif sensor_type in self.config['sensors']['secondary']:
               feature_names.extend(self.config['sensors']['secondary'][sensor_type]['features'])
       return feature_names

   def _apply_normalization(self, data: Dict[str, pd.DataFrame], scaler_params: Dict) -> Dict[str, pd.DataFrame]:
       """Apply saved normalization parameters to new data"""
       normalized_data = {}
       
       for sensor_type, df in data.items():
           if sensor_type in scaler_params:
               params = scaler_params[sensor_type]
               features = params['feature_names']
               
               # 스케일러 재생성
               scaler = MinMaxScaler()
               scaler.min_ = np.array(params['min'])
               scaler.scale_ = (np.array(params['max']) - scaler.min_)
               
               # 정규화 적용
               normalized_values = scaler.transform(df[features])
               normalized_df = pd.DataFrame(
                   normalized_values,
                   columns=features,
                   index=df.index
               )
               
               normalized_data[sensor_type] = normalized_df
       
       return normalized_data

   def cleanup(self):
       """Cleanup resources"""
       if hasattr(self, 'client'):
           self.client.close()
       self.logger.info("Resources cleaned up")

if __name__ == "__main__":
   try:
       filter = DroneDataFilter()
       data = filter.get_all_sensor_data()
       sequences, scaler_params = filter.filter_batch(data)
       filter.cleanup()
   except Exception as e:
       logging.error(f"Error in main execution: {e}", exc_info=True)
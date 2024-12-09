#!/usr/bin/env python3

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class DroneDataFilter:
    def __init__(self, config_path: str = 'filter_config.yaml'):
        """드론 데이터 전처리 클래스 초기화"""
        self.load_config(config_path)
        self.setup_logging()
        self.setup_influxdb()
        self.setup_scalers()
        
    def load_config(self, config_path: str) -> None:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 주요 파라미터 설정
        self.bucket = self.config['influxdb']['bucket']
        self.sampling_rate = self.config['preprocessing']['sampling_rate']
        self.chunk_size = self.config['preprocessing']['chunk_size']
        self.sequence_length = self.config['preprocessing']['sequence_length']
        
    def setup_logging(self) -> None:
        """로깅 설정"""
        log_dir = Path(self.config['logging']['directory'])
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'filter_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(log_file))
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_influxdb(self) -> None:
        """InfluxDB 연결 설정"""
        try:
            self.client = InfluxDBClient(
                url=self.config['influxdb']['url'],
                token=self.config['influxdb']['token'],
                org=self.config['influxdb']['org']
            )
            self.query_api = self.client.query_api()
            self.logger.info("Successfully connected to InfluxDB")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            raise
            
    def setup_scalers(self) -> None:
        """데이터 스케일러 초기화"""
        self.scaler = MinMaxScaler()
        self.logger.info("Initialized MinMaxScaler")

    def fetch_data_chunk(self, measurement: str, start_time: str, end_time: str) -> pd.DataFrame:
        """지정된 시간 범위의 데이터 청크 조회"""
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            '''
            df = self.query_api.query_data_frame(query)
            if df.empty:
                self.logger.warning(f"No data found for {measurement} between {start_time} and {end_time}")
                return pd.DataFrame()
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data chunk: {str(e)}")
            raise

    def extract_vibration_features(self, imu_data: pd.DataFrame) -> pd.DataFrame:
        """진동 관련 특성 추출"""
        if imu_data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=imu_data.index)
        
        # 각속도와 가속도 magnitude 계산
        angular_vel = np.sqrt(
            imu_data['angular_velocity_x']**2 +
            imu_data['angular_velocity_y']**2 +
            imu_data['angular_velocity_z']**2
        )
        
        accel = np.sqrt(
            imu_data['linear_acceleration_x']**2 +
            imu_data['linear_acceleration_y']**2 +
            imu_data['linear_acceleration_z']**2
        )
        
        window = self.config['preprocessing']['features']['vibration']['window_size']
        
        # 기본 통계 특성
        features['angular_vel_mean'] = angular_vel.rolling(window=window, min_periods=1).mean()
        features['angular_vel_std'] = angular_vel.rolling(window=window, min_periods=1).std()
        features['accel_mean'] = accel.rolling(window=window, min_periods=1).mean()
        features['accel_std'] = accel.rolling(window=window, min_periods=1).std()
        
        # 주파수 도메인 특성 계산
        if self.config['preprocessing']['features']['vibration']['use_fft']:
            freqs, psd = signal.welch(angular_vel.fillna(0), fs=self.sampling_rate)
            features['dominant_freq'] = freqs[np.argmax(psd)]
            features['spectral_energy'] = np.sum(psd)
            
        return features

    def extract_battery_features(self, battery_data: pd.DataFrame) -> pd.DataFrame:
        """배터리 관련 특성 추출"""
        if battery_data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=battery_data.index)
        window = self.config['preprocessing']['features']['battery']['window_size']
        
        # 기본 특성
        features['voltage'] = battery_data['voltage']
        features['current'] = battery_data['current']
        features['power'] = battery_data['voltage'] * battery_data['current']
        
        # 변화율 특성
        features['voltage_change'] = battery_data['voltage'].diff()
        features['current_change'] = battery_data['current'].diff()
        
        # 통계 특성
        features['voltage_ma'] = battery_data['voltage'].rolling(window=window, min_periods=1).mean()
        features['voltage_std'] = battery_data['voltage'].rolling(window=window, min_periods=1).std()
        features['current_ma'] = battery_data['current'].rolling(window=window, min_periods=1).mean()
        features['current_std'] = battery_data['current'].rolling(window=window, min_periods=1).std()
        
        return features

    def preprocess_chunk(self, imu_data: pd.DataFrame, battery_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """데이터 청크 전처리"""
        # 특성 추출
        imu_features = self.extract_vibration_features(imu_data)
        battery_features = self.extract_battery_features(battery_data)
        
        # 특성 통합
        all_features = pd.concat([imu_features, battery_features], axis=1)
        
        # 결측치 처리
        all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 정규화
        scaled_data = self.scaler.fit_transform(all_features)
        
        return scaled_data, all_features.columns.tolist()

    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """LSTM 시퀀스 생성"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i+self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def process_time_range(self, start_time: str, end_time: str) -> None:
        """시간 범위 데이터 처리"""
        try:
            self.logger.info(f"Processing data from {start_time} to {end_time}")
            current_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time)
            
            all_sequences = []
            feature_names = None
            
            # 청크 단위로 처리
            while current_time < end_time:
                next_time = min(
                    current_time + pd.Timedelta(seconds=self.chunk_size/self.sampling_rate),
                    end_time
                )
                
                # 청크 데이터 조회
                imu_data = self.fetch_data_chunk("drone_imu", 
                    current_time.isoformat(), 
                    next_time.isoformat()
                )
                battery_data = self.fetch_data_chunk("drone_battery",
                    current_time.isoformat(),
                    next_time.isoformat()
                )
                
                # 데이터 처리
                if not (imu_data.empty or battery_data.empty):
                    scaled_data, feature_names = self.preprocess_chunk(imu_data, battery_data)
                    sequences = self.create_sequences(scaled_data)
                    all_sequences.append(sequences)
                
                current_time = next_time
                self.logger.info(f"Processed chunk until {current_time}")
            
            if all_sequences:
                # 모든 시퀀스 통합
                final_sequences = np.concatenate(all_sequences, axis=0)
                
                # 결과 저장
                self.save_processed_data(final_sequences, feature_names)
                self.logger.info("Data processing completed successfully")
            else:
                self.logger.warning("No data was processed")
                
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def save_processed_data(self, sequences: np.ndarray, feature_names: List[str]) -> None:
        """처리된 데이터 저장"""
        try:
            output_dir = Path(self.config['output']['directory'])
            output_dir.mkdir(exist_ok=True)
            
            # 시퀀스 데이터 저장
            np.save(output_dir / 'sequences.npy', sequences)
            
            # 특성 이름 저장
            with open(output_dir / 'feature_names.json', 'w') as f:
                json.dump(feature_names, f)
            
            # 스케일러 파라미터 저장
            scaler_params = {
                'data_min_': self.scaler.data_min_.tolist(),
                'data_max_': self.scaler.data_max_.tolist()
            }
            with open(output_dir / 'scaler_params.json', 'w') as f:
                json.dump(scaler_params, f)
                
            self.logger.info(f"Saved processed data to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            self.client.close()
            self.logger.info("Cleaned up resources")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    filter = DroneDataFilter('filter_config.yaml')
    
    try:
        # 최근 7일간의 데이터 처리
        start_time = (datetime.now() - timedelta(days=7)).isoformat()
        end_time = datetime.now().isoformat()
        
        filter.process_time_range(start_time, end_time)
    except Exception as e:
        logging.error(f"Failed to process data: {str(e)}")
    finally:
        filter.cleanup()
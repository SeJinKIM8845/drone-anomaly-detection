import os
import json
import yaml
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DroneDataFilter:
    def __init__(self, config: Dict):
        """
        Initialize DroneDataFilter with configuration and setup necessary components
        """
        self.config = config
        self.setup_logging()
        self.setup_influxdb_client()
        
        # 스케일러 초기화
        self.scaler_params = {}
        self.scalers = {
            'imu': MinMaxScaler(feature_range=(-1, 1)),
            'path': MinMaxScaler(feature_range=(-1, 1))
        }
        
        # IMU 데이터 필드 정의
        self.imu_fields = [
            'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
        ]
        
        # 경로 데이터 필드 정의
        self.path_fields = [
            'position_x', 'position_y', 'position_z',
            'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'
        ]
        
        # 이상 상황 감지를 위한 임계값 설정
        self.anomaly_thresholds = {
            'angular_velocity': {
                'max_change': 1.0,  # 초당 최대 변화율
                'range': (-3.0, 3.0)  # 허용 범위
            },
            'linear_acceleration': {
                'max_change': 2.0,
                'range': (-10.0, 10.0)
            },
            'position': {
                'max_deviation': 5.0  # 예상 경로로부터 최대 허용 편차
            }
        }

        # 결측치 처리 설정
        self.missing_value_config = {
            'max_consecutive': 5,
            'interpolation_method': 'linear',
            'statistical_method': 'mean'
        }

    def setup_logging(self) -> None:
        """
        Configure logging settings
        """
        log_dir = self.config.get('logging', {}).get('dir', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(
            log_dir,
            f'filter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=self.config.get('logging', {}).get('level', 'INFO'),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_influxdb_client(self) -> None:
        """
        Initialize InfluxDB client connection
        """
        try:
            self.influx_client = InfluxDBClient(
                url=self.config['influxdb']['url'],
                token=self.config['influxdb']['token'],
                org=self.config['influxdb']['org']
            )
            self.query_api = self.influx_client.query_api()
            self.logger.info("InfluxDB client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup InfluxDB client: {e}")
            raise
    
    def get_data_from_influxdb(self, start_time: str = "-30d") -> pd.DataFrame:
        """
        Retrieve and preprocess drone data from InfluxDB
        """
        try:
            # IMU 데이터 쿼리
            imu_query = f'''
                from(bucket: "{self.config['influxdb']['bucket']}")
                    |> range(start: {start_time})
                    |> filter(fn: (r) => r["_measurement"] == "drone_imu")
                    |> pivot(rowKey:["_time", "simulation_id"], 
                            columnKey: ["_field"], 
                            valueColumn: "_value")
            '''
            
            # 경로 데이터 쿼리
            path_query = f'''
                from(bucket: "{self.config['influxdb']['bucket']}")
                    |> range(start: {start_time})
                    |> filter(fn: (r) => r["_measurement"] == "drone_local_pose")
                    |> pivot(rowKey:["_time", "simulation_id"], 
                            columnKey: ["_field"], 
                            valueColumn: "_value")
            '''

            # 쿼리 실행
            imu_result = self.query_api.query_data_frame(imu_query)
            path_result = self.query_api.query_data_frame(path_query)

            # 필요한 컬럼만 선택
            imu_df = imu_result[['_time', 'simulation_id', 
                                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                                'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z']]
            
            path_df = path_result[['_time', 'simulation_id',
                                'position_x', 'position_y', 'position_z']]

            # 컬럼명 변경
            imu_df = imu_df.rename(columns={'_time': 'timestamp'})
            path_df = path_df.rename(columns={'_time': 'timestamp'})

            # 데이터 병합
            merged_df = pd.merge_asof(
                imu_df.sort_values('timestamp'),
                path_df.sort_values('timestamp'),
                on='timestamp',
                by='simulation_id',
                tolerance=pd.Timedelta('100ms')
            )

            # 데이터 타입 확인 로그
            self.logger.info(f"Retrieved {len(merged_df)} records from InfluxDB")
            self.logger.info(f"Data distribution across simulations: {merged_df['simulation_id'].value_counts().to_dict()}")
            self.logger.info(f"Data types of columns:\n{merged_df.dtypes}")

            return merged_df

        except Exception as e:
            self.logger.error(f"Error retrieving data from InfluxDB: {e}")
            raise

    def filter_batch(self, start_time: str = "-30d") -> Tuple[pd.DataFrame, Dict]:
       """
       Process batch data for model training with comprehensive data filtering
       """
       try:
           # 데이터 가져오기
           data = self.get_data_from_influxdb(start_time)
           self.logger.info("Starting batch filtering process")
           
           # 결측치 처리 (dropna 옵션 포함)
           if self.config.get('missing_values', {}).get('use_dropna', False):
               data = data.dropna()
               self.logger.info(f"Dropped rows with missing values. Remaining rows: {len(data)}")
           else:
               data = self._handle_missing_values(data)
           
           # 급격한 변화 및 이상치 감지
           data = self._detect_anomalies(data)
           
           # 이상치 제거
           data = self._remove_outliers(data)
           
           # 데이터 정규화
           normalized_data = self._normalize_data(data)
           
           # 시뮬레이션 ID 무작위화 및 데이터 셔플링
           normalized_data = self._randomize_and_shuffle(normalized_data)
           
           # 정규화 파라미터 저장
           self._save_normalization_params()
           
           self.logger.info("Batch filtering completed successfully")
           return normalized_data, self.scaler_params
           
       except Exception as e:
           self.logger.error(f"Error in batch filtering: {e}")
           raise

    def filter_inf(self, data: Dict) -> Dict:
       """
       Process real-time inference data with loaded normalization parameters
       """
       try:
           # JSON 파라미터 로드 (없을 경우)
           if not self.scaler_params:
               self._load_normalization_params()
           
           # 데이터 검증
           if not isinstance(data, dict):
               raise ValueError("Input data must be dictionary")
           
           # 결측치 처리
           processed_data = self._handle_missing_values_realtime(data)
           
           # 급격한 변화 감지
           processed_data = self._check_rapid_changes(processed_data)
           
           # 정규화 적용
           normalized_data = self._normalize_realtime(processed_data)
           
           return normalized_data
           
       except Exception as e:
           self.logger.error(f"Error in inference filtering: {e}")
           raise

    def _detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
       """
       Detect anomalies in batch data including rapid changes and path deviations
       """
       data = data.copy()
       data['is_anomaly'] = False
       
       # IMU 급격한 변화 감지
       for field in ['angular_velocity', 'linear_acceleration']:
           for axis in ['x', 'y', 'z']:
               col = f"{field}_{axis}"
               if col in data.columns:
                   # 변화율 계산
                   change_rate = data[col].diff().abs()
                   threshold = self.anomaly_thresholds[field]['max_change']
                   data.loc[change_rate > threshold, 'is_anomaly'] = True
       
       # 경로 이탈 감지
       if all(f'position_{axis}' in data.columns for axis in ['x', 'y', 'z']):
           # 이동 평균을 이용한 예상 경로 계산
           window_size = 50
           expected_position = data[['position_x', 'position_y', 'position_z']].rolling(window=window_size).mean()
           
           # 실제 위치와 예상 경로의 차이 계산
           deviation = np.sqrt(
               (data['position_x'] - expected_position['position_x'])**2 +
               (data['position_y'] - expected_position['position_y'])**2 +
               (data['position_z'] - expected_position['position_z'])**2
           )
           
           # 임계값을 초과하는 편차를 이상으로 표시
           data.loc[deviation > self.anomaly_thresholds['position']['max_deviation'], 'is_anomaly'] = True
       
       self.logger.info(f"Detected {data['is_anomaly'].sum()} anomalous points")
       return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
       """
       Handle missing values with configurable methods
       """
       for column in data.columns:
           if column in ['timestamp', 'simulation_id', 'is_anomaly']:
               continue
               
           # 연속된 결측치 개수 확인
           null_groups = data[column].isnull().astype(int).groupby(
               data[column].notnull().astype(int).cumsum()
           ).sum()
           
           if null_groups.max() < self.missing_value_config['max_consecutive']:
               # 선형 보간
               data[column] = data[column].interpolate(
                   method=self.missing_value_config['interpolation_method']
               )
           else:
               # 통계적 대체
               if self.missing_value_config['statistical_method'] == 'mean':
                   fill_value = data[column].mean()
               elif self.missing_value_config['statistical_method'] == 'median':
                   fill_value = data[column].median()
               else:
                   fill_value = data[column].mode()[0]
               data[column] = data[column].fillna(fill_value)
       
       return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
       """
       Remove outliers using IQR method with adjusted thresholds
       """
       for column in data.columns:
           if column in ['timestamp', 'simulation_id', 'is_anomaly']:
               continue
               
           Q1 = data[column].quantile(0.25)
           Q3 = data[column].quantile(0.75)
           IQR = Q3 - Q1
           lower_bound = Q1 - 1.5 * IQR
           upper_bound = Q3 + 1.5 * IQR
           
           data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
       
       return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data and store parameters for inference
        """
        norm_data = data.copy()
        
        # 데이터 타입 변환
        numeric_columns = data.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in ['timestamp', 'simulation_id']:
                norm_data[col] = pd.to_numeric(norm_data[col], errors='coerce')

        # IMU 데이터 정규화
        imu_columns = [col for col in norm_data.columns if any(field in col for field in self.imu_fields)]
        if imu_columns:
            # 숫자형 데이터만 선택
            imu_data = norm_data[imu_columns].select_dtypes(include=[np.number])
            norm_data[imu_data.columns] = self.scalers['imu'].fit_transform(imu_data)
            self.scaler_params['imu'] = {
                'min': self.scalers['imu'].data_min_.tolist(),
                'max': self.scalers['imu'].data_max_.tolist(),
                'columns': imu_data.columns.tolist()
            }
        
        # 경로 데이터 정규화
        path_columns = [col for col in norm_data.columns if any(field in col for field in self.path_fields)]
        if path_columns:
            # 숫자형 데이터만 선택
            path_data = norm_data[path_columns].select_dtypes(include=[np.number])
            norm_data[path_data.columns] = self.scalers['path'].fit_transform(path_data)
            self.scaler_params['path'] = {
                'min': self.scalers['path'].data_min_.tolist(),
                'max': self.scalers['path'].data_max_.tolist(),
                'columns': path_data.columns.tolist()
            }
        
        return norm_data

    def _randomize_and_shuffle(self, data: pd.DataFrame) -> pd.DataFrame:
       """
       Randomize simulation IDs and shuffle data while maintaining sequence integrity
       """
       # 시뮬레이션 ID 무작위화
       if 'simulation_id' in data.columns:
           unique_ids = data['simulation_id'].unique()
           id_mapping = {old_id: f"sim_{i:03d}" for i, old_id in enumerate(unique_ids)}
           data['simulation_id'] = data['simulation_id'].map(id_mapping)
       
       # 시뮬레이션별로 데이터 셔플링
       shuffled_data = pd.DataFrame()
       simulation_groups = data.groupby('simulation_id')
       simulation_ids = list(simulation_groups.groups.keys())
       np.random.shuffle(simulation_ids)
       
       for sim_id in simulation_ids:
           sim_data = simulation_groups.get_group(sim_id)
           shuffled_data = pd.concat([shuffled_data, sim_data])
       
       return shuffled_data.reset_index(drop=True)

    def _save_normalization_params(self) -> None:
       """
       Save normalization parameters to JSON file
       """
       save_dir = os.path.join(self.config.get('save_dir', 'saved_models'))
       os.makedirs(save_dir, exist_ok=True)
       
       save_path = os.path.join(save_dir, 'normalization_params.json')
       with open(save_path, 'w') as f:
           json.dump(self.scaler_params, f, indent=4)
       
       self.logger.info(f"Saved normalization parameters to {save_path}")

    def _load_normalization_params(self) -> None:
       """
       Load normalization parameters from JSON file
       """
       load_path = os.path.join(
           self.config.get('save_dir', 'saved_models'),
           'normalization_params.json'
       )
       
       try:
           with open(load_path, 'r') as f:
               self.scaler_params = json.load(f)
           self.logger.info(f"Loaded normalization parameters from {load_path}")
       except FileNotFoundError:
           self.logger.error(f"Normalization parameters file not found at {load_path}")
           raise

    def _handle_missing_values_realtime(self, data: Dict) -> Dict:
       """
       Handle missing values in real-time data
       """
       for key in data:
           if data[key] is None:
               if key.startswith('angular_velocity'):
                   data[key] = 0.0
               elif key.startswith('linear_acceleration'):
                   data[key] = 9.81 if key.endswith('z') else 0.0
               elif key.startswith('orientation'):
                   data[key] = 0.0
               elif key.startswith('position'):
                   data[key] = 0.0
       return data

    def _check_rapid_changes(self, data: Dict) -> Dict:
       """
       Check for rapid changes in real-time data
       """
       if hasattr(self, 'previous_data'):
           for key, value in data.items():
               if key in self.previous_data:
                   change = abs(value - self.previous_data[key])
                   
                   if key.startswith('angular_velocity'):
                       threshold = self.anomaly_thresholds['angular_velocity']['max_change']
                       if change > threshold:
                           self.logger.warning(f"Rapid change detected in {key}: {change}")
                           
                   elif key.startswith('linear_acceleration'):
                       threshold = self.anomaly_thresholds['linear_acceleration']['max_change']
                       if change > threshold:
                           self.logger.warning(f"Rapid change detected in {key}: {change}")
       
       self.previous_data = data.copy()
       return data

    def _normalize_realtime(self, data: Dict) -> Dict:
        """
        Apply normalization to real-time data using stored parameters
        """
        try:
            normalized_data = {}
            for key, value in data.items():
                if key in self.scaler_params.get('imu', {}).get('columns', []):
                    idx = self.scaler_params['imu']['columns'].index(key)
                    min_val = self.scaler_params['imu']['min'][idx]
                    max_val = self.scaler_params['imu']['max'][idx]
                    
                    # IMU 데이터 정규화
                    normalized_data[key] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
                    
                elif key in self.scaler_params.get('path', {}).get('columns', []):
                    idx = self.scaler_params['path']['columns'].index(key)
                    min_val = self.scaler_params['path']['min'][idx]
                    max_val = self.scaler_params['path']['max'][idx]
                    
                    # 경로 데이터 정규화
                    normalized_data[key] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
                    
                else:
                    self.logger.warning(f"Key {key} not found in scaler parameters")
                    continue
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error in realtime normalization: {e}")
            self.logger.error(f"Error details - Key: {key}, Value: {value}")
            raise

    def cleanup(self):
       """
       Cleanup resources and connections
       """
       try:
           if hasattr(self, 'influx_client'):
               self.influx_client.close()
           self.logger.info("Resources cleaned up successfully")
       except Exception as e:
           self.logger.error(f"Error during cleanup: {e}")

    def verify_data_quality(self, data: pd.DataFrame) -> bool:
       """
       Verify the quality of processed data
       """
       try:
           # 데이터 크기 검증
           if len(data) < 1000:  # 최소 데이터 수 확인
               self.logger.warning(f"Insufficient data points: {len(data)}")
               return False
               
           # 결측치 비율 검증
           missing_ratio = data.isnull().sum() / len(data)
           if any(missing_ratio > 0.1):  # 10% 이상 결측치가 있는 컬럼 확인
               self.logger.warning(f"High missing value ratio: {missing_ratio[missing_ratio > 0.1]}")
               return False
               
           # 이상치 비율 검증
           if 'is_anomaly' in data.columns:
               anomaly_ratio = data['is_anomaly'].mean()
               if anomaly_ratio > 0.2:  # 20% 이상 이상치 확인
                   self.logger.warning(f"High anomaly ratio: {anomaly_ratio:.2%}")
                   return False
           
           return True
           
       except Exception as e:
           self.logger.error(f"Error in data quality verification: {e}")
           return False

if __name__ == "__main__":
   # 설정 파일 로드
   try:
       with open('config/filter_config.yaml', 'r') as f:
           config = yaml.safe_load(f)
   except Exception as e:
       logging.error(f"Failed to load configuration: {e}")
       raise

   # DroneDataFilter 인스턴스 생성 및 실행
   try:
       filter = DroneDataFilter(config)
       processed_data, scaler_params = filter.filter_batch()
       
       if filter.verify_data_quality(processed_data):
           logging.info("Data processing completed successfully")
       else:
           logging.warning("Data quality verification failed")
           
   except Exception as e:
       logging.error(f"Error in main execution: {e}")
       raise
   finally:
       filter.cleanup()
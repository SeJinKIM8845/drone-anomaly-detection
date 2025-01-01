import pandas as pd
import numpy as np
import json
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional

class DroneDataFilter:
   def __init__(self):
       # InfluxDB 설정
       self.INFLUXDB_URL = "http://3.37.135.231:8086"
       self.INFLUXDB_TOKEN = "your_token"
       self.INFLUXDB_ORG = "drone_org"
       self.INFLUXDB_BUCKET = "drone_data"
       self.client = InfluxDBClient(
           url=self.INFLUXDB_URL,
           token=self.INFLUXDB_TOKEN,
           org=self.INFLUXDB_ORG
       )
       
       # 스케일러와 윈도우 설정
       self.imu_scaler = MinMaxScaler()
       self.path_scaler = MinMaxScaler()
       self.window_size = 50
       self.batch_size = 1000
       
       # 로깅 설정
       self.setup_logging()
       self.load_scaler_params()

   def setup_logging(self):
       """로깅 설정"""
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(levelname)s - %(message)s',
           handlers=[
               logging.FileHandler('drone_filter.log'),
               logging.StreamHandler()
           ]
       )
       self.logger = logging.getLogger(__name__)

   def load_scaler_params(self):
       """스케일러 파라미터 로드"""
       try:
           with open("scaler_params.json", "r") as f:
               params = json.load(f)
               self.imu_scaler.min_ = np.array(params["imu"]["min"])
               self.imu_scaler.scale_ = np.array(params["imu"]["scale"])
               self.path_scaler.min_ = np.array(params["path"]["min"])
               self.path_scaler.scale_ = np.array(params["path"]["scale"])
               self.logger.info("Scaler parameters loaded successfully")
       except FileNotFoundError:
           self.logger.warning("Scaler parameters not found. Initializing new scalers.")

   def save_scaler_params(self):
       """스케일러 파라미터 저장"""
       params = {
           "imu": {
               "min": self.imu_scaler.min_.tolist(),
               "scale": self.imu_scaler.scale_.tolist()
           },
           "path": {
               "min": self.path_scaler.min_.tolist(),
               "scale": self.path_scaler.scale_.tolist()
           }
       }
       with open("scaler_params.json", "w") as f:
           json.dump(params, f)
       self.logger.info("Scaler parameters saved successfully")

   def fetch_data(self, simulation_id: Optional[str] = None, hours: int = 12) -> pd.DataFrame:
       """InfluxDB에서 데이터 조회"""
       query = f"""
       from(bucket: "{self.INFLUXDB_BUCKET}")
         |> range(start: -{hours}h)
         |> filter(fn: (r) => r["_field"] in [
             "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
             "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
             "position_x", "position_y", "position_z"
         ])
       """
       if simulation_id:
           query += f'  |> filter(fn: (r) => r["simulation_id"] == "{simulation_id}")'
       
       try:
           tables = self.client.query_api().query(query)
           df = self._tables_to_dataframe(tables)
           df['value'].fillna(0, inplace=True)
           self.logger.info(f"Successfully fetched {len(df)} records")
           return df
       except Exception as e:
           self.logger.error(f"Error fetching data: {str(e)}", exc_info=True)
           raise

   def _tables_to_dataframe(self, tables) -> pd.DataFrame:
       """InfluxDB 결과를 DataFrame으로 변환"""
       data = []
       for table in tables:
           for record in table.records:
               data.append({
                   'timestamp': record.get_time(),
                   'measurement': record.get_measurement(),
                   'field': record.get_field(),
                   'value': record.get_value(),
                   'simulation_id': record.values.get('simulation_id')
               })
       return pd.DataFrame(data)

   def validate_features(self, features: dict) -> bool:
       """특성 검증"""
       required_fields = {
           'imu': [
               'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
               'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
           ],
           'path': ['position_x', 'position_y', 'position_z']
       }
       
       for category, fields in required_fields.items():
           if not all(field in features[category]['field'].values for field in fields):
               self.logger.error(f"Missing required fields in {category} data")
               return False
       return True

   def validate_data_quality(self, df: pd.DataFrame) -> bool:
       """데이터 품질 검증"""
       validations = {
           'missing_values': df['value'].isnull().sum(),
           'unique_simulations': df['simulation_id'].nunique(),
           'timespan': df['timestamp'].max() - df['timestamp'].min(),
           'feature_completeness': self.validate_features(self._extract_features(df))
       }
       self.logger.info(f"Data validation results: {validations}")
       return validations['missing_values'] == 0 and validations['feature_completeness']

   def _extract_features(self, df: pd.DataFrame) -> dict:
       """특성 추출"""
       imu_fields = [
           'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
           'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
       ]
       path_fields = ['position_x', 'position_y', 'position_z']
       
       return {
           'imu': df[df['field'].isin(imu_fields)],
           'path': df[df['field'].isin(path_fields)]
       }

   def create_feature_windows(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       """특성 윈도우 생성"""
       self.logger.info("Creating feature windows...")
       windows = []
       labels = []
       
       # 시뮬레이션 ID 셔플링
       simulation_ids = df['simulation_id'].unique()
       if is_training:
           np.random.seed(42)
           np.random.shuffle(simulation_ids)
       
       with ThreadPoolExecutor() as executor:
           futures = []
           for sim_id in simulation_ids:
               sim_data = df[df['simulation_id'] == sim_id].sort_values('timestamp')
               futures.append(executor.submit(
                   self._process_simulation_data, sim_data, sim_id, is_training))
           
           for future in futures:
               result = future.result()
               if result:
                   sim_windows, sim_labels = result
                   windows.extend(sim_windows)
                   if is_training:
                       labels.extend(sim_labels)
       
       # 윈도우 레벨 셔플링
       if is_training and windows:
           indices = np.arange(len(windows))
           np.random.shuffle(indices)
           windows = np.array(windows)[indices]
           labels = np.array(labels)[indices]
           
           # 학습 데이터 저장
           np.save('X_train.npy', windows)
           np.save('y_train.npy', labels)
       
       return np.array(windows), np.array(labels) if is_training else None

   def _process_simulation_data(self, sim_data: pd.DataFrame, sim_id: str, is_training: bool):
       """시뮬레이션 데이터 처리"""
       try:
           features = self._extract_features(sim_data)
           if not self.validate_features(features):
               return None
           
           imu_windows = self._create_windows(features['imu'], 'imu')
           path_windows = self._create_windows(features['path'], 'path')
           
           if len(imu_windows) == 0 or len(path_windows) == 0:
               return None
           
           combined_windows = np.concatenate([imu_windows, path_windows], axis=2)
           
           if is_training:
               is_abnormal = self._is_abnormal_simulation(sim_id)
               labels = [is_abnormal] * len(combined_windows)
               return combined_windows, labels
           
           return combined_windows, None
           
       except Exception as e:
           self.logger.error(f"Error processing simulation {sim_id}: {str(e)}", exc_info=True)
           return None

   def _create_windows(self, data: pd.DataFrame, scaler_type: str, overlap: float = 0.5) -> np.ndarray:
       """데이터 윈도우 생성"""
       if len(data) < self.window_size:
           self.logger.warning("Data length less than window size")
           return np.array([])
       
       values = data['value'].values.reshape(-1, 1)
       scaled_values = self.apply_scaling(values, scaler_type)
       
       stride = int(self.window_size * (1 - overlap))
       windows = []
       
       for i in range(0, len(scaled_values) - self.window_size + 1, stride):
           window = scaled_values[i:i + self.window_size]
           if len(window) == self.window_size:
               windows.append(window)
       
       return np.array(windows)

   def apply_scaling(self, data: np.ndarray, scaler_type: str) -> np.ndarray:
       """스케일링 적용"""
       scaler = self.imu_scaler if scaler_type == 'imu' else self.path_scaler
       return scaler.fit_transform(data)

   def _is_abnormal_simulation(self, simulation_id: str) -> bool:
       """이상치 시뮬레이션 판별"""
       sim_number = int(simulation_id.split('_')[-1])
       return sim_number % 10 == 0

   def prepare_for_prediction(self, data: pd.DataFrame) -> np.ndarray:
       """LSTM 모델 입력용 데이터 준비"""
       if not self.validate_data_quality(data):
           raise ValueError("Invalid data quality")
       
       features = self._extract_features(data)
       imu_data = self._create_windows(features['imu'], 'imu')
       path_data = self._create_windows(features['path'], 'path')
       
       if len(imu_data) == 0 or len(path_data) == 0:
           raise ValueError("Insufficient data for prediction")
       
       combined_data = np.concatenate([imu_data, path_data], axis=2)
       return combined_data.reshape(-1, self.window_size, combined_data.shape[-1])

if __name__ == "__main__":
   filter = DroneDataFilter()
   try:
       raw_data = filter.fetch_data(hours=24)
       
       if filter.validate_data_quality(raw_data):
           X_train, y_train = filter.create_feature_windows(raw_data, is_training=True)
           filter.save_scaler_params()
           
           filter.logger.info(f"Processed {len(X_train)} windows")
           filter.logger.info(f"Abnormal ratio: {np.mean(y_train):.2%}")
       else:
           filter.logger.error("Data validation failed")
           
   except Exception as e:
       filter.logger.error(f"Error in main execution: {str(e)}", exc_info=True)
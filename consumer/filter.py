import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import MinMaxScaler
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import sys

class DroneDataFilter:
    def __init__(self, config_path: str = "filter_config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_paths()
        self.setup_logging()
        self.client = self._setup_influxdb()
        self.validate_config()

    def validate_config(self) -> None:
        """설정 파일 유효성 검사"""
        required_sections = ['filter', 'fields', 'influxdb', 'save', 'logging', 'performance']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")
        if len(self.config['fields']) == 0:
            raise ValueError("No sensor fields defined in config")
        self.logger.info("Config validation successful")

    def setup_paths(self) -> None:
        """저장 경로 설정"""
        for path_name, path in self.config['save'].items():
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            setattr(self, f"{path_name}_path", path_obj)

    def setup_logging(self) -> None:
        """로깅 설정"""
        log_file = Path(self.config['save']['log_dir']) / self.config['logging']['file_name']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")

    def _setup_influxdb(self) -> InfluxDBClient:
        """InfluxDB 클라이언트 설정"""
        return InfluxDBClient(
            url=self.config['influxdb']['url'],
            token=self.config['influxdb']['token'],
            org=self.config['influxdb']['org']
        )

    def validate_data(self, df: pd.DataFrame, sensor_type: str) -> None:
        if df.empty:
            raise ValueError(f"Empty dataframe for sensor type: {sensor_type}")
        required_features = self.config['fields'][sensor_type]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features for {sensor_type}: {missing_features}")
        self.logger.info(f"Data validation successful for {sensor_type}")
        self.logger.info(f"Shape: {df.shape}, Features: {df.columns.tolist()}")

    def validate_shapes(self, data: Dict[str, pd.DataFrame], sequences: np.ndarray, sim_ids: np.ndarray) -> None:
        first_sensor = next(iter(data.keys()))
        df_first = data[first_sensor].sort_values('_time')
        sequence_length = self.config['filter']['data_processing']['sequence_length']
        stride = self.config['filter']['data_processing']['stride']
        expected_count = 0
        for sim_id, group in df_first.groupby('simulation_id'):
            count = (len(group) - sequence_length) // stride + 1 if len(group) >= sequence_length else 0
            expected_count += count
        actual_count = len(sequences)
        self.logger.info(f"Expected sequences: {expected_count}, Generated sequences: {actual_count}")
        expected_features = sum(len(features) for features in self.config['fields'].values())
        if sequences.shape[2] != expected_features:
            raise ValueError(f"Feature count mismatch. Expected: {expected_features}, Got: {sequences.shape[2]}")

    def get_sensor_data(self, sensor_type: str) -> pd.DataFrame:
        query = f'''
        from(bucket: "{self.config['influxdb']['bucket']}")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "drone_{sensor_type}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = self.client.query_api().query_data_frame(query)
        if isinstance(result, list):
            result = pd.concat(result)
        self.logger.info(f"Retrieved columns for {sensor_type}: {result.columns.tolist()}")
        if 'simulation_id' not in result.columns:
            self.logger.warning(f"simulation_id column missing for {sensor_type}, defaulting to 0")
            result['simulation_id'] = 0
        if result.empty:
            self.logger.warning(f"No data found for sensor type: {sensor_type}")
            return result
        features = self.config['fields'][sensor_type]
        required_columns = ['_time', 'simulation_id'] + features
        missing = [col for col in required_columns if col not in result.columns]
        if missing:
            raise KeyError(f"Missing columns for {sensor_type}: {missing}")
        result = result[required_columns].sort_values('_time')
        self.validate_data(result, sensor_type)
        return result

    def get_all_sensor_data(self) -> Dict[str, pd.DataFrame]:
        """모든 센서 데이터 조회"""
        return {sensor_type: self.get_sensor_data(sensor_type)
                for sensor_type in self.config['fields']}

    def filter_batch(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        self.logger.info("Starting batch filtering process")
        clean_data = self._handle_missing_values(data)
        sync_data = self._synchronize_data(clean_data)
        normalized_data, scaler_params = self._normalize_data(sync_data)
        sequences, sim_ids = self._create_sequences(normalized_data)
        sequences, sim_ids = self._shuffle_data(sequences, sim_ids)
        self.validate_shapes(sync_data, sequences, sim_ids)
        self._save_processed_data(sequences, 'batch')
        self._save_scaler_params(scaler_params)
        self.logger.info("Batch filtering completed")
        return sequences, sim_ids, scaler_params

    def filter_inf(self, data: Dict[str, pd.DataFrame], scaler_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        clean_data = self._handle_missing_values(data)
        sync_data = self._synchronize_data(clean_data)
        normalized_data = self._apply_normalization(sync_data, scaler_params)
        sequences, sim_ids = self._create_sequences(normalized_data)
        self.validate_shapes(sync_data, sequences, sim_ids)
        return sequences, sim_ids

    def _handle_missing_values(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        clean_data = {}
        max_gap = self.config['filter']['missing_values']['max_consecutive_missing']
        for sensor_type, df in data.items():
            self.logger.info(f"Handling missing values for {sensor_type}")
            initial_missing = df.isnull().sum().sum()
            df_clean = df.copy().interpolate(
                method=self.config['filter']['missing_values']['interpolation_method'],
                limit=max_gap
            )
            remaining_missing = df_clean.isnull().sum().sum()
            if remaining_missing > 0:
                self.logger.warning(f"Filling remaining {remaining_missing} missing values with mean for {sensor_type}")
                df_clean = df_clean.fillna(df_clean.mean())
            clean_data[sensor_type] = df_clean
            self.logger.info(f"Handled {initial_missing - remaining_missing} missing values for {sensor_type}")
        return clean_data

    def _synchronize_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.logger.info("Starting sensor data synchronization using imu timestamps as base")
        # 모든 센서 데이터의 _time을 datetime으로 변환 및 정렬
        for sensor_type, df in data.items():
            df['_time'] = pd.to_datetime(df['_time'])
            df.sort_values('_time', inplace=True)
        
        # imu 데이터가 반드시 존재해야 함 (기준 센서)
        if 'imu' not in data or data['imu'].empty:
            raise ValueError("IMU 데이터가 존재하지 않아 기준 인덱스를 생성할 수 없습니다.")
        base_df = data['imu'].copy()  # imu 데이터가 기준
        base_df.reset_index(drop=True, inplace=True)
        base_times = base_df['_time']
        self.logger.info(f"Base sensor (imu) data shape: {base_df.shape}")
        
        # tolerance: merge_asof 적용 시 허용 시간차 (설정 파일에서 지정, 예: '25ms')
        tolerance = pd.Timedelta(self.config['filter']['data_processing'].get('tolerance', '25ms'))
        
        sync_data = {}
        # imu는 기준 데이터 그대로 사용
        sync_data['imu'] = base_df
        
        # 나머지 센서들에 대해 imu 타임스탬프를 기준으로 merge_asof 적용
        for sensor_type, df in data.items():
            if sensor_type == 'imu':
                continue
            self.logger.info(f"Synchronizing sensor: {sensor_type} using imu timestamps as base")
            base_times_df = pd.DataFrame({'_time': base_times})
            # merge_asof: imu의 각 타임스탬프에 대해, 해당 센서 데이터에서 가장 가까운 값을 선택
            merged = pd.merge_asof(base_times_df, df, on='_time', tolerance=tolerance, direction='nearest')
            # 결측값 보간: 앞/뒤 값 채우기
            merged = merged.fillna(method='ffill').fillna(method='bfill')
            if 'simulation_id' not in merged.columns:
                merged['simulation_id'] = 0
            sync_data[sensor_type] = merged
            self.logger.info(f"Synchronized {sensor_type} data: shape {merged.shape}")
        return sync_data

    def _normalize_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        normalized_data = {}
        scaler_params = {}
        feature_range = tuple(self.config['filter']['normalization']['feature_range'])
        for sensor_type, df in data.items():
            self.logger.info(f"Normalizing {sensor_type} data")
            df_grouped = df.groupby('simulation_id')
            normalized_dfs = []
            for sim_id, group in df_grouped:
                scaler = MinMaxScaler(feature_range=feature_range)
                features = self.config['fields'][sensor_type]
                scaler.fit(group[features])
                normalized_values = scaler.transform(group[features])
                # 만약 feature의 범위가 0이면, 해당 column은 0으로 대체
                diff = scaler.data_max_ - scaler.data_min_
                for col_idx, d in enumerate(diff):
                    if d == 0:
                        normalized_values[:, col_idx] = 0
                normalized_df = pd.DataFrame(normalized_values, columns=features, index=group.index)
                normalized_df['simulation_id'] = sim_id
                normalized_dfs.append(normalized_df)
                if sensor_type not in scaler_params:
                    scaler_params[sensor_type] = []
                scaler_params[sensor_type].append({
                    'simulation_id': sim_id,
                    'min': scaler.data_min_.tolist(),
                    'max': scaler.data_max_.tolist(),
                    'feature_names': features
                })
            if normalized_dfs:
                normalized_data[sensor_type] = pd.concat(normalized_dfs)
            else:
                raise ValueError(f"No normalized groups for sensor {sensor_type}; cannot concatenate.")
            self.logger.info(f"Normalized {len(normalized_dfs)} simulation groups for {sensor_type}")
        return normalized_data, scaler_params

    def _create_sequences(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        sequence_length = self.config['filter']['data_processing']['sequence_length']
        stride = self.config['filter']['data_processing']['stride']
        valid_features = []
        sim_id_df = None
        for sensor_type, df in data.items():
            if not df.empty:
                feature_df = df.drop('simulation_id', axis=1)
                feature_df.columns = [f"{sensor_type}_{col}" for col in feature_df.columns]
                valid_features.append(feature_df)
                if sim_id_df is None:
                    sim_id_df = df[['simulation_id']]
        if not valid_features or sim_id_df is None:
            raise ValueError("No valid data to create sequences")
        combined_features = pd.concat(valid_features, axis=1)
        combined_data = pd.concat([sim_id_df, combined_features], axis=1)
        self.logger.info(f"Combined data shape: {combined_data.shape}")
        sequences = []
        sequence_sim_ids = []
        for sim_id, group in combined_data.groupby('simulation_id'):
            self.logger.info(f"Creating sequences for simulation {sim_id}")
            group = group.sort_index()
            for i in range(0, len(group) - sequence_length + 1, stride):
                sequence = group.iloc[i:i + sequence_length].drop('simulation_id', axis=1).values
                sequences.append(sequence)
                sequence_sim_ids.append(sim_id)
            self.logger.info(f"Created {len(range(0, len(group) - sequence_length + 1, stride))} sequences for simulation {sim_id}")
        return np.array(sequences), np.array(sequence_sim_ids)

    def _shuffle_data(self, sequences: np.ndarray, sim_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique_ids = np.unique(sim_ids)
        np.random.shuffle(unique_ids)
        new_sequences = []
        new_sim_ids = []
        for uid in unique_ids:
            indices = np.where(sim_ids == uid)[0]
            new_sequences.extend(sequences[indices])
            new_sim_ids.extend(sim_ids[indices])
            self.logger.info(f"Simulation {uid}: {len(indices)} sequences added")
        return np.array(new_sequences), np.array(new_sim_ids)

    def _save_processed_data(self, sequences: np.ndarray, mode: str) -> None:
        save_path = Path(self.config['save']['base_dir'])
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f'{mode}_sequences.npy', sequences)
        reshaped_data = sequences.reshape(-1, sequences.shape[-1])
        feature_names = self._get_feature_names()
        pd.DataFrame(reshaped_data, columns=feature_names).to_csv(save_path / f'{mode}_data.csv', index=False)
        self.logger.info(f"Processed data saved at {save_path}")

    def _save_scaler_params(self, scaler_params: Dict) -> None:
        save_path = Path(self.config['save']['base_dir']) / self.config['filter']['normalization']['params_path']
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        self.logger.info(f"Scaler parameters saved at {save_path}")

    def _get_feature_names(self) -> List[str]:
        feature_names = []
        for sensor_type in self.config['fields']:
            for feature in self.config['fields'][sensor_type]:
                feature_names.append(f"{sensor_type}_{feature}")
        return feature_names

    def _apply_normalization(self, data: Dict[str, pd.DataFrame], scaler_params: Dict) -> Dict[str, pd.DataFrame]:
        normalized_data = {}
        feature_range = tuple(self.config['filter']['normalization']['feature_range'])
        for sensor_type, df in data.items():
            normalized_dfs = []
            for sim_params in scaler_params[sensor_type]:
                sim_id = sim_params['simulation_id']
                sim_data = df[df['simulation_id'] == sim_id]
                if sim_data.empty:
                    continue
                data_min = np.array(sim_params['min'])
                data_max = np.array(sim_params['max'])
                scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
                min_val = feature_range[0] - data_min * scale
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.scale_ = scale
                scaler.min_ = min_val
                features = sim_params['feature_names']
                normalized_values = scaler.transform(sim_data[features])
                diff = data_max - data_min
                for col_idx, d in enumerate(diff):
                    if d == 0:
                        normalized_values[:, col_idx] = 0
                normalized_df = pd.DataFrame(normalized_values, columns=features, index=sim_data.index)
                normalized_df['simulation_id'] = sim_id
                normalized_dfs.append(normalized_df)
            if normalized_dfs:
                normalized_data[sensor_type] = pd.concat(normalized_dfs)
                self.logger.info(f"Applied normalization for {sensor_type}")
            else:
                self.logger.warning(f"No data found for sensor {sensor_type} during inference normalization")
        return normalized_data

    def cleanup(self) -> None:
        """리소스 정리 (InfluxDB 클라이언트 종료)"""
        if hasattr(self, 'client'):
            self.client.close()
        self.logger.info("Resources cleaned up")

if __name__ == "__main__":
    try:
        drone_filter = DroneDataFilter()
        data = drone_filter.get_all_sensor_data()
        sequences, sim_ids, scaler_params = drone_filter.filter_batch(data)
        drone_filter.cleanup()
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)

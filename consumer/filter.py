#!/usr/bin/env python3
import os
import json
import yaml
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

class FilterPreprocessor:
    def __init__(self, config_path="filter_config.yaml"):
        self.load_config(config_path)
        self.setup_logging()
        self.setup_influxdb_client()
        # Load filter-specific parameters from config
        self.resample_freq = self.config["filter"].get("resample_freq", "33ms")
        self.input_window_size = self.config["filter"].get("input_window_size", 90)
        self.output_window_size = self.config["filter"].get("output_window_size", 150)
        self.target_columns = self.config["filter"].get("target_columns", [
            "drone_local_pose_position_x",
            "drone_local_pose_position_y",
            "drone_local_pose_position_z"
        ])
        self.scaler_file = self.config["filter"].get("scaler_file", "scaler_params.json")
        # 사용할 센서 측정치 목록 (LSTM 모델에 필요한 데이터)
        self.measurements = ["drone_imu", "drone_local_pose", "drone_velocity"]

    def load_config(self, config_path):
        config_full_path = os.path.join(os.path.dirname(__file__), config_path)
        with open(config_full_path, "r") as f:
            self.config = yaml.safe_load(f)

    def setup_logging(self):
        log_dir = self.config["logging"]["dir"]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'filter_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)

    def setup_influxdb_client(self):
        self.influx_client = InfluxDBClient(
            url=self.config["influxdb"]["url"],
            token=self.config["influxdb"]["token"],
            org=self.config["influxdb"]["org"]
        )
        self.query_api = self.influx_client.query_api()
        self.bucket = self.config["influxdb"]["bucket"]
        self.logger.info("InfluxDB client initialized")

    def get_simulation_ids(self, days=-30):
        """InfluxDB에서 최근 30일간의 distinct simulation_id 목록을 조회합니다."""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {days}d)
          |> distinct(column: "simulation_id")
        '''
        sim_ids = set()
        try:
            result = self.query_api.query(query)
            for table in result:
                for record in table.records:
                    sim_ids.add(record.get_value())
            sim_ids = sorted(list(sim_ids))
            self.logger.info(f"Found {len(sim_ids)} simulation IDs.")
            return sim_ids
        except Exception as e:
            self.logger.error(f"Error retrieving simulation IDs: {e}")
            return []

    def query_measurement(self, measurement, simulation_id, start, stop):
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["simulation_id"] == "{simulation_id}")
        '''
        try:
            df = self.query_api.query_data_frame(query)
            if df.empty:
                self.logger.warning(f"No data returned for measurement {measurement} simulation {simulation_id}")
                return pd.DataFrame()
            # 피벗하여 _field가 컬럼이 되도록 변환
            if "_field" in df.columns:
                df = df.pivot(index="_time", columns="_field", values="_value").reset_index()
            # 각 컬럼에 measurement 접두사 추가
            new_columns = {}
            for col in df.columns:
                if col != "_time":
                    new_columns[col] = f"{measurement}_{col}"
            df = df.rename(columns=new_columns)
            df["simulation_id"] = simulation_id
            df["_time"] = pd.to_datetime(df["_time"])
            return df
        except Exception as e:
            self.logger.error(f"Query error for measurement {measurement}: {e}")
            return pd.DataFrame()

    def fetch_and_merge_data(self, simulation_id, start, stop):
        dfs = []
        for m in self.measurements:
            df = self.query_measurement(m, simulation_id, start, stop)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            self.logger.error(f"No data found for simulation {simulation_id}")
            return pd.DataFrame()
        # _time 기준으로 각 데이터프레임 병합 (merge_asof 사용)
        base_df = dfs[0].sort_values("_time")
        for df in dfs[1:]:
            df = df.sort_values("_time")
            base_df = pd.merge_asof(base_df, df, on="_time", by="simulation_id", tolerance=pd.Timedelta("50ms"))
        base_df = base_df.sort_values("_time").reset_index(drop=True)
        return base_df

    def resample_and_interpolate(self, df):
        # _time을 인덱스로 설정하고 지정 주기(예: 33ms)로 리샘플링
        df = df.set_index("_time")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_resampled = df[numeric_cols].resample(self.resample_freq).mean()
        # 선형 보간으로 결측치 채움
        df_resampled = df_resampled.interpolate(method="linear")
        # simulation_id는 forward-fill 처리
        df_resampled["simulation_id"] = df["simulation_id"].resample(self.resample_freq).ffill()
        df_resampled = df_resampled.reset_index()
        return df_resampled

    def normalize_data(self, df, scaler_params=None):
        norm_df = df.copy()
        feature_cols = [col for col in df.columns if col not in ["_time", "simulation_id"]]
        params = {}
        for col in feature_cols:
            if scaler_params is None:
                col_min = norm_df[col].min()
                col_max = norm_df[col].max()
                params[col] = {"min": float(col_min), "max": float(col_max)}
            else:
                col_min = scaler_params[col]["min"]
                col_max = scaler_params[col]["max"]
            # 0으로 나누는 경우 처리
            if col_max - col_min != 0:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.0
        return norm_df, params

    def create_sliding_windows(self, df, input_size, output_size):
        # feature_cols: _time, simulation_id 제외한 모든 피처
        feature_cols = [col for col in df.columns if col not in ["_time", "simulation_id"]]
        X_windows = []
        Y_windows = []
        total_length = len(df)
        # 슬라이딩 윈도우: 입력은 input_size, 타깃은 target_columns (예, 평탄화된 drone_local_pose의 position)
        for i in range(total_length - input_size - output_size + 1):
            X_window = df.iloc[i:i+input_size][feature_cols].values
            Y_window = df.iloc[i+input_size:i+input_size+output_size][self.target_columns].values
            X_windows.append(X_window)
            Y_windows.append(Y_window)
        return np.array(X_windows), np.array(Y_windows)

    def filter_batch(self, simulation_ids, start, stop, input_size=None, output_size=None):
        if input_size is None:
            input_size = self.input_window_size
        if output_size is None:
            output_size = self.output_window_size
        all_X = []
        all_Y = []
        scaler_all = {}
        for sim_id in simulation_ids:
            self.logger.info(f"Processing simulation: {sim_id}")
            df_raw = self.fetch_and_merge_data(sim_id, start, stop)
            if df_raw.empty:
                continue
            df_sync = self.resample_and_interpolate(df_raw)
            df_norm, scaler_params = self.normalize_data(df_sync)
            scaler_all[sim_id] = scaler_params
            X, Y = self.create_sliding_windows(df_norm, input_size, output_size)
            all_X.append(X)
            all_Y.append(Y)
        if not all_X:
            self.logger.error("No simulation data processed.")
            return None, None
        # simulation 단위 블록을 무작위 셔플 (내부 순서는 그대로 유지)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        X_shuffled = np.concatenate([all_X[i] for i in indices], axis=0)
        Y_shuffled = np.concatenate([all_Y[i] for i in indices], axis=0)
        with open(self.scaler_file, "w") as f:
            json.dump(scaler_all, f, indent=4)
        self.logger.info(f"Scaler parameters saved to {self.scaler_file}")
        return X_shuffled, Y_shuffled

    def filter_inf(self, simulation_id, start, stop, input_size=None):
        if input_size is None:
            input_size = self.input_window_size
        try:
            with open(self.scaler_file, "r") as f:
                scaler_all = json.load(f)
            scaler_params = scaler_all.get(simulation_id, None)
            if scaler_params is None:
                self.logger.error(f"No scaler parameters found for simulation {simulation_id}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading scaler parameters: {e}")
            return None
        df_raw = self.fetch_and_merge_data(simulation_id, start, stop)
        if df_raw.empty:
            return None
        df_sync = self.resample_and_interpolate(df_raw)
        df_norm, _ = self.normalize_data(df_sync, scaler_params=scaler_params)
        if len(df_norm) < input_size:
            self.logger.error("Not enough data for inference window.")
            return None
        X_inf = df_norm.iloc[-input_size:]
        return X_inf

def main():
    preprocessor = FilterPreprocessor()
    simulation_ids = preprocessor.get_simulation_ids(days=-30)
    if not simulation_ids:
        preprocessor.logger.error("No simulation IDs found.")
        return

    start_time = "2025-01-01T02:00:00Z"
    stop_time = "2025-12-31T02:30:00Z"
    X, Y = preprocessor.filter_batch(simulation_ids, start_time, stop_time)
    if X is not None and Y is not None:
        preprocessor.logger.info(f"Batch filtering complete. X shape: {X.shape}, Y shape: {Y.shape}")
    else:
        preprocessor.logger.error("Batch filtering failed.")

    simulation_id = simulation_ids[0]
    X_inf = preprocessor.filter_inf(simulation_id, start_time, stop_time)
    if X_inf is not None:
        preprocessor.logger.info(f"Inference window shape: {X_inf.shape}")
    else:
        preprocessor.logger.error("Inference filtering failed.")

if __name__ == "__main__":
    main()

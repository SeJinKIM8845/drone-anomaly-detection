#!/usr/bin/env python3

import os
import json
import yaml
import logging
import socket
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class DroneDataConsumer:
    def __init__(self):
        self.load_config()
        self.setup_logging()
        self.setup_kafka_consumer()
        self.setup_influxdb_client()

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def setup_logging(self):
        log_path = os.path.join(os.path.dirname(__file__), 
                               self.config['logging']['dir'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            
        log_file = os.path.join(
            log_path, 
            f'drone_consumer_{socket.gethostname()}_{datetime.now().strftime("%Y%m%d")}.log'
        )
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_kafka_consumer(self):
        self.kafka_config = {
            'bootstrap_servers': self.config['kafka']['bootstrap_servers'],
            'group_id': self.config['kafka']['consumer']['group_id'],
            'client_id': f"drone_consumer_{socket.gethostname()}",
            'auto_offset_reset': self.config['kafka']['consumer']['auto_offset_reset'],
            'enable_auto_commit': self.config['kafka']['consumer']['enable_auto_commit'],
            'value_deserializer': lambda x: json.loads(x.decode('utf-8'))
        }
        
        try:
            self.consumer = KafkaConsumer(
                *self.config['kafka']['topics'], 
                **self.kafka_config
            )
            self.logger.info("Kafka Consumer initialized and subscribed to topics")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka Consumer: {e}")
            raise

    def setup_influxdb_client(self):
        self.influx_client = InfluxDBClient(
            url=self.config['influxdb']['url'],
            token=self.config['influxdb']['token'],
            org=self.config['influxdb']['org']
        )
        self.bucket = self.config['influxdb']['bucket']
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

    def write_to_influxdb(self, topic, data):
        try:
            # simulation_id를 태그로 사용하여 데이터 구분
            point = Point(topic)\
                .tag("simulation_id", data.get('simulation_id', 'default'))\
                .time(datetime.utcnow(), WritePrecision.NS)

            # 데이터 필드 추가
            if 'data' in data:  
                for key, value in data['data'].items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            field_name = f"{key}_{sub_key}"
                            if isinstance(sub_value, (int, float, bool, str)):
                                point.field(field_name, sub_value)
                    else:
                        if isinstance(value, (int, float, bool, str)):
                            point.field(key, value)
            
            for key, value in data.items():
                if key not in ['data', 'simulation_id', 'timestamp', 'topic'] and isinstance(value, (int, float, bool, str)):
                    point.field(key, value)

            self.write_api.write(bucket=self.bucket, org=self.config['influxdb']['org'], record=point)
            self.logger.info(f"Data written to InfluxDB. Topic: {topic}, Simulation ID: {data.get('simulation_id', 'default')}")
        except Exception as e:
            self.logger.error(f"Failed to write data to InfluxDB. Topic: {topic}, Error: {e}")

    def process_messages(self):
        try:
            for message in self.consumer:
                if message and message.value:
                    self.logger.info(f"Received message from Kafka. Topic: {message.topic}, Simulation ID: {message.value.get('simulation_id', 'default')}")
                    self.write_to_influxdb(message.topic, message.value)
        except KafkaError as e:
            self.logger.error(f"Error processing messages from Kafka: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.consumer.close()

    def run(self):
        try:
            self.logger.info("Starting DroneDataConsumer...")
            self.process_messages()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Shutting down consumer...")
        except Exception as e:
            self.logger.error(f"Unexpected error in run: {e}")
        finally:
            if hasattr(self, 'consumer'):
                self.consumer.close()
            if hasattr(self, 'influx_client'):
                self.influx_client.close()

if __name__ == "__main__":
    consumer = DroneDataConsumer()
    consumer.run()
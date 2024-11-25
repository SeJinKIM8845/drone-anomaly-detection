#!/usr/bin/env python3

import os
import json
import logging
import socket
from datetime import datetime, timezone
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DroneDataConsumer:
    def __init__(self):
        self.setup_logging()
        self.setup_kafka_consumer()
        self.setup_influxdb_client()

    def setup_logging(self):
        log_path = os.getenv("LOG_PATH", os.path.join(os.getcwd(), 'logs'))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, f'drone_consumer_{socket.gethostname()}_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_kafka_consumer(self):
        kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS").split(',')
        self.kafka_config = {
            'bootstrap_servers': kafka_servers,
            'group_id': 'drone_data_group',
            'client_id': f"drone_consumer_{socket.gethostname()}",
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'value_deserializer': lambda x: json.loads(x.decode('utf-8'))
        }
        self.topics = os.getenv("KAFKA_TOPICS").split(',')
        try:
            self.consumer = KafkaConsumer(*self.topics, **self.kafka_config)
            self.logger.info("Kafka Consumer initialized and subscribed to topics")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka Consumer: {e}")
            raise

    def setup_influxdb_client(self):
        self.influx_client = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL"),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG")
        )
        self.bucket = os.getenv("INFLUXDB_BUCKET")
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

    def write_to_influxdb(self, topic, data):
        try:
            point = Point(topic).time(datetime.utcnow(), WritePrecision.NS)
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        point.field(f"{key}_{sub_key}", sub_value)
                else:
                    point.field(key, value)

            self.write_api.write(bucket=self.bucket, org=os.getenv("INFLUXDB_ORG"), record=point)
            self.logger.info(f"Data written to InfluxDB. Topic: {topic}, Data: {data}")
        except Exception as e:
            self.logger.error(f"Failed to write data to InfluxDB. Topic: {topic}, Error: {e}")

    def process_messages(self):
        try:
            for message in self.consumer:
                if message and message.value:
                    self.logger.info(f"Received message from Kafka. Topic: {message.topic}, Data: {message.value}")
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
            if self.consumer:
                self.consumer.close()
            if hasattr(self, 'influx_client'):
                self.influx_client.close()

if __name__ == "__main__":
    consumer = DroneDataConsumer()
    consumer.run()

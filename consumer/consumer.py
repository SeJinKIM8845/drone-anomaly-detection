#!/usr/bin/env python3

import os
import json
import yaml
import logging
from datetime import datetime
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class DroneDataConsumer:
    def __init__(self):
        self.load_config()
        self.setup_logging()
        self.setup_kafka_consumer()
        self.setup_influxdb_client()

    def load_config(self):
        """Load configuration from consumer_config.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), 'consumer_config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging configuration."""
        log_path = os.path.join(os.path.dirname(__file__), self.config['logging']['dir'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        log_file = os.path.join(log_path, f'drone_consumer_{datetime.now().strftime("%Y%m%d")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format=self.config['logging']['format'],
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)

    def setup_kafka_consumer(self):
        """Initialize Kafka Consumer."""
        self.consumer = KafkaConsumer(
            *self.config['kafka']['topics'],
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset=self.config['kafka']['consumer']['auto_offset_reset'],
            group_id=self.config['kafka']['consumer']['group_id']
        )
        self.logger.info("Kafka Consumer initialized and subscribed to topics")

    def setup_influxdb_client(self):
        """Initialize InfluxDB Client."""
        self.influx_client = InfluxDBClient(
            url=self.config['influxdb']['url'],
            token=self.config['influxdb']['token'],
            org=self.config['influxdb']['org']
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.bucket = self.config['influxdb']['bucket']
        self.logger.info("InfluxDB client initialized")

    def convert_timestamp_to_influxdb_format(self, timestamp):
        """
        Convert the given timestamp to InfluxDB-compatible ISO 8601 format.
        """
        try:
            # Parse the timestamp with timezone information
            dt = datetime.fromisoformat(timestamp.replace("Z", ""))
            # Format as ISO 8601 with microseconds and timezone
            return dt.isoformat()
        except ValueError as e:
            self.logger.error(f"Timestamp conversion error: {e}")
            raise

    def write_to_influxdb(self, topic, message):
        """
        Write a Kafka message to InfluxDB.
        :param topic: Kafka topic name.
        :param message: The message received from Kafka.
        """
        try:
            # Convert the timestamp to InfluxDB-compatible format
            influx_timestamp = self.convert_timestamp_to_influxdb_format(message["timestamp"])
            point = Point(topic)\
                .tag("simulation_id", message["simulation_id"])\
                .time(influx_timestamp, WritePrecision.NS)

            for key, value in message["data"].items():
                if isinstance(value, dict):  # Handle nested fields (e.g., IMU data)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, str, bool)):
                            point.field(f"{key}_{sub_key}", sub_value)
                elif isinstance(value, (int, float, str, bool)):
                    point.field(key, value)

            self.write_api.write(bucket=self.bucket, org=self.config['influxdb']['org'], record=point)
            self.logger.info(f"Data written to InfluxDB. Topic: {topic}, Timestamp: {message['timestamp']}")
        except Exception as e:
            self.logger.error(f"Failed to write data to InfluxDB. Error: {e}")

    def process_messages(self):
        """Consume messages from Kafka and write to InfluxDB."""
        try:
            for message in self.consumer:
                if message and message.value:
                    self.logger.info(f"Received message from Kafka. Topic: {message.topic}, Data: {message.value}")
                    self.write_to_influxdb(message.topic, message.value)
        except Exception as e:
            self.logger.error(f"Error while processing messages: {e}")
        finally:
            self.consumer.close()

    def run(self):
        """Run the Kafka Consumer."""
        try:
            self.logger.info("Starting DroneDataConsumer...")
            self.process_messages()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Shutting down...")
        finally:
            self.consumer.close()
            self.influx_client.close()

if __name__ == "__main__":
    consumer = DroneDataConsumer()
    consumer.run()
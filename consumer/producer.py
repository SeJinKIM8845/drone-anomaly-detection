#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import json
import logging
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import socket
from threading import Lock
from sensor_msgs.msg import Imu, BatteryState, NavSatFix
from mavros_msgs.msg import State, VfrHud
from geometry_msgs.msg import PoseStamped, TwistStamped


class DroneKafkaPublisher(Node):
    def __init__(self):
        super().__init__('drone_kafka_publisher')
        self.setup_logging()
        self.setup_kafka_producer()
        self.producer_lock = Lock()
        self.setup_subscribers()
        self.get_logger().info("DroneKafkaPublisher initialized and ready to publish data")

    def setup_logging(self):
        current_path = os.getcwd()
        log_path = os.path.join(current_path, 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, f'drone_producer_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_kafka_producer(self):
        self.kafka_config = {
            'bootstrap_servers': [
                'kafka01:9093',
                'kafka02:9093',
                'kafka03:9093'
            ],
            'client_id': f'drone_producer_{socket.gethostname()}',
            'acks': 'all',
            'retries': 5,
            'retry_backoff_ms': 1000,
            'compression_type': 'gzip',
            'key_serializer': lambda x: x.encode('utf-8'),
            'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8')
        }
        try:
            self.producer = KafkaProducer(**self.kafka_config)
            self.logger.info("Kafka Producer initialized successfully")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise

    def send_data(self, topic, data):
        try:
            key = str(datetime.utcnow().timestamp())
            with self.producer_lock:
                self.logger.info(f"Sending data to Kafka topic '{topic}': {data}")
                future = self.producer.send(
                    topic,
                    key=key,
                    value=data,
                    timestamp_ms=int(datetime.utcnow().timestamp() * 1000)
                )
                future.get(timeout=10)
                self.producer.flush(timeout=5)
                self.logger.info(f"Data sent to topic '{topic}' successfully")
                return True
        except KafkaError as e:
            self.logger.error(f"Error sending data to topic '{topic}': {e}")
            return False

    def setup_subscribers(self):
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,  
        durability=DurabilityPolicy.VOLATILE,       
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(BatteryState, '/mavros/battery', self.battery_callback, qos_profile)
        self.create_subscription(VfrHud, '/mavros/vfr_hud', self.vfr_callback, qos_profile)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_profile)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.global_position_callback, qos_profile)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self.velocity_callback, qos_profile)
        self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile)

    def imu_callback(self, msg):
        imu_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            },
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            },
            'linear_acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            }
        }
        self.send_data('drone_imu', imu_data)

    def battery_callback(self, msg):
        battery_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'voltage': msg.voltage,
            'current': msg.current,
            'percentage': msg.percentage
        }
        self.send_data('drone_battery', battery_data)

    def vfr_callback(self, msg):
        vfr_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'airspeed': msg.airspeed,
            'groundspeed': msg.groundspeed,
            'altitude': msg.altitude
        }
        self.send_data('drone_vfr', vfr_data)

    def local_pose_callback(self, msg):
        pose_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'position': {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            },
            'orientation': {
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z,
                'w': msg.pose.orientation.w
            }
        }
        self.send_data('drone_local_pose', pose_data)

    def global_position_callback(self, msg):
        position_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }
        self.send_data('drone_global_position', position_data)

    def velocity_callback(self, msg):
        velocity_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'linear_velocity': {
                'x': msg.twist.linear.x,
                'y': msg.twist.linear.y,
                'z': msg.twist.linear.z
            },
            'angular_velocity': {
                'x': msg.twist.angular.x,
                'y': msg.twist.angular.y,
                'z': msg.twist.angular.z
            }
        }
        self.send_data('drone_velocity', velocity_data)

    def state_callback(self, msg):
        state_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'connected': msg.connected,
            'armed': msg.armed,
            'mode': msg.mode
        }
        self.send_data('drone_state', state_data)

    def shutdown(self):
        if hasattr(self, 'producer'):
            self.logger.info("Shutting down Kafka Producer")
            self.producer.close()

def main(args=None):
    rclpy.init(args=args)
    node = DroneKafkaPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node...")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
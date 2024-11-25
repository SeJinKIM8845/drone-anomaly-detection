#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import json
import yaml
import logging
import uuid
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
        self.load_config()
        self.simulation_id = self.generate_simulation_id()
        self.sequence_number = 0
        self.setup_logging()
        self.setup_kafka_producer()
        self.producer_lock = Lock()
        self.setup_subscribers()
        self.get_logger().info(f"Started new simulation with ID: {self.simulation_id}")

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def generate_simulation_id(self):
        """시뮬레이션 고유 ID 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"sim_{timestamp}_{unique_id}"

    def setup_logging(self):
        log_path = os.path.join(os.path.dirname(__file__), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            
        log_file = os.path.join(
            log_path, 
            f'drone_producer_{self.simulation_id}.log'
        )
        
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
            'bootstrap_servers': self.config['kafka']['bootstrap_servers'],
            'client_id': f'drone_producer_{socket.gethostname()}',
            'acks': 'all',                    
            'retries': 5,                     
            'retry_backoff_ms': 1000,        
            'compression_type': 'gzip',        
            'key_serializer': lambda x: x.encode('utf-8'),
            'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8'),
            'request_timeout_ms': 30000,      
            'max_block_ms': 60000,            
            'batch_size': 16384,              
            'linger_ms': 50                  
        }

        try:
            self.producer = KafkaProducer(**self.kafka_config)
            self.logger.info(f"Kafka Producer initialized for simulation {self.simulation_id}")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise

    def create_message(self, topic, data):
        """메시지에 시뮬레이션 ID와 시퀀스 번호 추가"""
        self.sequence_number += 1
        return {
            'simulation_id': self.simulation_id,
            'sequence': self.sequence_number,
            'timestamp': datetime.utcnow().isoformat(),
            'topic': topic,
            'data': data
        }

    def send_data(self, topic, data):
        try:
            message = self.create_message(topic, data)
            key = f"{self.simulation_id}_{self.sequence_number}"
            
            with self.producer_lock:
                future = self.producer.send(
                    topic=topic,
                    key=key,
                    value=message,
                    timestamp_ms=int(datetime.utcnow().timestamp() * 1000)
                )
                
                record_metadata = future.get(timeout=10)
                self.logger.debug(
                    f"Sent data to {topic} partition {record_metadata.partition} "
                    f"offset {record_metadata.offset}"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Error sending data to {topic}: {e}")
            return False

    def setup_subscribers(self):
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ROS2 subscribers setup
        self.create_subscription(
            Imu, '/mavros/imu/data', self.imu_callback, qos_profile)
        self.create_subscription(
            BatteryState, '/mavros/battery', self.battery_callback, qos_profile)
        self.create_subscription(
            VfrHud, '/mavros/vfr_hud', self.vfr_callback, qos_profile)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', 
            self.local_pose_callback, qos_profile)
        self.create_subscription(
            NavSatFix, '/mavros/global_position/global', 
            self.global_position_callback, qos_profile)
        self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', 
            self.velocity_callback, qos_profile)
        self.create_subscription(
            State, '/mavros/state', self.state_callback, qos_profile)

    def imu_callback(self, msg):
        imu_data = {
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
            'voltage': msg.voltage,
            'current': msg.current,
            'percentage': msg.percentage
        }
        self.send_data('drone_battery', battery_data)

    def vfr_callback(self, msg):
        vfr_data = {
            'airspeed': msg.airspeed,
            'groundspeed': msg.groundspeed,
            'altitude': msg.altitude
        }
        self.send_data('drone_vfr', vfr_data)

    def local_pose_callback(self, msg):
        pose_data = {
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
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }
        self.send_data('drone_global_position', position_data)

    def velocity_callback(self, msg):
        velocity_data = {
            'linear': {
                'x': msg.twist.linear.x,
                'y': msg.twist.linear.y,
                'z': msg.twist.linear.z
            },
            'angular': {
                'x': msg.twist.angular.x,
                'y': msg.twist.angular.y,
                'z': msg.twist.angular.z
            }
        }
        self.send_data('drone_velocity', velocity_data)

    def state_callback(self, msg):
        state_data = {
            'connected': msg.connected,
            'armed': msg.armed,
            'mode': msg.mode
        }
        self.send_data('drone_state', state_data)

    def shutdown(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'producer'):
            self.logger.info(f"Shutting down producer for simulation {self.simulation_id}")
            self.producer.flush()
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
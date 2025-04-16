# Drone Simulation Anomaly Detection System

이 프로젝트는 **WSL (Ubuntu22.04)** 환경에서 **ROS2 Humble** 기반 드론 시뮬레이션 데이터를 실시간으로 처리하여, **Kafka**를 통해 AWS EC2 인스턴스(예: Kafka01, Kafka02, Kafka03)에 데이터를 전송하고 **InfluxDB**에 저장합니다. 저장된 데이터는 500만 건 이상이며, 정상 데이터(90%)와 비정상 데이터(10%)가 혼재되어 있습니다. 이후 `filter.py`로 전처리한 데이터를 **LSTM 기반 AI 모델** (`model.py`)을 통해 이상 감지 예측에 활용하며, 최종 데이터는 **Grafana**를 통해 시각화됩니다.

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [데이터 토픽 및 메시지 구조](#데이터-토픽-및-메시지-구조)
  - [/mavros/imu/data (IMU 데이터)](#mavrosimu-data-imu-데이터)
  - [/mavros/battery (배터리 데이터)](#mavrosbattery-배터리-데이터)
  - [/mavros/local_position/pose (로컬 위치 데이터)](#mavroslocal_positionpose-로컬-위치-데이터)
  - [/mavros/global_position/global (GPS 글로벌 위치 데이터)](#mavrosglobal_positionglobal-gps-글로벌-위치-데이터)
  - [/mavros/state (드론 상태 정보)](#mavrosstate-드론-상태-정보)
  - [/mavros/vfr_hud (비행 상태 정보)](#mavrosvfr_hud-비행-상태-정보)
  - [/mavros/local_position/velocity_local (로컬 속도 정보)](#mavroslocal_positionvelocity_local-로컬-속도-정보)
- [주요 기술 스택](#주요-기술-스택)
- [설치 및 구성](#설치-및-구성)
- [실행 방법](#실행-방법)
- [데이터 처리 및 이상 감지 파이프라인](#데이터-처리-및-이상-감지-파이프라인)
- [Grafana 데이터 시각화](#grafana-데이터-시각화)

---

## 프로젝트 개요

- **목표:**  
  - ROS2 드론 시뮬레이션 데이터를 Kafka를 통해 AWS EC2의 Kafka 클러스터로 전송  
  - 데이터를 InfluxDB에 저장(500만 건 이상, 정상 90% 및 비정상 10% 혼재)  
  - 전처리(`filter.py`) 및 LSTM 모델(`model.py`)을 사용한 이상 감지 예측  
  - Grafana 대시보드를 통한 실시간 데이터 및 이상 감지 결과 시각화

- **주요 기능:**  
  - ROS2를 통한 드론 시뮬레이션 데이터 생성  
  - AWS EC2 상의 Kafka 클러스터(예: Kafka01, Kafka02, Kafka03)를 활용한 실시간 데이터 전송  
  - InfluxDB에 데이터 기록 및 저장  
  - Python을 이용한 전처리와 AI 기반 이상 감지  
  - Grafana를 통한 데이터 모니터링

---

## 시스템 아키텍처

1. **데이터 생성 (ROS2 드론 시뮬레이션):**  
   - 다양한 센서 데이터를 퍼블리시하여 드론의 IMU, 배터리, 위치, 속도 등 정보를 제공합니다.

2. **데이터 전송 (Kafka):**  
   - Kafka Producer가 ROS2 토픽의 데이터를 캡처하여 AWS EC2에 구축된 Kafka 클러스터로 전송합니다.

3. **데이터 저장 (InfluxDB):**  
   - Kafka Consumer 또는 Connector가 데이터를 InfluxDB에 기록합니다.

4. **데이터 처리 및 이상 감지:**  
   - `filter.py`를 통해 데이터 전처리를 수행하고,  
   - `model.py`에서 LSTM 기반 AI 모델이 이상 감지 예측을 수행합니다.

5. **데이터 시각화 (Grafana):**  
   - InfluxDB에 저장된 데이터와 이상 감지 결과를 Grafana 대시보드를 통해 실시간으로 모니터링합니다.

---

## 데이터 토픽 및 메시지 구조

### /mavros/imu/data (IMU 데이터)
- **header.stamp:** IMU 데이터가 퍼블리시된 시간 (초, 나노초)
- **header.frame_id:** 데이터가 참조하는 프레임 (기본적으로 `base_link`)
- **orientation.x, y, z, w:** 드론의 자세를 나타내는 쿼터니언 값 (roll, pitch, yaw 계산 가능)
- **orientation_covariance:** 자세 값 신뢰도를 나타내는 공분산 행렬
- **angular_velocity.x, y, z:** X, Y, Z 축 기준의 각속도 (rad/s)
- **angular_velocity_covariance:** 각속도 값 신뢰도 공분산 행렬
- **linear_acceleration.x, y, z:** X, Y, Z 축 기준의 선속도 (m/s²)
- **linear_acceleration_covariance:** 선속도 값 신뢰도 공분산 행렬

### /mavros/battery (배터리 데이터)
- **header.stamp:** 배터리 데이터가 퍼블리시된 시간
- **voltage:** 배터리 전압 (V)
- **temperature:** 배터리 온도 (°C)
- **current:** 배터리 전류 (A)
- **charge:** 남은 배터리 충전량 (미지원 시 `.nan`)
- **capacity:** 현재 배터리 용량 (미지원 시 `.nan`)
- **design_capacity:** 배터리 설계 용량 (미지원 시 `.nan`)
- **percentage:** 배터리 잔량 비율 (0.0 ~ 1.0)
- **power_supply_status:** 배터리 상태 코드 (예: 2 = 방전 중)
- **power_supply_health:** 배터리 건강 상태 (예: 0 = UNKNOWN, 1 = GOOD)
- **power_supply_technology:** 배터리 기술 (예: 3 = LiPo)
- **present:** 배터리 존재 여부 (true/false)
- **cell_voltage:** 개별 셀 전압 리스트
- **cell_temperature:** 개별 셀 온도 리스트 (미지원 시 빈 리스트)
- **location:** 배터리 위치 (ID 정보)
- **serial_number:** 배터리 고유 일련번호 (없을 경우 빈 문자열)

### /mavros/local_position/pose (로컬 위치 데이터)
- **header.stamp:** 위치 데이터가 퍼블리시된 시간
- **header.frame_id:** 참조 프레임 (기본적으로 `map` 또는 `odom`)
- **pose.position.x, y, z:** 드론의 로컬 좌표계 기준 위치 (m)
- **pose.orientation.x, y, z, w:** 드론의 자세를 나타내는 쿼터니언 값

### /mavros/global_position/global (GPS 글로벌 위치 데이터)
- **header.stamp:** GPS 데이터가 퍼블리시된 시간
- **header.frame_id:** 참조 프레임 (기본적으로 `base_link`)
- **status.status:** GPS 상태 (0: No Fix, 1: 2D Fix, 2: 3D Fix)
- **status.service:** GPS 서비스 타입 (예: 1 = GPS)
- **latitude:** 위도 (°)
- **longitude:** 경도 (°)
- **altitude:** 고도 (m)
- **position_covariance:** GPS 좌표 데이터의 공분산 행렬
- **position_covariance_type:** 공분산 데이터 신뢰도 (0: Unknown, 1: Approximated)

### /mavros/state (드론 상태 정보)
- **header.stamp:** 상태 데이터가 퍼블리시된 시간
- **connected:** MAVROS와 드론의 연결 여부 (true/false)
- **armed:** 드론의 암드 상태 (true = armed, false = disarmed)
- **guided:** 가이드 모드 활성화 여부 (true/false)
- **manual_input:** 수동 입력 활성화 여부 (true/false)
- **mode:** 현재 비행 모드 (예: `AUTO.RTL`)
- **system_status:** 드론 시스템 상태 코드 (예: 3 = Standby, 4 = Active, 5 = Critical)

### /mavros/vfr_hud (비행 상태 정보)
- **header.stamp:** 비행 상태 데이터 퍼블리시 시간
- **airspeed:** 공기 속도를 기반으로 한 드론 속도 (m/s)
- **groundspeed:** 지면 기준 드론 속도 (m/s)
- **heading:** 드론의 방향 (방위각, 0~360°)
- **throttle:** 스로틀 입력 값 (0.0 ~ 1.0)
- **altitude:** 현재 고도 (m)
- **climb:** 상승/하강률 (m/s)

### /mavros/local_position/velocity_local (로컬 속도 정보)
- **header.stamp:** 속도 데이터 퍼블리시 시간
- **header.frame_id:** 참조 프레임 (기본적으로 `base_link`)
- **twist.linear.x, y, z:** 로컬 좌표계 선속도 (m/s)
- **twist.angular.x, y, z:** 로컬 좌표계 각속도 (rad/s)

---

## 주요 기술 스택

- **운영체제:** WSL (Ubuntu22.04)
- **드론 시뮬레이션:** ROS2 Humble
- **메시징:** Apache Kafka (AWS EC2 클러스터: Kafka01, Kafka02, Kafka03)
- **데이터 저장소:** InfluxDB (타임 시리즈 데이터 기록)
- **데이터 처리 및 AI:** Python  
  - 데이터 전처리: `filter.py`  
  - 이상 감지 AI 모델 (LSTM 기반): `model.py`
- **데이터 시각화:** Grafana

---

# 2024DE

### **`/mavros/imu/data` (IMU 데이터)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | IMU 데이터가 퍼블리시된 시간 (초, 나노초). |
| `header.frame_id` | 데이터가 참조하는 프레임 (기본적으로 `base_link`). |
| `orientation.x, y, z, w` | 드론의 자세를 나타내는 쿼터니언 값 (roll, pitch, yaw 계산 가능). |
| `orientation_covariance` | 자세 값의 신뢰도를 나타내는 공분산 행렬. |
| `angular_velocity.x, y, z` | X, Y, Z 축 기준의 각속도 (rad/s). |
| `angular_velocity_covariance` | 각속도 값의 신뢰도를 나타내는 공분산 행렬. |
| `linear_acceleration.x, y, z` | X, Y, Z 축 기준의 선속도 (m/s²). |
| `linear_acceleration_covariance` | 선속도 값의 신뢰도를 나타내는 공분산 행렬. |

---

### **2. `/mavros/battery` (배터리 데이터)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | 배터리 데이터가 퍼블리시된 시간. |
| `voltage` | 배터리 전압 (V). |
| `temperature` | 배터리 온도 (°C). |
| `current` | 배터리 전류 (A). |
| `charge` | 남은 배터리 충전량 (미지원 시 `.nan`). |
| `capacity` | 현재 배터리 용량 (미지원 시 `.nan`). |
| `design_capacity` | 배터리 설계 용량 (미지원 시 `.nan`). |
| `percentage` | 배터리 잔량 비율 (0.0~1.0). |
| `power_supply_status` | 배터리 상태 코드 (예: 2 = 방전 중). |
| `power_supply_health` | 배터리 건강 상태 (예: 0 = UNKNOWN, 1 = GOOD). |
| `power_supply_technology` | 배터리 기술 (예: 3 = LiPo). |
| `present` | 배터리 존재 여부 (true/false). |
| `cell_voltage` | 개별 셀 전압 리스트. |
| `cell_temperature` | 개별 셀 온도 리스트 (미지원 시 빈 리스트). |
| `location` | 배터리 위치 (ID 정보). |
| `serial_number` | 배터리 고유 일련번호 (없을 경우 빈 문자열). |

---

### **3. `/mavros/local_position/pose` (로컬 위치 데이터)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | 위치 데이터가 퍼블리시된 시간. |
| `header.frame_id` | 참조 프레임 (기본적으로 `map` 또는 `odom`). |
| `pose.position.x, y, z` | 드론의 로컬 좌표계 기준 X, Y, Z 위치 (m). |
| `pose.orientation.x, y, z, w` | 드론의 자세를 나타내는 쿼터니언 값. |

---

### **4. `/mavros/global_position/global` (GPS 글로벌 위치 데이터)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | GPS 데이터가 퍼블리시된 시간. |
| `header.frame_id` | 참조 프레임 (기본적으로 `base_link`). |
| `status.status` | GPS 상태 (0: No Fix, 1: 2D Fix, 2: 3D Fix). |
| `status.service` | GPS 서비스 타입 (1: GPS). |
| `latitude` | 위도 (°). |
| `longitude` | 경도 (°). |
| `altitude` | 고도 (m). |
| `position_covariance` | GPS 좌표 데이터의 공분산 행렬. |
| `position_covariance_type` | 공분산 데이터의 신뢰도 (0: Unknown, 1: Approximated). |

---

### **5. `/mavros/state` (드론 상태 정보)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | 상태 데이터가 퍼블리시된 시간. |
| `connected` | MAVROS와 드론 연결 여부 (true/false). |
| `armed` | 드론의 암드 상태 (true: armed, false: disarmed). |
| `guided` | 가이드 모드 활성화 여부 (true/false). |
| `manual_input` | 수동 입력 활성화 여부 (true/false). |
| `mode` | 현재 비행 모드 (예: `AUTO.RTL`). |
| `system_status` | 드론 시스템 상태 코드 (3: Standby, 4: Active, 5: Critical). |

---

### **6. `/mavros/vfr_hud` (비행 상태 정보)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | 비행 상태 데이터가 퍼블리시된 시간. |
| `airspeed` | 공기 속도를 기반으로 한 드론 속도 (m/s). |
| `groundspeed` | 지면 기준 드론 속도 (m/s). |
| `heading` | 드론의 방향(방위각, 0~360°). |
| `throttle` | 스로틀 입력 값 (0.0~1.0). |
| `altitude` | 현재 고도 (m). |
| `climb` | 상승/하강률 (m/s). |

---

### **7. `/mavros/local_position/velocity_local` (로컬 속도 정보)**

| **필드** | **데이터 설명** |
| --- | --- |
| `header.stamp` | 속도 데이터가 퍼블리시된 시간. |
| `header.frame_id` | 참조 프레임 (기본적으로 `base_link`). |
| `twist.linear.x, y, z` | 로컬 좌표계에서의 선속도 (m/s). |
| `twist.angular.x, y, z` | 로컬 좌표계에서의 각속도 (rad/s). |

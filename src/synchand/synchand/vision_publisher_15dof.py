import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from collections import deque

# 1. 3차원 공간에서 세 점(x, y, z) 사이의 각도를 구하는 함수
def calculate_3d_angle(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    
    # 벡터 정규화 (크기를 1로 만듦)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm
    
    # 벡터 내적 (Dot Product)으로 각도 계산
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

# 2. 센서 노이즈를 부드럽게 잡아주는 이동평균 필터 클래스
class MovingAverageFilter:
    def __init__(self, num_channels=15, window_size=5):
        self.num_channels = num_channels
        # 15개의 각도를 각각 저장할 큐(Deque) 생성
        self.queues = [deque(maxlen=window_size) for _ in range(num_channels)]

    def update(self, new_values):
        filtered_values = []
        for i in range(self.num_channels):
            self.queues[i].append(new_values[i])
            # 최근 5개의 평균값을 반환
            avg = sum(self.queues[i]) / len(self.queues[i])
            filtered_values.append(int(avg))
        return filtered_values

def map_angle(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    mapped_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(mapped_value)

class VisionPublisher15DoF(Node):
    def __init__(self):
        super().__init__('vision_publisher_15dof')
        self.publisher_ = self.create_publisher(Int32MultiArray, 'finger_angles', 10)
        
        home_dir = os.path.expanduser('~')
        model_path = os.path.join(home_dir, 'synchand_ws/src/synchand/synchand/hand_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # 15자유도용 이동평균 필터 인스턴스 생성
        self.ma_filter = MovingAverageFilter(num_channels=15, window_size=5)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        raw_angles = [0] * 15

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # 깊이(z) 계산을 위해 모니터 비율을 곱해 3D 좌표계를 맞춤
            h, w, c = frame.shape
            lm_3d = [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks]

            # ----------------------------------------------------
            # [1] 엄지 (Thumb)
            # ----------------------------------------------------
            # 1. 밑동 굽힘 (0번-1번-2번)
            raw_angles[0] = calculate_3d_angle(lm_3d[0], lm_3d[1], lm_3d[2])
            # 2. 마디 굽힘 (2번-3번-4번)
            raw_angles[1] = calculate_3d_angle(lm_3d[2], lm_3d[3], lm_3d[4])
            # 3. 벌어짐 (엄지 2번과 검지 5번이 손목(0번)을 기준으로 벌어진 각도)
            raw_angles[2] = calculate_3d_angle(lm_3d[2], lm_3d[0], lm_3d[5])

            # ----------------------------------------------------
            # [2] 검지 (Index)
            # ----------------------------------------------------
            raw_angles[3] = calculate_3d_angle(lm_3d[0], lm_3d[5], lm_3d[6]) # 밑동
            raw_angles[4] = calculate_3d_angle(lm_3d[5], lm_3d[6], lm_3d[8]) # 마디
            raw_angles[5] = calculate_3d_angle(lm_3d[9], lm_3d[0], lm_3d[5]) # 벌어짐 (중지 대비)

            # ----------------------------------------------------
            # [3] 중지 (Middle)
            # ----------------------------------------------------
            raw_angles[6] = calculate_3d_angle(lm_3d[0], lm_3d[9], lm_3d[10]) # 밑동
            raw_angles[7] = calculate_3d_angle(lm_3d[9], lm_3d[10], lm_3d[12])# 마디
            raw_angles[8] = 90 # 중지는 손의 중심축이므로 벌어짐을 고정하거나, 구조에 따라 생략

            # ----------------------------------------------------
            # [4] 약지 (Ring)
            # ----------------------------------------------------
            raw_angles[9] = calculate_3d_angle(lm_3d[0], lm_3d[13], lm_3d[14])# 밑동
            raw_angles[10] = calculate_3d_angle(lm_3d[13], lm_3d[14], lm_3d[16])# 마디
            raw_angles[11] = calculate_3d_angle(lm_3d[9], lm_3d[0], lm_3d[13])# 벌어짐 (중지 대비)

            # ----------------------------------------------------
            # [5] 소지 (Pinky)
            # ----------------------------------------------------
            raw_angles[12] = calculate_3d_angle(lm_3d[0], lm_3d[17], lm_3d[18])# 밑동
            raw_angles[13] = calculate_3d_angle(lm_3d[17], lm_3d[18], lm_3d[20])# 마디
            raw_angles[14] = calculate_3d_angle(lm_3d[13], lm_3d[0], lm_3d[17])# 벌어짐 (약지 대비)

            # 각도들을 모터 제어용(0~180)으로 매핑 (각 관절마다 in_min, in_max 테스트 필요)
            mapped_angles = [map_angle(a, 10, 160, 0, 180) for a in raw_angles]

            # 필터 적용하여 노이즈 제거
            final_angles = self.ma_filter.update(mapped_angles)

            # 2D 화면 시각화
            for lm in lm_3d:
                cv2.circle(frame, (int(lm[0]), int(lm[1])), 4, (255, 0, 0), -1)

            msg = Int32MultiArray()
            msg.data = final_angles
            self.publisher_.publish(msg)
            self.get_logger().info(f'15DoF Angles: {final_angles}')

        cv2.imshow('SyncHand Vision (15DoF)', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher15DoF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'cap'):
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
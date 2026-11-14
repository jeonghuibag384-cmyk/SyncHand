import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# 세 점 사이의 각도를 구하는 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

def map_angle(value, in_min, in_max, out_min, out_max):
    # 인식된 값이 범위를 벗어나지 않게 제한(Clamping)
    value = max(min(value, in_max), in_min)
    
    # 입력 범위를 출력 범위로 선형 변환
    mapped_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(mapped_value)

class VisionPublisher(Node):
    def __init__(self):
        super().__init__('vision_publisher')
        self.publisher_ = self.create_publisher(Int32MultiArray, 'finger_angles', 10)
        
        # 1. 모델 파일 경로 설정 (방금 다운로드한 파일)
        # ROS 2 실행 시 경로 문제를 방지하기 위해 절대 경로 사용
        home_dir = os.path.expanduser('~')
        model_path = os.path.join(home_dir, 'synchand_ws/src/synchand/synchand/hand_landmarker.task')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return

        # 2. 최신 MediaPipe Tasks API 초기화
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # OpenCV(BGR) 이미지를 MediaPipe용 규격(RGB)으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 손 인식 수행
        detection_result = self.detector.detect(mp_image)
        angles = [0, 0, 0, 0, 0]

        if detection_result.hand_landmarks:
            # 첫 번째 인식된 손의 랜드마크 데이터 가져오기
            hand_landmarks = detection_result.hand_landmarks[0]
            h, w, c = frame.shape
            
            # 정규화된 좌표(0~1)를 화면 픽셀 좌표로 변환
            landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks]

            # 1. 먼저 순수하게 인식된 날것의 각도(Raw Angle)를 계산합니다.
            raw_thumb = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            raw_index = calculate_angle(landmarks[5], landmarks[6], landmarks[8])
            raw_middle = calculate_angle(landmarks[9], landmarks[10], landmarks[12])
            raw_ring = calculate_angle(landmarks[13], landmarks[14], landmarks[16])
            raw_pinky = calculate_angle(landmarks[17], landmarks[18], landmarks[20])

            # 2. 날것의 각도를 서보모터 제어용(0~180도)으로 매핑합니다.
            # map_angle(현재값, 내 손의 최소각, 내 손의 최대각, 모터 최소각, 모터 최대각)
                
            # 엄지는 보통 100~170 사이에서 움직이므로 범위를 좁게 잡아줍니다.
            # (주의: 꽉 쥐었을 때 값이 작아지면 in_min을 작게, 반대면 반대로 세팅해야 합니다)
            angles[0] = map_angle(raw_thumb, 100, 170, 0, 180)
                
            # 나머지 손가락은 대체로 30~170도 사이에서 움직입니다.
            angles[1] = map_angle(raw_index, 30, 170, 0, 180)
            angles[2] = map_angle(raw_middle, 30, 170, 0, 180)
            angles[3] = map_angle(raw_ring, 30, 170, 0, 180)
            angles[4] = map_angle(raw_pinky, 30, 170, 0, 180)

            # 화면에 인식된 관절 그리기 (관절 점 찍기)
            for lm in landmarks:
                cv2.circle(frame, (lm[0], lm[1]), 5, (0, 255, 0), -1)

        # 계산된 각도를 ROS 2 토픽으로 전송
        msg = Int32MultiArray()
        msg.data = angles
        self.publisher_.publish(msg)
        self.get_logger().info(f'전송 각도: {angles}')

        cv2.imshow('SyncHand Vision (Tasks API)', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher()
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
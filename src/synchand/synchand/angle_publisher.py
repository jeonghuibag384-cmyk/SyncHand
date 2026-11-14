import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import random

class AnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_publisher')
        # 'finger_angles'라는 토픽으로 Int32MultiArray 메시지를 발행
        self.publisher_ = self.create_publisher(Int32MultiArray, 'finger_angles', 10)
        # 0.5초마다 timer_callback 함수 실행
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        msg = Int32MultiArray()
        # 엄지, 검지, 중지, 약지, 소지의 가상 각도 (0~180 사이 랜덤 값)
        msg.data = [random.randint(0, 180) for _ in range(5)]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Angles: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = AnglePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
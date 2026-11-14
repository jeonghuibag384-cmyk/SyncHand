import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

class AngleSubscriber(Node):
    def __init__(self):
        super().__init__('angle_subscriber')
        # 'finger_angles' 토픽을 구독하고, 데이터가 오면 listener_callback 실행
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'finger_angles',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received Angles: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = AngleSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
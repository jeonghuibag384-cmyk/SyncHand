"""SyncHand: 15-channel hand-pose estimates from MediaPipe world landmarks.

ROS 2 Humble / MediaPipe Tasks API. Publishes RAW geometric estimates in
[0, 180] degrees on /finger_angles_15dof (Int32MultiArray). These are not
calibrated servo setpoints and must NOT be sent directly to actuators.

Output order (three values per digit):
  0 thumb CMC out-of-palm-plane proxy, 1 thumb IP flex, 2 thumb spread
  3 index MCP flex proxy, 4 index PIP flex, 5 index spread
  6 middle MCP flex proxy, 7 middle PIP flex, 8 middle spread
  9 ring MCP flex proxy, 10 ring PIP flex, 11 ring spread
 12 pinky MCP flex proxy, 13 pinky PIP flex, 14 pinky spread

Proxies are useful hand-pose features, not independent anatomical joint angles.
"""

from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EPS = 1e-9
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def unit(v):
    norm = float(np.linalg.norm(v))
    if norm < EPS:
        raise ValueError('Degenerate hand vector')
    return v / norm


def joint_flex(a, b, c):
    """0 = straight, ~90 = bent at B (unsigned geometric flex proxy)."""
    v1, v2 = unit(a - b), unit(c - b)
    interior = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return float(np.clip(180.0 - interior, 0.0, 180.0))


def signed_spread(direction, lateral, forward, normal):
    """Signed angle of a projected vector relative to palm-forward axis."""
    planar = direction - np.dot(direction, normal) * normal
    planar = unit(planar)
    return float(np.degrees(np.arctan2(np.dot(planar, lateral),
                                       np.dot(planar, forward))))


def palm_frame(p):
    """Build hand-fixed axes from wrist, index MCP, middle MCP, pinky MCP."""
    # Lateral increases index -> pinky; forward runs wrist -> middle MCP.
    lateral = unit(p[17] - p[5])
    forward_seed = p[9] - p[0]
    forward = unit(forward_seed - np.dot(forward_seed, lateral) * lateral)
    normal = unit(np.cross(lateral, forward))
    return lateral, forward, normal


def angle_features(p):
    """Return 15 raw features in degrees; no servo calibration is applied."""
    lateral, forward, normal = palm_frame(p)
    out = []

    # Thumb: CMC out-of-plane movement, IP flexion, CMC spread in palm plane.
    thumb_base = unit(p[2] - p[1])
    thumb_cmc_out = abs(np.degrees(np.arcsin(
        np.clip(np.dot(thumb_base, normal), -1.0, 1.0))))
    thumb_ip = joint_flex(p[2], p[3], p[4])
    thumb_spread = abs(signed_spread(thumb_base, lateral, forward, normal))
    out.extend([thumb_cmc_out, thumb_ip, thumb_spread])

    # Non-thumb fingers: MCP elevation, PIP flexion, signed spread relative
    # to palm-forward axis. Middle spread is measured, never hardcoded.
    for mcp, pip, dip in [(5, 6, 7), (9, 10, 11),
                          (13, 14, 15), (17, 18, 19)]:
        proximal = unit(p[pip] - p[mcp])
        mcp_elevation = abs(np.degrees(np.arcsin(
            np.clip(np.dot(proximal, normal), -1.0, 1.0))))
        pip_flex = joint_flex(p[mcp], p[pip], p[dip])
        spread = signed_spread(proximal, lateral, forward, normal)
        # +90 offset encodes signed [-90, 90] as [0, 180].
        out.extend([mcp_elevation, pip_flex, spread + 90.0])

    return np.clip(np.asarray(out, dtype=float), 0.0, 180.0)


class VisionWorld15DoF(Node):
    def __init__(self):
        super().__init__('vision_world_15dof')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('model_path', str(
            Path.home() / 'synchand_ws/src/synchand/synchand/hand_landmarker.task'))
        self.declare_parameter('ema_alpha', 0.35)
        self.declare_parameter('fps', 15.0)

        camera_index = int(self.get_parameter('camera_index').value)
        model_path = Path(self.get_parameter('model_path').value).expanduser()
        self.alpha = float(self.get_parameter('ema_alpha').value)
        fps = float(self.get_parameter('fps').value)
        if not (0.0 < self.alpha <= 1.0 and fps > 0):
            raise ValueError('ema_alpha must be in (0,1], fps must be > 0')
        if not model_path.is_file():
            raise FileNotFoundError(f'MediaPipe model not found: {model_path}')

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.detector.close()
            raise RuntimeError(f'Could not open camera index {camera_index}')

        self.publisher = self.create_publisher(
            Int32MultiArray, '/finger_angles_15dof', 10)
        self.filtered = None
        self.last_log = 0.0
        self.timer = self.create_timer(1.0 / fps, self.update)
        self.get_logger().info('Publishing 15 raw channels on /finger_angles_15dof')

    def update(self):
        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning('Camera frame unavailable')
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        now = time.monotonic()

        if result.hand_world_landmarks and result.hand_landmarks:
            coords = np.array([
                [lm.x, lm.y, lm.z]
                for lm in result.hand_world_landmarks[0]
            ], dtype=np.float64)
            try:
                raw = angle_features(coords)
            except ValueError:
                self.filtered = None
                self.get_logger().warning('Degenerate geometry: skipping frame')
                raw = None

            if raw is not None and np.all(np.isfinite(raw)):
                self.filtered = (raw if self.filtered is None else
                                 self.alpha * raw + (1.0 - self.alpha) * self.filtered)
                values = np.rint(self.filtered).astype(int).tolist()
                msg = Int32MultiArray()
                msg.data = values
                self.publisher.publish(msg)
                if now - self.last_log >= 1.0:
                    self.get_logger().info(f'15DoF raw (degrees): {values}')
                    self.last_log = now

            height, width = frame.shape[:2]
            xy = [(int(lm.x * width), int(lm.y * height))
                  for lm in result.hand_landmarks[0]]
            for a, b in CONNECTIONS:
                cv2.line(frame, xy[a], xy[b], (0, 200, 0), 2)
            for x, y in xy:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        else:
            # No hand: never publish zero commands; reset smoothing for reacquisition.
            self.filtered = None
            if now - self.last_log >= 2.0:
                self.get_logger().warning('No hand detected; not publishing')
                self.last_log = now

        cv2.putText(frame, 'SyncHand World 15DoF - q: quit',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow('SyncHand World 15DoF', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def close(self):
        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionWorld15DoF()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

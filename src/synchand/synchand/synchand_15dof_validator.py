#!/usr/bin/env python3
"""SyncHand 15-channel feature validator with an approximate forward-kinematic hand.

Displays a webcam, the MediaPipe world landmark hand, and an FK reconstruction
from the SAME 15 angle features used by vision_world_15dof.py. It deliberately
uses observed palm/knuckle anchors and bone lengths, since those are not
encoded in the 15 angles. The model also assumes coupled DIP motion and a
straight thumb MCP. Thus RMS error is *model reconstruction error*, not sensor
accuracy or a ground-truth measure.

Requirements: Python 3, numpy, opencv-python, mediapipe (Tasks API),
             hand_landmarker.task. No ROS dependencies.

Usage:
  python3 synchand_15dof_validator.py --model ~/synchand_ws/src/synchand/synchand/hand_landmarker.task
  python3 synchand_15dof_validator.py --demo

Controls: q quit, space pause, a/d rotate, w/s tilt, r reset view,
          1 invert MCP normal direction, 2 invert PIP/DIP direction,
          3 invert thumb-side direction.
"""

import argparse
from pathlib import Path
import math
import time

import cv2
import numpy as np

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_NAMES = ('index', 'middle', 'ring', 'pinky')
FINGER_COLORS = ((85, 220, 255), (120, 230, 120), (255, 190, 100),
                 (235, 125, 235))
EPS = 1e-9


def unit(v):
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ValueError('Degenerate or missing landmark geometry')
    return np.asarray(v, dtype=np.float64) / n


def joint_flex(a, b, c):
    v1, v2 = unit(a - b), unit(c - b)
    interior = math.degrees(math.acos(float(np.clip(np.dot(v1, v2), -1, 1))))
    return float(np.clip(180 - interior, 0, 180))


def palm_frame(p):
    # Identical orientation convention to vision_world_15dof.py.
    lateral = unit(p[17] - p[5])
    seed = p[9] - p[0]
    forward = unit(seed - np.dot(seed, lateral) * lateral)
    normal = unit(np.cross(lateral, forward))
    return lateral, forward, normal


def signed_spread(direction, lateral, forward, normal):
    planar = unit(direction - np.dot(direction, normal) * normal)
    return math.degrees(math.atan2(float(np.dot(planar, lateral)),
                                   float(np.dot(planar, forward))))


def angle_features(p):
    """Same 15-channel definitions/order as the existing SyncHand node."""
    lateral, forward, normal = palm_frame(p)
    thumb_dir = unit(p[2] - p[1])
    thumb_out = abs(math.degrees(math.asin(float(np.clip(np.dot(thumb_dir, normal), -1, 1)))))
    thumb_ip = joint_flex(p[2], p[3], p[4])
    thumb_spread = abs(signed_spread(thumb_dir, lateral, forward, normal))
    output = [thumb_out, thumb_ip, thumb_spread]
    for mcp, pip, dip, tip in FINGERS:
        proximal = unit(p[pip] - p[mcp])
        elev = abs(math.degrees(math.asin(float(np.clip(np.dot(proximal, normal), -1, 1)))))
        pip_flex = joint_flex(p[mcp], p[pip], p[dip])
        spread = signed_spread(proximal, lateral, forward, normal)
        output.extend((elev, pip_flex, spread + 90.0))
    thumb_mcp_flex = joint_flex(p[1], p[2], p[3])
    thumb_ip_flex = joint_flex(p[2], p[3], p[4])
    print(
        f"Thumb MCP: {thumb_mcp_flex:.1f}°, "
        f"IP: {thumb_ip_flex:.1f}°"
    )    
    return np.clip(np.asarray(output, dtype=float), 0, 180)
    

def direction(elevation_deg, yaw_deg, lateral, forward, normal):
    elev, yaw = np.radians([elevation_deg, yaw_deg])
    in_plane = math.sin(yaw) * lateral + math.cos(yaw) * forward
    return unit(math.cos(elev) * in_plane + math.sin(elev) * normal)


def reconstruct(p, angles, mcp_sign=1, pip_sign=1, thumb_sign=-1,
                dip_ratio=0.65):
    """Approximate 3D FK, using ONLY 15 angles for bending directions.

    Auxiliary observed inputs: palm/MCP anchors, wrist/thumb CMC anchor, and
    bone lengths. These are NOT contained in the 15 numbers. Signs and DIP
    coupling are assumptions; they can be adjusted with keyboard toggles.
    """
    lateral, forward, normal = palm_frame(p)
    q = np.zeros_like(p)
    # Palm landmarks are anchored to the detected shape, not reconstructed.
    for j in (0, 1, 5, 9, 13, 17):
        q[j] = p[j]

    # Thumb: feature channels = [CMC out-of-plane, IP flex, unsigned spread].
    # Thumb MCP rotation is UNOBSERVED in these 15 channels and assumed zero.
    th_elev = mcp_sign * float(angles[0])
    th_yaw = thumb_sign * float(angles[2])
    base_dir = direction(th_elev, th_yaw, lateral, forward, normal)
    q[2] = q[1] + np.linalg.norm(p[2] - p[1]) * base_dir
    q[3] = q[2] + np.linalg.norm(p[3] - p[2]) * base_dir
    tip_dir = direction(th_elev + pip_sign * float(angles[1]), th_yaw,
                        lateral, forward, normal)
    q[4] = q[3] + np.linalg.norm(p[4] - p[3]) * tip_dir

    for finger_idx, (mcp, pip, dip, tip) in enumerate(FINGERS):
        a = 3 + 3 * finger_idx
        elev = mcp_sign * float(angles[a])
        pip_flex = float(angles[a + 1])
        yaw = float(angles[a + 2]) - 90.0
        q[pip] = q[mcp] + np.linalg.norm(p[pip] - p[mcp]) * direction(
            elev, yaw, lateral, forward, normal)
        q[dip] = q[pip] + np.linalg.norm(p[dip] - p[pip]) * direction(
            elev + pip_sign * pip_flex, yaw, lateral, forward, normal)
        q[tip] = q[dip] + np.linalg.norm(p[tip] - p[dip]) * direction(
            elev + pip_sign * (pip_flex * (1 + dip_ratio)), yaw,
            lateral, forward, normal)
    return q


def error_metrics(p, q):
    # Ignore palm anchors, which are copied exactly.
    ids = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
    err = np.linalg.norm((q[ids] - p[ids]) * 1000, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(err)), err


def camera_basis(yaw_deg, pitch_deg):
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    # Orthographic projection in the palm-local coordinate frame.
    view = np.array([math.sin(yaw) * math.cos(pitch),
                     -math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = unit(np.cross([0, 0, 1], view))
    up = unit(np.cross(view, right))
    return right, up, view


def project_points(points, real, yaw, pitch, size=(480, 520)):
    lateral, forward, normal = palm_frame(real)
    hand_axes = np.column_stack([lateral, forward, normal])
    local = (points - real[0]) @ hand_axes
    right, up, view = camera_basis(yaw, pitch)
    uv = np.column_stack([local @ right, local @ up])
    # Stable scale based on wrist-to-middle-MCP length, not per-hand bbox.
    hand_scale = float(np.linalg.norm(real[9] - real[0]))
    px_per_meter = min(size) * 0.23 / max(hand_scale, 0.02)
    xy = uv * px_per_meter
    xy[:, 1] *= -1
    xy += np.array([size[0] * 0.48, size[1] * 0.68])
    return np.rint(xy).astype(int)


def draw_skeleton(canvas, coords, main_color=(105, 225, 155),
                  anchor_color=(210, 210, 210)):
    for a, b in CONNECTIONS:
        finger_id = next((i for i, f in enumerate(FINGERS) if a in f and b in f), -1)
        color = FINGER_COLORS[finger_id] if finger_id >= 0 else main_color
        cv2.line(canvas, tuple(coords[a]), tuple(coords[b]), color, 3, cv2.LINE_AA)
    for j, (x, y) in enumerate(coords):
        color = anchor_color if j in (0, 1, 5, 9, 13, 17) else (230, 235, 240)
        cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)


def text_line(image, text, x, y, scale=0.52, color=(240, 240, 240)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def demo_landmarks(t):
    """Synthetic, clearly labelled demonstration (not real tracking)."""
    p = np.zeros((21, 3), dtype=float)
    p[0] = [0, 0, 0]
    p[5], p[9], p[13], p[17] = [
        [-0.034, 0.074, 0], [-0.012, 0.087, 0],
        [0.012, 0.085, 0], [0.034, 0.069, 0]]
    p[1] = [-0.044, 0.024, 0]
    lateral, forward, normal = palm_frame(p)
    t = 0.5 + 0.5 * math.sin(t * 1.4)
    angle = np.array([15, 15 + 30 * t, 55,
                      12 + 40 * t, 12 + 65 * t, 75,
                      15 + 35 * t, 15 + 70 * t, 90,
                      18 + 25 * t, 15 + 65 * t, 105,
                      18 + 20 * t, 20 + 50 * t, 112], dtype=float)
    # Fill lengths for synthetic hand so reconstruct() can use them.
    for mcp, pip, dip, tip in FINGERS:
        p[pip] = p[mcp] + 0.039 * forward
        p[dip] = p[pip] + 0.025 * forward
        p[tip] = p[dip] + 0.018 * forward
    p[2] = p[1] + 0.028 * forward
    p[3] = p[2] + 0.024 * forward
    p[4] = p[3] + 0.020 * forward
    return reconstruct(p, angle), angle


def render(frame, p, q, angles, fps, yaw, pitch, rms, maximum,
           mcp_sign, pip_sign, thumb_sign, demo, paused, note):
    out_w, out_h = 480, 690
    bg = (24, 29, 36)
    panels = [np.full((out_h, out_w, 3), bg, dtype=np.uint8) for _ in range(3)]
    if frame is not None:
        h, w = frame.shape[:2]
        scale = min(out_w / w, 470 / h)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ox = (out_w - resized.shape[1]) // 2
        oy = 68 + (470 - resized.shape[0]) // 2
        panels[0][oy:oy+resized.shape[0], ox:ox+resized.shape[1]] = resized

    text_line(panels[0], 'RGB CAMERA' if not demo else 'SYNTHETIC DEMO', 16, 33, 0.75)
    text_line(panels[1], 'MEDIAPIPE WORLD (REFERENCE)', 12, 33, 0.64)
    text_line(panels[2], '15-ANGLE FK (RECONSTRUCTED)', 12, 33, 0.64)

    if p is not None and q is not None:
        xy1 = project_points(p, p, yaw, pitch)
        xy2 = project_points(q, p, yaw, pitch)
        xy1[:, 1] += 54
        xy2[:, 1] += 54
        draw_skeleton(panels[1], xy1)
        draw_skeleton(panels[2], xy2)
        for panel in panels[1:]:
            text_line(panel, f'View yaw {yaw:.0f} / pitch {pitch:.0f} deg', 14, 545, 0.47)
        text_line(panels[2], f'FK RMS {rms:.1f} mm  |  Max {maximum:.1f} mm',
                  12, 572, 0.55, (125, 230, 250))
        groups = ('TH', 'IN', 'MI', 'RI', 'PI')
        for i in range(5):
            vals = angles[i*3:i*3+3]
            text_line(panels[0],
                      f'{groups[i]} : {vals[0]:5.1f}  {vals[1]:5.1f}  {vals[2]:5.1f}',
                      14, 548+i*16, 0.40)
    else:
        text_line(panels[1], 'No hand / no valid landmarks', 70, 300, 0.65)
        text_line(panels[2], 'Waiting for valid 15D pose', 70, 300, 0.65)

    combined = np.hstack(panels)
    status = 'PAUSED' if paused else ('DEMO' if demo else 'LIVE')
    controls = f'{status} | q quit | SPACE pause | a/d rotate | w/s tilt | r reset | 1 MCP sign | 2 PIP sign | 3 thumb sign'
    cv2.putText(combined, controls, (12, 666), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (230, 230, 230), 1, cv2.LINE_AA)
    if note:
        text_line(combined, note, 15, 524, 0.52, (90, 235, 255))
    return combined


def main():
    ap = argparse.ArgumentParser(description='SyncHand 15D FK validator')
    ap.add_argument('--model', type=Path, default=Path.home() /
                    'synchand_ws/src/synchand/synchand/hand_landmarker.task')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--demo', action='store_true',
                    help='run a synthetic preview without webcam or MediaPipe')
    ap.add_argument('--fps', type=float, default=12.0)
    args = ap.parse_args()

    if args.fps <= 0:
        ap.error('--fps must be > 0')
    cap = detector = None
    mp = None
    if not args.demo:
        if not args.model.expanduser().is_file():
            ap.error(f'MediaPipe task model not found: {args.model}')
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model.expanduser())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            detector.close()
            raise RuntimeError(f'Camera index {args.camera} not available')

    yaw, pitch = 0.0, 80.0
    mcp_sign, pip_sign, thumb_sign = 1, 1, -1
    last_good = None
    paused = False
    last = time.monotonic()
    fps_smoothed = 0.0
    note = ''
    print('SyncHand validator started: q quit, space pause, a/d w/s view, 1/2/3 sign')
    try:
        while True:
            started = time.monotonic()
            if not paused or last_good is None:
                frame, p, provided_angles = None, None, None
                if args.demo:
                    p, provided_angles = demo_landmarks(started)
                else:
                    ok, frame = cap.read()
                    if ok:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                        if result.hand_world_landmarks and result.hand_landmarks:
                            p = np.array([(v.x, v.y, v.z)
                                          for v in result.hand_world_landmarks[0]], dtype=np.float64)
                            h, w = frame.shape[:2]
                            xy = [(int(v.x*w), int(v.y*h)) for v in result.hand_landmarks[0]]
                            for a, b in CONNECTIONS:
                                cv2.line(frame, xy[a], xy[b], (50, 235, 105), 2, cv2.LINE_AA)
                            for x, y in xy:
                                cv2.circle(frame, (x, y), 3, (50, 215, 250), -1)
                if p is not None:
                    try:
                        angles = provided_angles if provided_angles is not None else angle_features(p)
                        last_good = (frame, p, angles)
                        note = ''
                    except ValueError as exc:
                        last_good = None
                        note = str(exc)
                else:
                    last_good = None
            if last_good is not None:
                frame, p, angles = last_good
                q = reconstruct(p, angles, mcp_sign, pip_sign, thumb_sign)
                rms, maximum, _ = error_metrics(p, q)
            else:
                frame, p, angles, q, rms, maximum = None, None, None, None, 0, 0
            elapsed = max(started - last, 1e-4)
            last = started
            fps_smoothed = fps_smoothed * 0.85 + (1 / elapsed) * 0.15
            canvas = render(frame, p, q, angles, fps_smoothed, yaw, pitch,
                            rms, maximum, mcp_sign, pip_sign, thumb_sign,
                            args.demo, paused, note)
            cv2.imshow('SyncHand 15D FK Validator', canvas)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                yaw -= 10
            elif key == ord('d'):
                yaw += 10
            elif key == ord('w'):
                pitch = min(80, pitch + 10)
            elif key == ord('s'):
                pitch = max(-80, pitch - 10)
            elif key == ord('r'):
                yaw, pitch = 0.0, 80.0
            elif key == ord('1'):
                mcp_sign *= -1
            elif key == ord('2'):
                pip_sign *= -1
            elif key == ord('3'):
                thumb_sign *= -1
            wait = 1 / args.fps - (time.monotonic() - started)
            if wait > 0:
                time.sleep(wait)
    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

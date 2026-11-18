#!/usr/bin/env python3
"""SyncHand: compare original 15-channel FK against FK + measured thumb MCP.

Real-time displays:
  (1) RGB camera landmarks
  (2) MediaPipe world landmarks (reference estimate)
  (3) Original 15-angle FK (thumb MCP assumed straight)
  (4) 15-angle FK + independent thumb MCP bending measurement

The same 15 feature definitions are used as vision_world_15dof.py.
The additional MCP flexion is measured as joint_flex(p[1], p[2], p[3]).
Only visualization changes; no ROS commands are published, no motors move.

IMPORTANT: This comparison deliberately uses observed palm anchors and segment
lengths for both FK reconstructions. It is NOT a pure 15/16-value hand generator
and MediaPipe world landmarks are NOT ground-truth joint locations.

Keys: q/Esc exit, space pause, a/d yaw, w/s pitch, r reset view,
      1 reverse all finger MCP elevation, 2 reverse PIP/IP bend,
      3 reverse thumb spread, 4 reverse added thumb MCP bend.

Examples:
 python3 synchand_thumb_fk_compare.py --model hand_landmarker.task
 python3 synchand_thumb_fk_compare.py --demo
"""

import argparse
import math
import time
from pathlib import Path

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
FINGER_COLORS = ((85, 220, 255), (120, 230, 120),
                 (255, 190, 100), (235, 125, 235))
EPS = 1e-9


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ValueError('Degenerate hand landmark geometry')
    return v / n


def joint_flex(a, b, c):
    """Unsigned segment bend at b: 0 deg when straight."""
    u, v = unit(b - a), unit(c - b)
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))


def palm_frame(p):
    """Same palm-local frame as the prior SyncHand 15-channel node."""
    lateral = unit(p[17] - p[5])
    forward_seed = p[9] - p[0]
    forward = unit(forward_seed - np.dot(forward_seed, lateral) * lateral)
    normal = unit(np.cross(lateral, forward))
    return lateral, forward, normal


def signed_spread(v, lateral, forward, normal):
    planar = unit(v - np.dot(v, normal) * normal)
    return math.degrees(math.atan2(float(np.dot(planar, lateral)),
                                   float(np.dot(planar, forward))))


def angle_features(p):
    """Identical 15-channel order and geometric meaning to prior validator."""
    lateral, forward, normal = palm_frame(p)
    thumb_dir = unit(p[2] - p[1])
    thumb_out = abs(math.degrees(math.asin(float(
        np.clip(np.dot(thumb_dir, normal), -1, 1)))))
    thumb_ip = joint_flex(p[2], p[3], p[4])
    thumb_spread = abs(signed_spread(thumb_dir, lateral, forward, normal))
    out = [thumb_out, thumb_ip, thumb_spread]
    for mcp, pip, dip, tip in FINGERS:
        proximal = unit(p[pip] - p[mcp])
        elev = abs(math.degrees(math.asin(float(
            np.clip(np.dot(proximal, normal), -1, 1)))))
        pip_flex = joint_flex(p[mcp], p[pip], p[dip])
        spread = signed_spread(proximal, lateral, forward, normal)
        out.extend((elev, pip_flex, spread + 90.0))
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 180.0)


def direction(elevation_deg, yaw_deg, lateral, forward, normal):
    elev, yaw = np.radians([elevation_deg, yaw_deg])
    in_plane = math.sin(yaw) * lateral + math.cos(yaw) * forward
    return unit(math.cos(elev) * in_plane + math.sin(elev) * normal)


def reconstruct_pair(p, angles, mcp_sign=1, pip_sign=1,
                     thumb_sign=-1, thumb_mcp_sign=1, dip_ratio=0.65):
    """Build 15-FK and thumb-MCP-augmented FK from identical inputs/assumptions.

    Both use observed wrist, thumb CMC, knuckle anchors & bone lengths.
    The only difference is 1 extra unsigned bend magnitude at thumb MCP,
    rotated in the same thumb elevation plane (fixed-axis approximation).
    The sign is unknown and can be toggled with key 4.
    """
    lateral, forward, normal = palm_frame(p)
    base = np.zeros_like(p)
    for j in (0, 1, 5, 9, 13, 17):
        base[j] = p[j]

    th_elev = mcp_sign * float(angles[0])
    th_yaw = thumb_sign * float(angles[2])
    base_dir = direction(th_elev, th_yaw, lateral, forward, normal)
    len12 = float(np.linalg.norm(p[2] - p[1]))
    len23 = float(np.linalg.norm(p[3] - p[2]))
    len34 = float(np.linalg.norm(p[4] - p[3]))

    base[2] = base[1] + len12 * base_dir

    # Original 15-feature FK: thumb MCP is forced to zero bend.
    old = base.copy()
    old[3] = old[2] + len23 * base_dir
    old_ip_dir = direction(th_elev + pip_sign * float(angles[1]),
                           th_yaw, lateral, forward, normal)
    old[4] = old[3] + len34 * old_ip_dir

    # Extra observed thumb MCP magnitude; sign/axis are MODEL assumptions.
    measured_thumb_mcp = joint_flex(p[1], p[2], p[3])
    new = base.copy()
    mcp_elev = th_elev + thumb_mcp_sign * measured_thumb_mcp
    mcp_dir = direction(mcp_elev, th_yaw, lateral, forward, normal)
    new[3] = new[2] + len23 * mcp_dir
    new_ip_dir = direction(mcp_elev + pip_sign * float(angles[1]),
                           th_yaw, lateral, forward, normal)
    new[4] = new[3] + len34 * new_ip_dir

    # Other four fingers unchanged, so whole-hand RMS isolates thumb effect.
    for finger_idx, (mcp, pip, dip, tip) in enumerate(FINGERS):
        a = 3 + 3 * finger_idx
        elev = mcp_sign * float(angles[a])
        flex = float(angles[a + 1])
        yaw = float(angles[a + 2]) - 90.0
        lengths = [float(np.linalg.norm(p[b] - p[a])) for a, b in
                   ((mcp, pip), (pip, dip), (dip, tip))]
        chain = [
            direction(elev, yaw, lateral, forward, normal),
            direction(elev + pip_sign * flex, yaw, lateral, forward, normal),
            direction(elev + pip_sign * flex * (1 + dip_ratio),
                      yaw, lateral, forward, normal),
        ]
        for q in (old, new):
            for origin, dest, length, axis in zip(
                    (mcp, pip, dip), (pip, dip, tip), lengths, chain):
                q[dest] = q[origin] + length * axis
    return old, new, measured_thumb_mcp


def thumb_error(p, q):
    """RMS of thumb joints 2,3,4 and point-wise TIP discrepancy, mm."""
    err = np.linalg.norm((q[[2, 3, 4]] - p[[2, 3, 4]]) * 1000, axis=1)
    return float(np.sqrt(np.mean(err**2))), float(err[-1]), err


def all_finger_error(p, q):
    ids = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12,
                    14, 15, 16, 18, 19, 20])
    err = np.linalg.norm((q[ids] - p[ids]) * 1000, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def camera_basis(yaw_deg, pitch_deg):
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    view = np.array([math.sin(yaw) * math.cos(pitch),
                     -math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = unit(np.cross([0, 0, 1], view))
    up = unit(np.cross(view, right))
    return right, up


def project_points(points, reference, yaw, pitch, width=560, view_height=275):
    lateral, forward, normal = palm_frame(reference)
    axes = np.column_stack([lateral, forward, normal])
    local = (points - reference[0]) @ axes
    right, up = camera_basis(yaw, pitch)
    uv = np.column_stack([local @ right, local @ up])
    hand_scale = float(np.linalg.norm(reference[9] - reference[0]))
    px_per_meter = min(width, view_height) * 0.24 / max(hand_scale, 0.02)
    xy = uv * px_per_meter
    xy[:, 1] *= -1
    xy += np.array([width * 0.50, view_height * 0.78 + 40])
    return np.rint(xy).astype(int)


def draw_skeleton(canvas, xy, emphasize_thumb=True):
    for a, b in CONNECTIONS:
        finger_i = next((i for i, finger in enumerate(FINGERS)
                         if a in finger and b in finger), -1)
        color = FINGER_COLORS[finger_i] if finger_i >= 0 else (110, 218, 163)
        if emphasize_thumb and (a, b) in ((0, 1), (1, 2), (2, 3), (3, 4)):
            color = (45, 205, 255)
        cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), color,
                 4 if finger_i == -1 else 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(xy):
        color = (70, 205, 250) if i in (2, 3, 4) else (230, 230, 234)
        cv2.circle(canvas, (int(x), int(y)), 5 if i in (2, 3, 4) else 3,
                   color, -1, cv2.LINE_AA)


def line(img, text, x, y, scale=.51, color=(235, 235, 235), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def place_camera(panel, frame, width=560):
    if frame is None:
        line(panel, 'No camera: synthetic demo', 130, 174, .58)
        return
    h, w = frame.shape[:2]
    avail_h = 261
    factor = min((width - 20) / w, avail_h / h)
    resized = cv2.resize(frame, (max(1, round(w*factor)), max(1, round(h*factor))))
    ox = (width - resized.shape[1]) // 2
    oy = 34 + (avail_h - resized.shape[0]) // 2
    panel[oy:oy+resized.shape[0], ox:ox+resized.shape[1]] = resized


def render(frame, p, old, new, angles, thumb_mcp, yaw, pitch,
           errs, demo=False, paused=False, note='', thumb_mcp_sign=1):
    """Four panes in a 2x2 layout, legible on 1366px laptops."""
    pw, ph = 560, 344
    panels = [np.full((ph, pw, 3), (24, 29, 36), np.uint8) for _ in range(4)]
    line(panels[0], '1. RGB CAMERA', 14, 24, .68)
    line(panels[1], '2. MEDIAPIPE WORLD (ML ESTIMATE)', 14, 24, .58)
    line(panels[2], '3. ORIGINAL 15-FK (MCP=0)', 14, 24, .61)
    line(panels[3], '4. 15-FK + MEASURED THUMB MCP', 14, 24, .61)
    place_camera(panels[0], frame, pw)
    if p is not None:
        for pane, points in ((panels[1], p), (panels[2], old), (panels[3], new)):
            draw_skeleton(pane, project_points(points, p, yaw, pitch, pw))
        old_rms, old_tip, new_rms, new_tip, old_full, new_full = errs
        improvement = 100 * (old_rms - new_rms) / old_rms if old_rms > 1e-6 else 0.0
        line(panels[0], f'Thumb MCP: {thumb_mcp:.1f} deg | IP: {angles[1]:.1f} deg',
             12, 324, .51, (110, 225, 255))
        line(panels[1], f'Compare reference for thumb joints 2 / 3 / 4',
             12, 325, .45, (198, 202, 210))
        line(panels[2], f'Thumb RMS {old_rms:.1f} mm | TIP {old_tip:.1f} mm',
             12, 304, .52, (85, 197, 250))
        line(panels[2], f'All finger RMS {old_full:.1f} mm', 12, 328, .47)
        line(panels[3], f'Thumb RMS {new_rms:.1f} mm | TIP {new_tip:.1f} mm',
             12, 287, .52, (85, 197, 250))
        line(panels[3], f'All finger RMS {new_full:.1f} mm', 12, 306, .47)
        label = ('better' if improvement > .1 else 'worse' if improvement < -.1 else 'same')
        line(panels[3], f'Thumb change: {improvement:+.1f}% ({label})',
             12, 328, .48,
             (130, 230, 150) if improvement > 0 else (125, 160, 255))
    else:
        for idx in (1, 2, 3):
            line(panels[idx], 'Waiting for valid hand landmarks...', 70, 186, .60)
    if note:
        line(panels[0], note[:60], 12, 341, .43, (130, 180, 250))
    img = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    status = 'PAUSED' if paused else ('DEMO' if demo else 'LIVE')
    footer = np.full((32, 2*pw, 3), (39, 43, 52), dtype=np.uint8)
    line(footer, f'{status} | q quit | space pause | a/d yaw | w/s tilt | r reset',
         12, 13, .40, (220, 220, 220))
    line(footer, f'1 MCP elev | 2 PIP/IP | 3 thumb side | 4 thumb MCP sign: {thumb_mcp_sign:+d}',
         12, 28, .40, (220, 220, 220))
    return np.vstack([img, footer])


def demo_landmarks(t):
    """Synthetic hand with genuine nonzero thumb MCP bend for offline tests."""
    p = np.zeros((21, 3), dtype=np.float64)
    p[0] = [0, 0, 0]
    p[5], p[9], p[13], p[17] = [
        [-.034, .074, 0], [-.012, .087, 0],
        [.012, .085, 0], [.034, .069, 0],
    ]
    p[1] = [-.044, .024, 0]
    lateral, forward, normal = palm_frame(p)
    blend = (1 + math.sin(1.4*t)) * .5
    values = np.array([15., 15. + 65*blend, 55.,
                       12+40*blend, 12+65*blend, 75,
                       15+35*blend, 15+70*blend, 90,
                       18+25*blend, 15+65*blend, 105,
                       18+20*blend, 20+50*blend, 112])
    for mcp, pip, dip, tip in FINGERS:
        p[pip] = p[mcp] + .039 * forward
        p[dip] = p[pip] + .025 * forward
        p[tip] = p[dip] + .018 * forward
    p[2] = p[1] + .028 * forward
    p[3] = p[2] + .024 * forward
    p[4] = p[3] + .020 * forward
    # First generate a baseline hand to establish a proper synthetic reference.
    dummy = reconstruct_pair(p, values)[0]
    # Now insert the missing independent thumb MCP rotation into the reference.
    mcp_bend = 5. + 48*blend
    th_elev, th_yaw = float(values[0]), -float(values[2])
    dummy[3] = dummy[2] + .024 * direction(
        th_elev + mcp_bend, th_yaw, lateral, forward, normal)
    dummy[4] = dummy[3] + .020 * direction(
        th_elev + mcp_bend + float(values[1]), th_yaw,
        lateral, forward, normal)
    return dummy


def main():
    parser = argparse.ArgumentParser(description='SyncHand thumb MCP FK comparison')
    parser.add_argument('--model', type=Path,
                        default=Path.home() / 'synchand_ws/src/synchand/synchand/hand_landmarker.task')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--fps', type=float, default=12.0)
    parser.add_argument('--demo', action='store_true',
                        help='run without a webcam or MediaPipe')
    parser.add_argument('--window-scale', type=float, default=1.,
                        help='shrink window if screen is small, e.g. 0.8')
    args = parser.parse_args()
    if args.fps <= 0 or not (0.4 <= args.window_scale <= 1.5):
        parser.error('--fps must be positive; --window-scale between 0.4 and 1.5')

    cap = None
    detector = None
    mp = None
    if not args.demo:
        if not args.model.expanduser().is_file():
            parser.error(f'MediaPipe .task file not found: {args.model}')
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(args.model.expanduser())),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.7,
            )
        )
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            detector.close()
            raise RuntimeError(f'Camera {args.camera} could not be opened')

    # Start front-facing, as requested in the earlier validator discussion.
    yaw, pitch = 0., 80.
    mcp_sign, pip_sign, thumb_sign, thumb_mcp_sign = 1, 1, -1, 1
    paused = False
    last_good = None
    last_debug = 0.0
    note = ''
    print('SyncHand comparison | 15-FK vs 15-FK+Thumb MCP')
    print('q exit, SPACE pause, a/d, w/s rotate, r reset, 1/2/3/4 toggle signs')
    try:
        while True:
            started = time.monotonic()
            if not paused or last_good is None:
                frame = None
                p = None
                if args.demo:
                    p = demo_landmarks(started)
                else:
                    ok, frame = cap.read()
                    if ok:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = detector.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                        if result.hand_world_landmarks and result.hand_landmarks:
                            p = np.array([(v.x, v.y, v.z)
                                          for v in result.hand_world_landmarks[0]],
                                         dtype=np.float64)
                            h, w = frame.shape[:2]
                            xy = [(int(lm.x*w), int(lm.y*h))
                                  for lm in result.hand_landmarks[0]]
                            for a, b in CONNECTIONS:
                                cv2.line(frame, xy[a], xy[b], (50, 235, 105), 2,
                                         cv2.LINE_AA)
                            for x, y in xy:
                                cv2.circle(frame, (x, y), 3, (50, 215, 250), -1)
                if p is not None:
                    try:
                        angles = angle_features(p)
                        last_good = (frame, p, angles)
                        note = ''
                    except ValueError as exc:
                        last_good = None
                        note = str(exc)
                else:
                    last_good = None
            if last_good is not None:
                frame, p, angles = last_good
                old, new, thumb_mcp = reconstruct_pair(
                    p, angles, mcp_sign, pip_sign, thumb_sign, thumb_mcp_sign)
                er_old, tip_old, old_err = thumb_error(p, old)
                er_new, tip_new, new_err = thumb_error(p, new)
                metrics = (er_old, tip_old, er_new, tip_new,
                           all_finger_error(p, old), all_finger_error(p, new))
                if not paused and time.monotonic() - last_debug >= 1.0:
                    print("\n===== Thumb Joint Error (mm) =====")
                    print("          MCP     IP     TIP")
                    print(
                        "Original:",
                        np.round(old_err, 1)
                    )
                    print(
                        "Improved:",
                        np.round(new_err, 1)
                    )
                    print(
                        f"Thumb MCP Flex: {thumb_mcp:.1f}°"
                    )
                    lateral, forward, normal = palm_frame(p)

                    def thumb_yaw(q, a, b):
                        return signed_spread(
                            q[b] - q[a],
                            lateral, forward, normal
                        )

                    print("\n===== Thumb Direction (deg) =====")

                    print(
                        f"MCP -> IP | "
                        f"Real: {thumb_yaw(p, 2, 3):.1f} | "
                        f"Old: {thumb_yaw(old, 2, 3):.1f} | "
                        f"New: {thumb_yaw(new, 2, 3):.1f}"
                    )

                    print(
                        f"IP -> TIP | "
                        f"Real: {thumb_yaw(p, 3, 4):.1f} | "
                        f"Old: {thumb_yaw(old, 3, 4):.1f} | "
                        f"New: {thumb_yaw(new, 3, 4):.1f}"
                    )
                    last_debug = time.monotonic()
            else:
                frame = p = old = new = angles = None
                thumb_mcp = 0.
                metrics = (0.,)*6
            canvas = render(frame, p, old, new, angles, thumb_mcp,
                            yaw, pitch, metrics, args.demo, paused, note,
                            thumb_mcp_sign)
            if args.window_scale != 1.:
                canvas = cv2.resize(canvas, None,
                                    fx=args.window_scale, fy=args.window_scale,
                                    interpolation=cv2.INTER_AREA)
            cv2.imshow('SyncHand - Compare Thumb MCP FK', canvas)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
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
                yaw, pitch = 0., 80.
            elif key == ord('1'):
                mcp_sign *= -1
            elif key == ord('2'):
                pip_sign *= -1
            elif key == ord('3'):
                thumb_sign *= -1
            elif key == ord('4'):
                thumb_mcp_sign *= -1
            wait = 1./args.fps - (time.monotonic()-started)
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

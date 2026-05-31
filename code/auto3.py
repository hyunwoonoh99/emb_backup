#!/usr/bin/env python3
"""
Pure Pursuit Lane Following
===========================
ld0.py 의 점선 2차 다항식 피팅 결과를 이용해
Pure Pursuit 알고리즘으로 로버를 제어한다.

Pure Pursuit 원리
-----------------
  1. 차량 위치(이미지 하단 중앙)에서 LOOKAHEAD_PX 픽셀 앞에 있는
     점선 경로 위의 목표점(lookahead point)을 찾는다.
  2. 목표점까지의 방향각 alpha = arctan2(dx, dy) 를 계산한다.
     (dx: 횡방향 오차, dy: 전방 거리, 이미지 y좌표 반전 적용)
  3. 순수 Pure Pursuit 조향각:
       delta = arctan(2 * L * sin(alpha) / Ld)
     결과를 [-1, +1] 로 정규화 (±45도 = ±1)
  4. 조향 크기에 비례해 속도를 감속 (코너 안전성) 

lane_following.py 와의 차이
---------------------------
  기존: offset_px 를 매 프레임 누적(incremental)해 조향 — P 제어 근사
  이번: 다항식 위 lookahead point 로 one-shot 조향 계산 — 진짜 Pure Pursuit
        조향 상태 누적 없음 → 오버슈트·진동 감소 기대

키 조작
-------
  M         : 자율주행 ↔ 수동 조작 토글
  w/s       : 수동 모드 — 전진 / 후진
  a/d       : 수동 모드 — 좌 / 우 회전
  Space     : 수동 모드 — 즉시 정지
  q / ESC   : 종료

실행 방법
---------
<<<<<<< HEAD
  python3 auto3.py          # CSI 카메라 (GStreamer)
  python3 auto3.py 0        # USB 카메라 /dev/video0
=======
  python3 pure_pursuit_lane.py          # CSI 카메라 (GStreamer)
  python3 pure_pursuit_lane.py 0        # USB 카메라 /dev/video0
  python3 pure_pursuit_lane.py video.mp4  # 영상 파일 (제어 없이 검출만)
>>>>>>> 0c45af6 (last backup)
"""

import cv2
import sys
import os
import time
import numpy as np
from pynput import keyboard as kb

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..', 'rover'))
sys.path.insert(0, _DIR)

from base_ctrl import BaseController
from ld0 import (extract_yellow_mask, make_roi_mask,
                 classify_contours, fit_dash_polynomial, CSI_PIPELINE)
from sign_detection import TRTInference, preprocess, postprocess, CLASS_NAMES, _gst_pipeline


# ── Pure Pursuit 파라미터 ──────────────────────────────────
LOOKAHEAD_PX    = 80    # lookahead 거리 (픽셀). 클수록 부드럽고 느린 응답.
WHEELBASE_PX    = 80     # 가상 축거(픽셀). 클수록 조향 감도 증가.
BASE_SPEED      = 0.1   # 직선 속도
TURN_SPEED      = 0.50   # 최대 조향 시 속도 (steer=1.0 → 외측 0.50 / 내측 0.10)
MAX_STEER       = 1.0    # 조향 상한 (±)
MAX_SPEED       = 0.5    # 바퀴 속도 상한
LOST_TIMEOUT    = 1.5    # 점선 미검출 후 정지까지 유예 시간 (초)

# ── 수동 모드 파라미터 ─────────────────────────────────────
MANUAL_SPEED_STEP  = 0.05
MANUAL_STEER_STEP  = 0.15
MANUAL_SPEED_DECAY = 0.85
MANUAL_STEER_DECAY = 0.60

# ── 표지판 인식 파라미터 ───────────────────────────────────
ENGINE_PATH    = os.path.expanduser('~/emb_backup/best.engine')
SIGN_EVERY_N   = 5      # N프레임마다 추론 (FPS 보호)
STRAIGHT_THRESH = 0.25  # 이 조향각 이하일 때만 표지판 동작 진입 (회전 중 오작동 방지)
STOP_DURATION  = 3.0    # stop 표지판: 정지 유지 시간 (초)
SLOW_DURATION  = 4.0    # slow 표지판: 감속 유지 시간 (초)
SLOW_SPEED     = 0.2   # slow 구간 속도
CROSS_SPEED    = 0.15   # 교차로 직진 속도
CROSS_DURATION = 3.0    # 교차로 직진 시간 (초)
SIGN_COOLDOWN  = 8.0    # 동일 표지판 재감지 방지 쿨다운 (초)

# ── 시리얼 포트 ────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200


# ─────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────
def _clip(val: float, limit: float) -> float:
    return max(-limit, min(limit, val))


def compute_wheel_speeds(steering: float, speed: float):
    """
    (steering, speed) → (L, R) 바퀴 명령값.
    lane_following.py 와 동일한 차동 구동 공식.
    반환값은 하드웨어 극성(-) 적용 완료. 그대로 송신하면 됨.
    """
    steer = _clip(steering, MAX_STEER)
    spd   = _clip(speed, MAX_SPEED)
    base  = abs(spd)

    if steer >= 0:          # 우회전: 왼쪽 내측 감속
        L = base * (1.0 - 0.9 * steer)
        R = base
    else:                   # 좌회전: 오른쪽 내측 감속
        L = base
        R = base * (1.0 + 0.9 * steer)

    L = _clip(L, MAX_SPEED)
    R = _clip(R, MAX_SPEED)
    if spd < 0:
        L, R = -L, -R

    return -L, -R   # 전진 = 음수 관례


# ─────────────────────────────────────────────────────────
# Pure Pursuit 핵심 함수
# ─────────────────────────────────────────────────────────
def get_lookahead_point(coeffs, car_x: int, car_y: int,
                        Ld_px: int, img_h: int):
    """
    다항식 x = f(y) 위에서 차량 위치 (car_x, car_y)로부터
    Ld_px 픽셀 이상 떨어진 첫 번째 점을 반환.

    y 는 하단(car_y)에서 ROI 상단(img_h*0.30)으로 올라가며 샘플링.
    Ld_px 에 도달하지 못하면 가장 먼 가시 점 반환.
    """
    y_top   = int(img_h * 0.30)
    y_range = np.linspace(car_y, y_top, 400)
    x_range = np.polyval(coeffs, y_range)
    dists   = np.hypot(x_range - car_x, y_range - car_y)

    over = np.where(dists >= Ld_px)[0]
    if len(over) == 0:
        return int(x_range[-1]), int(y_range[-1])
    idx = over[0]
    return int(x_range[idx]), int(y_range[idx])


def pure_pursuit_steering(tx: int, ty: int,
                          car_x: int, car_y: int,
                          wheelbase_px: int) -> float:
    """
    Pure Pursuit 조향각 계산.

    이미지 좌표계 (y 아래 = 양수)를 주행 좌표계 (앞 = 양수)로 변환:
      dx = tx - car_x          (횡방향 오차, 양수 = 우측)
      dy = car_y - ty          (전방 거리,   양수 = 전방)

    alpha = arctan2(dx, dy)    (양수 = 우측 목표)
    delta = arctan(2·L·sin(α) / Ld)
    정규화: ±(π/4) rad = ±1
    """
    dx  = float(tx - car_x)
    dy  = float(car_y - ty)
    Ld  = np.hypot(dx, dy)
    if Ld < 1e-3:
        return 0.0

    alpha     = np.arctan2(dx, dy)
    delta_rad = np.arctan2(2.0 * wheelbase_px * np.sin(alpha), Ld)
    steering  = delta_rad / (np.pi / 4.0)   # ±45° → ±1
    return float(np.clip(steering, -MAX_STEER, MAX_STEER))


# ─────────────────────────────────────────────────────────
# 검출 + 시각화
# ─────────────────────────────────────────────────────────
def detect_lane(frame):
    """
    ld0 의 단계별 함수를 직접 호출.
    Returns: (coeffs, fit_points, dash_centroids, solid_cnts)
    """
    h, w = frame.shape[:2]
    yellow_mask = extract_yellow_mask(frame)
    roi_mask    = make_roi_mask(frame.shape)
    masked      = cv2.bitwise_and(yellow_mask, roi_mask)
    dash_cnts, solid_cnts, dash_centroids = classify_contours(masked, w)
    coeffs, fit_points = fit_dash_polynomial(dash_centroids, h)
    return coeffs, fit_points, dash_centroids, solid_cnts


def draw_overlay(frame, fit_points, dash_centroids, solid_cnts,
                 tx, ty, car_x, car_y, status, color):
    """Pure Pursuit 시각화 오버레이."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 실선·점선 표시
    cv2.drawContours(vis, solid_cnts, -1, (0, 0, 255), 1)
    for (cx2, cy2) in dash_centroids:
        cv2.circle(vis, (cx2, cy2), 4, (0, 255, 0), -1)
    if len(fit_points) >= 2:
        for i in range(len(fit_points) - 1):
            cv2.line(vis, fit_points[i], fit_points[i + 1], (255, 255, 0), 2)

    # Lookahead 원 (반경 = LOOKAHEAD_PX)
    cv2.circle(vis, (car_x, car_y), LOOKAHEAD_PX, (180, 180, 180), 1)

    # 차량 중심선
    cv2.line(vis, (car_x, h // 2), (car_x, car_y), (0, 165, 255), 2)

    # 목표점 & 조향 벡터
    if tx is not None:
        cv2.circle(vis, (tx, ty), 10, (0, 255, 255), -1)
        cv2.arrowedLine(vis, (car_x, car_y), (tx, ty),
                        (0, 255, 255), 2, tipLength=0.15)

    cv2.putText(vis, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


# ─────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────
def main(source=None):
    # ── 카메라 초기화 ─────────────────────────────────────
    if source is None:
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
<<<<<<< HEAD
    else:
=======
    elif isinstance(source, int):
>>>>>>> 0c45af6 (last backup)
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
<<<<<<< HEAD
=======
    else:
        cap = cv2.VideoCapture(source)
>>>>>>> 0c45af6 (last backup)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")

<<<<<<< HEAD
    sim_mode = False  # 로버 연결 실패 시 True로 전환
    base = None
    try:
        base = BaseController(SERIAL_PORT, BAUD_RATE)
        print(f"[PP] 로버 연결: {SERIAL_PORT}  1초 후 주행 시작...")
        time.sleep(1.0)
    except Exception as e:
        print(f"[PP] 경고: 로버 연결 실패 ({e}) — 시각화만 실행합니다.")
        sim_mode = True
=======
    sim_mode = isinstance(source, str)
    base = None
    if not sim_mode:
        try:
            base = BaseController(SERIAL_PORT, BAUD_RATE)
            print(f"[PP] 로버 연결: {SERIAL_PORT}  1초 후 주행 시작...")
            time.sleep(1.0)
        except Exception as e:
            print(f"[PP] 경고: 로버 연결 실패 ({e}) — 시각화만 실행합니다.")
            sim_mode = True
>>>>>>> 0c45af6 (last backup)

    # ── TRT 엔진 초기화 ───────────────────────────────────────
    try:
        engine = TRTInference(ENGINE_PATH)
        print(f"[PP] TRT 엔진 로드 완료: {ENGINE_PATH}")
    except Exception as e:
        print(f"[PP] 경고: TRT 엔진 로드 실패 ({e}) — 표지판 인식 비활성화")
        engine = None

    # ── cam1: 표지판 전용 카메라 ──────────────────────────────
    cap_sign = None
    if not sim_mode:
        try:
            cap_sign = cv2.VideoCapture(_gst_pipeline(1), cv2.CAP_GSTREAMER)
            if cap_sign.isOpened():
                print("[PP] cam1 (표지판) 열기 완료")
            else:
                print("[PP] 경고: cam1 열기 실패 — cam0 프레임으로 fallback")
                cap_sign = None
        except Exception as e:
            print(f"[PP] 경고: cam1 초기화 실패 ({e})")
            cap_sign = None

    last_detected = time.time()

    auto_mode       = False
    manual_speed    = 0.0
    manual_steering = 0.0
    running         = True

    # ── 표지판 상태 머신 ──────────────────────────────────────
    # 상태: 'run' | 'stop_wait' | 'light_wait' | 'crossing' | 'slow_drive'
    sign_state       = 'run'
    sign_state_start = 0.0
    sign_cooldown_ts = {}   # {'stop': t, 'traffic': t, 'slow': t}
    pending_sign     = None  # 회전 중 감지된 표지판 — 직선 복귀 후 처리
    frame_count      = 0
    last_sign_vis    = None  # cam1 최신 시각화 프레임 (Sign Cam 창 유지용)

    def stop_motors():
        if base is not None:
            base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})

    def send(steering, speed):
        if not sim_mode and base is not None:
            cmd_L, cmd_R = compute_wheel_speeds(steering, speed)
            base.base_json_ctrl({"T": 1, "L": cmd_L, "R": cmd_R})
            return cmd_L, cmd_R
        return 0.0, 0.0

    pressed = set()

    def on_press(key):
        nonlocal auto_mode, manual_speed, manual_steering, running
        try:
            pressed.add(key.char)
            if key.char == 'q':
                running = False
            elif key.char == 'm':
                auto_mode = not auto_mode
                manual_speed    = 0.0
                manual_steering = 0.0
                stop_motors()
                print(f"[PP] 모드 전환 → {'AUTO' if auto_mode else 'MANUAL'}")
        except AttributeError:
            if key == kb.Key.space:
                manual_speed    = 0.0
                manual_steering = 0.0
            elif key == kb.Key.esc:
                running = False

    def on_release(key):
        try:
            pressed.discard(key.char)
        except AttributeError:
            pass

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    cv2.namedWindow("Pure Pursuit Lane", cv2.WINDOW_NORMAL)
<<<<<<< HEAD
    cv2.namedWindow("Sign Cam", cv2.WINDOW_NORMAL)
=======
   
>>>>>>> 0c45af6 (last backup)

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            h, w      = frame.shape[:2]
            car_x     = w // 2
            car_y     = h - 1
            now       = time.time()

            # ── 점선 검출 ──────────────────────────────────
            coeffs, fit_points, dash_centroids, solid_cnts = detect_lane(frame)

            tx, ty        = None, None
            steering_auto = 0.0

            if coeffs is not None:
                last_detected = now
                tx, ty = get_lookahead_point(coeffs, car_x, car_y,
                                              LOOKAHEAD_PX, h)
                steering_auto = pure_pursuit_steering(tx, ty, car_x, car_y,
                                                       WHEELBASE_PX)

            # ── 표지판 추론: cam1 전용 (N프레임마다) ───────────
            frame_count += 1
            detected_label = None
            detected_box   = None
            detected_conf  = None
            if engine is not None and frame_count % SIGN_EVERY_N == 0:
                # cam1 전용: 읽기 실패 시 추론 건너뜀 (cam0 fallback 없음)
                sign_src = None
                if cap_sign is not None:
                    ret_s, sf = cap_sign.read()
                    if ret_s:
                        sign_src = sf
<<<<<<< HEAD
=======
                elif sim_mode:
                    sign_src = frame  # 영상 파일 재생 시에만 cam0 사용
>>>>>>> 0c45af6 (last backup)

                if sign_src is not None:
                    try:
                        inp = preprocess(sign_src)
                        out = engine.infer(inp)
                        cls, conf, box = postprocess(out, sign_src.shape[:2])
                        if cls is not None:
                            detected_label = CLASS_NAMES[cls]
                            detected_box   = box
                            detected_conf  = conf
                    except Exception:
                        pass

                    # cam1 시각화 업데이트
                    last_sign_vis = sign_src.copy()
                    if detected_box is not None and detected_label is not None:
                        x1, y1, x2, y2 = [int(v) for v in detected_box]
                        cv2.rectangle(last_sign_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(last_sign_vis, f"{detected_label} {detected_conf:.2f}",
                                    (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(last_sign_vis, "No detection", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

            # ── 표지판 상태 전환 ────────────────────────────
            def _cooldown_ok(key):
                return now - sign_cooldown_ts.get(key, 0.0) > SIGN_COOLDOWN

            # 감지된 표지판을 pending에 저장 (회전 중이어도 일단 기억)
            if detected_label:
                pending_sign = detected_label

            if auto_mode:
                if sign_state == 'run':
                    # 직선 주행 중일 때만 표지판 동작 진입
                    if abs(steering_auto) < STRAIGHT_THRESH and pending_sign:
                        if pending_sign == 'stop' and _cooldown_ok('stop'):
                            sign_state = 'stop_wait'
                            sign_state_start = now
                            sign_cooldown_ts['stop'] = now
                            pending_sign = None
                        elif pending_sign in ('red', 'green') and _cooldown_ok('traffic'):
                            sign_state = 'light_wait'
                            sign_state_start = now
                            sign_cooldown_ts['traffic'] = now
                            pending_sign = None
                        elif pending_sign == 'slow' and _cooldown_ok('slow'):
                            sign_state = 'slow_drive'
                            sign_state_start = now
                            sign_cooldown_ts['slow'] = now
                            pending_sign = None
                elif sign_state == 'stop_wait':
                    if now - sign_state_start >= STOP_DURATION:
                        sign_state = 'run'
                elif sign_state == 'light_wait':
                    if detected_label == 'green':
                        sign_state = 'crossing'
                        sign_state_start = now
<<<<<<< HEAD
                    elif now - sign_state_start >= 4.0:
=======
                    elif now - sign_state_start >= 8.0:
>>>>>>> 0c45af6 (last backup)
                        sign_state = 'run'
                elif sign_state == 'crossing':
                    if now - sign_state_start >= CROSS_DURATION:
                        sign_state = 'run'
                elif sign_state == 'slow_drive':
                    if now - sign_state_start >= SLOW_DURATION:
                        sign_state = 'run'

            # ── 수동 모드 ──────────────────────────────────
            if not auto_mode:
                sign_state = 'run'   # 수동 전환 시 표지판 상태 리셋

                if 'w' in pressed:
                    manual_speed = _clip(manual_speed + MANUAL_SPEED_STEP, MAX_SPEED)
                elif 's' in pressed:
                    manual_speed = _clip(manual_speed - MANUAL_SPEED_STEP, MAX_SPEED)
                else:
                    manual_speed *= MANUAL_SPEED_DECAY

                if 'a' in pressed:
                    manual_steering = _clip(manual_steering - MANUAL_STEER_STEP, MAX_STEER)
                elif 'd' in pressed:
                    manual_steering = _clip(manual_steering + MANUAL_STEER_STEP, MAX_STEER)
                else:
                    manual_steering *= MANUAL_STEER_DECAY

                cmd_L, cmd_R = send(manual_steering, manual_speed)
                sign_tag = f"  [{detected_label}]" if detected_label else ""
                status = (f"[MANUAL]  spd:{manual_speed:+.2f}  "
                          f"steer:{manual_steering:+.2f}  "
                          f"L:{cmd_L:+.2f} R:{cmd_R:+.2f}{sign_tag}")
                color = (0, 165, 255)

            # ── 자율주행 모드: Pure Pursuit + 표지판 우선 ──
            else:
                # 표지판 상태가 'run'이 아니면 Pure Pursuit을 override
                if sign_state == 'stop_wait':
                    stop_motors()
                    elapsed_s = now - sign_state_start
                    status = (f"[SIGN] STOP  {elapsed_s:.1f}/{STOP_DURATION:.0f}s  "
                              f"재개까지 {STOP_DURATION - elapsed_s:.1f}s")
                    color = (0, 0, 220)

                elif sign_state == 'light_wait':
                    stop_motors()
                    status = f"[SIGN] WAIT GREEN  ({detected_label or '-'})"
                    color  = (0, 0, 220)

                elif sign_state == 'crossing':
                    elapsed_s = now - sign_state_start
                    cmd_L, cmd_R = send(0.0, CROSS_SPEED)
                    status = (f"[SIGN] CROSSING  {elapsed_s:.1f}/{CROSS_DURATION:.0f}s  "
                              f"L:{cmd_L:+.2f} R:{cmd_R:+.2f}")
                    color = (0, 220, 220)

                elif sign_state == 'slow_drive':
                    elapsed_s = now - sign_state_start
                    cmd_L, cmd_R = send(steering_auto, SLOW_SPEED)
                    status = (f"[SIGN] SLOW  {elapsed_s:.1f}/{SLOW_DURATION:.0f}s  "
                              f"steer:{steering_auto:+.3f}  L:{cmd_L:+.2f} R:{cmd_R:+.2f}")
                    color = (0, 200, 100)

                # Pure Pursuit 정상 주행
                elif coeffs is not None:
                    speed = BASE_SPEED + (TURN_SPEED - BASE_SPEED) * abs(steering_auto)
                    cmd_L, cmd_R = send(steering_auto, speed)
                    sign_tag = f"  [{detected_label}]" if detected_label else ""
                    status = (f"[AUTO-PP]  steer:{steering_auto:+.3f}  "
                              f"spd:{speed:.2f}  "
                              f"L:{cmd_L:+.2f} R:{cmd_R:+.2f}{sign_tag}")
                    color = (0, 220, 0)

                else:
                    elapsed = now - last_detected
                    if elapsed > LOST_TIMEOUT:
                        stop_motors()
                        status = f"[AUTO-PP] LOST {elapsed:.1f}s — STOPPED"
                        color  = (0, 0, 255)
                    else:
                        cmd_L, cmd_R = send(0.0, BASE_SPEED * 0.5)
                        status = f"[AUTO-PP] LOST {elapsed:.1f}s — coasting"
                        color  = (0, 200, 200)

            # ── 시각화 ─────────────────────────────────────
            vis = draw_overlay(frame, fit_points, dash_centroids, solid_cnts,
                               tx, ty, car_x, car_y, status, color)
            mode_label = "AUTO-PP" if auto_mode else "MANUAL"
            mode_color = (0, 220, 0) if auto_mode else (0, 165, 255)
            cv2.putText(vis, mode_label, (w - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

            cv2.imshow("Pure Pursuit Lane", vis)

<<<<<<< HEAD
            # cam1 Sign Cam 창 업데이트
            if last_sign_vis is not None:
                cv2.imshow("Sign Cam", last_sign_vis)
=======
>>>>>>> 0c45af6 (last backup)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[PP] Ctrl-C 감지")
    finally:
        listener.stop()
        stop_motors()
        cap.release()
        if cap_sign is not None:
            cap_sign.release()
        cv2.destroyAllWindows()
        print("[PP] 종료 완료")


if __name__ == "__main__":
<<<<<<< HEAD
    src = int(sys.argv[1]) if len(sys.argv) > 1 else None
=======
    if len(sys.argv) > 1:
        src = sys.argv[1]
        src = int(src) if src.isdigit() else src
    else:
        src = None
>>>>>>> 0c45af6 (last backup)
    main(src)




<<<<<<< HEAD

=======
>>>>>>> 0c45af6 (last backup)

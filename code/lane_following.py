#!/usr/bin/env python3
"""
Lane Following — Dashed Center Line
=====================================
ld0.py 의 점선 검출 결과(offset_px)를 이용해
로버가 점선 중앙을 따라가도록 제어한다.

제어 원리
---------
  offset_px > 0  →  점선이 차량 오른쪽  →  우회전 (steering > 0)
  offset_px < 0  →  점선이 차량 왼쪽   →  좌회전 (steering < 0)
  steering = Kp * (offset_px / half_width)   # 비례 제어(P)

모터 명령 형식 (base_ctrl.py 기준)
------------------------------------
  {"T":1, "L":<left_speed>, "R":<right_speed>}
  · 값 범위는 대략 -1.0 ~ +1.0 (실험 결과 ±0.5 이상은 너무 빠름)
  · 하드웨어 관례: 값이 음수 = 전진 (keyboard 코드의 -L, -R 전송과 동일)

키 조작
-------
  M         : 자율주행 ↔ 수동 조작 토글
  w/s       : 수동 모드 — 전진 / 후진
  a/d       : 수동 모드 — 좌회전 / 우회전
  Space     : 수동 모드 — 즉시 정지 (속도·조향 0)
  q / ESC   : 종료

실행 방법
---------
  python3 lane_following.py          # CSI 카메라 (GStreamer)
  python3 lane_following.py 0        # USB 카메라 /dev/video0
  python3 lane_following.py video.mp4  # 영상 파일 (제어 없이 검출만)
"""

import cv2
import sys
import os
import time
import threading
from pynput import keyboard as kb

# rover 패키지 경로 추가
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..', 'rover'))
sys.path.insert(0, _DIR)

from base_ctrl import BaseController
from ld0 import process_frame, CSI_PIPELINE


# ─────────────────────────────────────────────
# 제어 파라미터  (주행 중 튜닝 포인트)
# ─────────────────────────────────────────────
BASE_SPEED      = 0.15   # 기본 전진 속도 (권장 0.15 ~ 0.30)
KP              = 0.55    # 비례 게인: 값이 클수록 더 민감하게 꺾음
MAX_STEER       = 1.0    # 최대 조향값 (±MAX_STEER)
MAX_SPEED       = 0.5   # 바퀴 속도 상한
LOST_TIMEOUT = 1.5     # 점선 미검출 후 정지까지 유예 시간 (초)

# 수동 모드 조작 파라미터
MANUAL_SPEED_STEP  = 0.05   # w/s 한 번 누를 때 속도 변화량
MANUAL_STEER_STEP  = 0.15   # a/d 한 번 누를 때 조향 변화량
MANUAL_SPEED_DECAY = 0.85   # 키 미입력 시 속도 감쇠 (프레임당)
MANUAL_STEER_DECAY = 0.60   # 키 미입력 시 조향 복원

# ─────────────────────────────────────────────
# 시리얼 포트
# ─────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def _clip(val: float, limit: float) -> float:
    return max(-limit, min(limit, val))


def compute_wheel_speeds(steering: float, speed: float):
    """
    (steering, speed) → (L, R) 바퀴 명령값.

    ctrl_with_keyboard.py 의 update_vehicle_motion() 과 동일한 공식.
    steering > 0 → 우회전 (right wheel 빠름),  speed > 0 → 전진.
    반환값은 이미 하드웨어 극성(-) 적용된 값이므로 그대로 송신하면 됨.
    """
    steer = _clip(steering, MAX_STEER)
    spd   = _clip(speed,    MAX_SPEED)

    base = abs(spd)

    # 내측 바퀴는 base 고정, 외측 바퀴만 증가
    # steer > 0: 우회전 → L(내측) 고정, R(외측) 증가
    # steer < 0: 좌회전 → R(내측) 고정, L(외측) 증가
    if steer >= 0:
        L = base * (1.0 - 0.8 * steer)
        R = base
    else:
        L = base
        R = base * (1.0 + 0.8 * steer)

    L = _clip(L, MAX_SPEED)
    R = _clip(R, MAX_SPEED)

    if spd < 0:
        L, R = -L, -R

    # 하드웨어 관례: 전진 = 음수
    return -L, -R


# ─────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────
def main(source=None):
    # ── 카메라 초기화 ──────────────────────────
    if source is None:
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    elif isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")

    # ── 로버 컨트롤러 초기화 ──────────────────
    sim_mode = isinstance(source, str)   # 파일 재생이면 모터 미사용
    base = None
    if not sim_mode:
        try:
            base = BaseController(SERIAL_PORT, BAUD_RATE)
            print(f"[LF] 로버 연결: {SERIAL_PORT}  2초 후 주행 시작...")
            time.sleep(2.0)
        except Exception as e:
            print(f"[LF] 경고: 로버 연결 실패 ({e}) — 시각화만 실행합니다.")
            sim_mode = True

    img_w = 640
    last_detected = time.time()

    # ── 상태 변수 (스레드 공유) ──────────────────
    auto_mode       = False
    manual_speed    = 0.0
    manual_steering = 0.0
    auto_steering   = 0.0
    running         = True   # False 되면 메인루프 종료

    def stop_motors():
        if base is not None:
            base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})

    def send(steering, speed):
        if not sim_mode and base is not None:
            cmd_L, cmd_R = compute_wheel_speeds(steering, speed)
            base.base_json_ctrl({"T": 1, "L": cmd_L, "R": cmd_R})
            return cmd_L, cmd_R
        return 0.0, 0.0

    # ── pynput: 누르고 있는 키 집합으로 추적 ─────
    pressed = set()

    def on_press(key):
        nonlocal auto_mode, manual_speed, manual_steering, auto_steering, running
        try:
            pressed.add(key.char)
            if key.char == 'q':
                running = False
            elif key.char == 'm':
                auto_mode = not auto_mode
                manual_speed    = 0.0
                manual_steering = 0.0
                auto_steering   = 0.0
                stop_motors()
                print(f"[LF] 모드 전환 → {'AUTO' if auto_mode else 'MANUAL'}")
            elif key.char == ' ':
                manual_speed    = 0.0
                manual_steering = 0.0
        except AttributeError:
            # 스페이스, ESC 등 특수키
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

    cv2.namedWindow("Lane Following", cv2.WINDOW_NORMAL)

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            # ── 점선 검출 ───────────────────────
            view, offset_px = process_frame(frame)
            now = time.time()

            # ── 수동 모드: 현재 누른 키로 속도·조향 갱신 ──
            if not auto_mode:
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
                status = (f"[MANUAL]  spd:{manual_speed:+.2f}  "
                          f"steer:{manual_steering:+.2f}  "
                          f"L:{cmd_L:+.2f} R:{cmd_R:+.2f}")
                color = (0, 165, 255)

            # ── 자율주행 제어 ───────────────────
            else:
                if offset_px is not None:
                    last_detected = now
                    if offset_px > 0:
                        auto_steering = _clip(auto_steering + MANUAL_STEER_STEP, MAX_STEER)
                    elif offset_px < 0:
                        auto_steering = _clip(auto_steering - MANUAL_STEER_STEP, MAX_STEER)
                    else:
                        auto_steering *= MANUAL_STEER_DECAY
                    cmd_L, cmd_R = send(auto_steering, BASE_SPEED)
                    status = (f"[AUTO]  offset:{offset_px:+4d}px  "
                              f"steer:{auto_steering:+.2f}  "
                              f"L:{cmd_L:+.2f} R:{cmd_R:+.2f}")
                    color = (0, 220, 0)
                else:
                    elapsed = now - last_detected
                    auto_steering *= MANUAL_STEER_DECAY
                    if elapsed > LOST_TIMEOUT:
                        stop_motors()
                        status = f"[AUTO] LOST {elapsed:.1f}s — STOPPED"
                        color  = (0, 0, 255)
                    else:
                        cmd_L, cmd_R = send(auto_steering, BASE_SPEED * 0.7)
                        status = f"[AUTO] LOST {elapsed:.1f}s — coasting"
                        color  = (0, 200, 200)

            # ── 상태 표시 ────────────────────────
            cv2.putText(view, status,
                        (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
            mode_label = "AUTO" if auto_mode else "MANUAL"
            mode_color = (0, 220, 0) if auto_mode else (0, 165, 255)
            cv2.putText(view, mode_label,
                        (view.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            cv2.imshow("Lane Following", view)
            cv2.waitKey(1)   # OpenCV 창 이벤트 처리용 (키 입력은 pynput으로)

    except KeyboardInterrupt:
        print("\n[LF] Ctrl-C 감지")
    finally:
        listener.stop()
        stop_motors()
        cap.release()
        cv2.destroyAllWindows()
        print("[LF] 종료 완료")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
        src = int(src) if src.isdigit() else src
    else:
        src = None   # CSI 카메라
    main(src)

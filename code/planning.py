#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planning & Control node — Wave Rover 자율주행 경쟁 FSM

구독:
  /lane/offset   Float32  [-1, 1]   lane_detection.py 출력
  /detection     String   클래스명  팀원 YOLO 노드 출력
                          ('left_turn' | 'right_turn' | 'stop' | 'pedestrian' |
                           'red' | 'green' | 'npc')

제어:
  Wave Rover serial → base_ctrl.BaseController (T:1 명령)
"""

import sys
import os
import time
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String

# base_ctrl 경로 등록
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rover'))
from base_ctrl import BaseController


# ─────────────────────────────────────────────────────────────────────────────
# 설정값 — 실제 트랙에서 조정
# ─────────────────────────────────────────────────────────────────────────────

SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUD = 115200

# 이미지 너비 기준 점선 선택 x 좌표 (640px 기준)
IMG_CENTER_X       = 320   # 직진 기준 (십자교차로 직진)
ROUNDABOUT_LEFT_X  = 220   # 좌회전 CCW 유지 (중앙 왼쪽)
ROUNDABOUT_RIGHT_X = 420   # 우회전 첫 출구 (중앙 오른쪽)

# ── PID 조향 ──────────────────────────────────────────────────────────────────
KP = 0.40   # 비례 게인
KD = 0.08   # 미분 게인 (오버슈트 억제)

# ── 속도 ──────────────────────────────────────────────────────────────────────
BASE_SPEED       = 0.25   # 직선 일반 주행 [-0.5, 0.5]
SLOW_SPEED       = 0.13   # 서행 (회전교차로 / 보행자 구간)
ROUNDABOUT_SPEED = 0.13   # 회전교차로 호 주행

# ── 감지 확인 ─────────────────────────────────────────────────────────────────
CONFIRM_COUNT   = 5     # 같은 클래스 연속 N회 후 확정
CONFIRM_COOLDOWN = 5.0  # 상태 전환 후 재감지 방지 [s]

# ── 회전교차로 타이밍 [s] — 트랙 테스트 후 조정 ──────────────────────────────
#
# 회전교차로는 반드시 반시계방향(CCW)으로 주행 (도로교통법)
#
# [좌회전] 남쪽 진입 → CCW 약 270° 주행 → 서쪽(왼쪽) 출구
#   긴 호를 따라 돌아야 하므로 traverse 시간이 매우 길어야 함
#
# [우회전] 남쪽 진입 → CCW 약 90° 주행 → 동쪽(오른쪽) 출구
#   첫 번째 출구에서 바로 빠져나가므로 짧음
#
APPROACH_TIME      = 1.5    # 진입 서행 시간 (표지판 감지~원형 진입부)
LEFT_TRAVERSE_SEC  = 8.0    # 좌회전: ~270° CCW 호 주행 시간  ← 실측 후 조정
RIGHT_TRAVERSE_SEC = 2.5    # 우회전: ~90°  CCW 호 주행 시간  ← 실측 후 조정
EXIT_SETTLE_SEC    = 1.5    # 출구 후 직선 차선 재정착 시간

# ── 회전교차로 조향 편향 (PID 오프셋에 가산) ────────────────────────────────
# 좌회전: 우측 출구를 지나쳐 계속 CCW로 돌아야 함 → 미세 왼쪽 편향
# 우회전: 첫 우측 출구에서 빨리 빠져나가야 함 → 미세 오른쪽 편향
LEFT_BIAS  = -0.06   # 음수 = 왼쪽 편향 (우측 출구 무시하고 계속 CCW)
RIGHT_BIAS =  0.06   # 양수 = 오른쪽 편향 (첫 출구로 빠르게 이탈)

# ── NPC 회피 ─────────────────────────────────────────────────────────────────
NPC_STOP_SEC = 2.0   # NPC 감지 시 정지 대기 시간

# ── 표지판 대응 ───────────────────────────────────────────────────────────────
STOP_WAIT_SEC        = 2.5   # STOP 표지판: 정지 유지 시간
PEDESTRIAN_SLOW_SEC  = 4.0   # 보행자 표지판: 서행 구간 유지 시간


# ─────────────────────────────────────────────────────────────────────────────
# FSM 상태 상수
# ─────────────────────────────────────────────────────────────────────────────
class S:
    LANE_FOLLOW       = 'LANE_FOLLOW'
    ROUNDABOUT_ENTER  = 'ROUNDABOUT_ENTER'   # 진입 서행
    ROUNDABOUT_LEFT   = 'ROUNDABOUT_LEFT'    # 좌회전 호 주행
    ROUNDABOUT_RIGHT  = 'ROUNDABOUT_RIGHT'   # 우회전 호 주행 (NPC 감시)
    ROUNDABOUT_EXIT   = 'ROUNDABOUT_EXIT'    # 출구 재정착
    STOP_SIGN         = 'STOP_SIGN'          # 일시 정지
    PEDESTRIAN_SIGN   = 'PEDESTRIAN_SIGN'    # 서행
    TRAFFIC_RED       = 'TRAFFIC_RED'        # 적신호 대기


# ─────────────────────────────────────────────────────────────────────────────
# 감지 버퍼 — 같은 클래스 연속 N회 확인 후 확정
# ─────────────────────────────────────────────────────────────────────────────
class DetectionBuffer:
    def __init__(self, required: int = CONFIRM_COUNT, cooldown: float = CONFIRM_COOLDOWN):
        self._buf          = deque(maxlen=required)
        self._required     = required
        self._cooldown_end = 0.0

    def update(self, label: str) -> None:
        self._buf.append(label)

    def get_confirmed(self) -> Optional[str]:
        if time.monotonic() < self._cooldown_end:
            return None
        if len(self._buf) < self._required:
            return None
        if len(set(self._buf)) == 1 and self._buf[0]:
            return self._buf[0]
        return None

    def reset_cooldown(self, seconds: float = CONFIRM_COOLDOWN) -> None:
        self._buf.clear()
        self._cooldown_end = time.monotonic() + seconds


# ─────────────────────────────────────────────────────────────────────────────
# Planning Node
# ─────────────────────────────────────────────────────────────────────────────
class PlanningNode(Node):

    def __init__(self):
        super().__init__('planning_node')

        # ── 시리얼 모터 컨트롤러 ──────────────────────────────────────────────
        try:
            self.rover = BaseController(SERIAL_PORT, SERIAL_BAUD)
            self.get_logger().info(f"Serial 연결: {SERIAL_PORT}")
        except Exception as e:
            self.get_logger().error(f"Serial 연결 실패: {e}")
            self.rover = None

        # ── FSM 상태 ──────────────────────────────────────────────────────────
        self.state           = S.LANE_FOLLOW
        self._state_enter_t  = time.monotonic()
        self._roundabout_dir = None   # 'left' | 'right'

        # NPC 대기 관련
        self._npc_waiting     = False
        self._npc_wait_end    = 0.0
        self._extra_npc_time  = 0.0   # NPC 정지로 소비된 시간 보상

        # ── 센서 상태 ─────────────────────────────────────────────────────────
        self._offset      = 0.0
        self._prev_offset = 0.0
        self._prev_t      = time.monotonic()
        self._det_buf     = DetectionBuffer()
        self._npc_seen    = False

        # ── 구독 ──────────────────────────────────────────────────────────────
        self.create_subscription(Float32, '/lane/offset', self._offset_cb,    1)
        self.create_subscription(String,  '/detection',   self._detection_cb, 1)

        # 점선 선택 힌트 퍼블리셔 (-1=현재위치기준, 0~640=지정픽셀)
        self._select_x_pub = self.create_publisher(Int32, '/lane/select_x', 1)

        # ── 제어 루프 30 Hz ───────────────────────────────────────────────────
        self.create_timer(0.033, self._loop)

        self.get_logger().info("Planning node 시작")

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _offset_cb(self, msg: Float32) -> None:
        self._prev_offset = self._offset
        self._offset      = float(msg.data)

    def _detection_cb(self, msg: String) -> None:
        label = msg.data.strip().lower()
        self._det_buf.update(label)
        if label == 'npc':
            self._npc_seen = True

    # ─── 모터 출력 ────────────────────────────────────────────────────────────

    def _drive(self, L: float, R: float) -> None:
        if self.rover is None:
            return
        L = max(-0.5, min(0.5, L))
        R = max(-0.5, min(0.5, R))
        self.rover.base_speed_ctrl(L, R)

    def _stop(self) -> None:
        self._drive(0.0, 0.0)

    def _pid_drive(self, speed: float, bias: float = 0.0) -> None:
        """차선 오프셋 기반 PID 조향.
        offset > 0: 차선 중심이 오른쪽 → 우회전 필요 → L 증가, R 감소
        """
        now = time.monotonic()
        dt  = max(now - self._prev_t, 1e-3)
        self._prev_t = now

        err   = self._offset + bias
        d_err = (self._offset - self._prev_offset) / dt

        turn = KP * err + KD * d_err
        self._drive(speed + turn, speed - turn)

    def _publish_select_x(self) -> None:
        """FSM 상태에 따라 lane_detection에 점선 선택 기준 x를 전달."""
        if self.state == S.ROUNDABOUT_LEFT:
            x = ROUNDABOUT_LEFT_X
        elif self.state == S.ROUNDABOUT_RIGHT:
            x = ROUNDABOUT_RIGHT_X
        elif self.state in (S.LANE_FOLLOW, S.ROUNDABOUT_ENTER, S.ROUNDABOUT_EXIT,
                            S.STOP_SIGN, S.PEDESTRIAN_SIGN, S.TRAFFIC_RED):
            x = IMG_CENTER_X   # 직진 기준: 이미지 중앙 점선 선택
        else:
            x = -1             # 기본값: lane_detection이 현재 위치 기준으로 선택
        msg = Int32()
        msg.data = x
        self._select_x_pub.publish(msg)

    # ─── FSM 전환 ─────────────────────────────────────────────────────────────

    def _transition(self, new_state: str) -> None:
        self.get_logger().info(f"[FSM] {self.state} → {new_state}")

        # OLED 표시
        if self.rover:
            try:
                self.rover.base_oled(0, self.state[:16])
                self.rover.base_oled(1, f"-> {new_state[:13]}")
            except Exception:
                pass

        self.state          = new_state
        self._state_enter_t = time.monotonic()

        # NPC 대기 변수 초기화
        self._npc_waiting    = False
        self._npc_seen       = False
        self._extra_npc_time = 0.0

    # ─── FSM 메인 루프 ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        confirmed = self._det_buf.get_confirmed()
        elapsed   = time.monotonic() - self._state_enter_t

        # ── LANE_FOLLOW ────────────────────────────────────────────────────
        if self.state == S.LANE_FOLLOW:
            if confirmed in ('left_turn', 'right_turn'):
                self._roundabout_dir = 'left' if confirmed == 'left_turn' else 'right'
                self._transition(S.ROUNDABOUT_ENTER)
            elif confirmed == 'stop':
                self._transition(S.STOP_SIGN)
            elif confirmed == 'pedestrian':
                self._transition(S.PEDESTRIAN_SIGN)
            elif confirmed == 'red':
                self._transition(S.TRAFFIC_RED)
            else:
                self._pid_drive(BASE_SPEED)

        # ── ROUNDABOUT_ENTER ───────────────────────────────────────────────
        elif self.state == S.ROUNDABOUT_ENTER:
            self._pid_drive(SLOW_SPEED)
            if elapsed >= APPROACH_TIME:
                if self._roundabout_dir == 'left':
                    self._transition(S.ROUNDABOUT_LEFT)
                else:
                    self._transition(S.ROUNDABOUT_RIGHT)

        # ── ROUNDABOUT_LEFT ────────────────────────────────────────────────
        # CCW 약 270° 주행 후 서쪽(왼쪽) 출구 탈출
        # LEFT_BIAS: 우측 출구를 지나칠 수 있도록 미세 왼쪽 편향 유지
        elif self.state == S.ROUNDABOUT_LEFT:
            self._pid_drive(ROUNDABOUT_SPEED, bias=LEFT_BIAS)
            if elapsed >= LEFT_TRAVERSE_SEC:
                self._transition(S.ROUNDABOUT_EXIT)

        # ── ROUNDABOUT_RIGHT (NPC 감시 포함) ──────────────────────────────
        # CCW 약 90° 주행 후 동쪽(오른쪽) 첫 출구 탈출
        # RIGHT_BIAS: 첫 출구에서 빠르게 이탈하도록 미세 오른쪽 편향
        elif self.state == S.ROUNDABOUT_RIGHT:
            now = time.monotonic()

            if self._npc_seen and not self._npc_waiting:
                # NPC 처음 감지 → 정지 대기 시작
                self._npc_waiting    = True
                self._npc_wait_end   = now + NPC_STOP_SEC
                self._extra_npc_time += NPC_STOP_SEC
                self._npc_seen       = False

            if self._npc_waiting:
                self._stop()
                if now >= self._npc_wait_end:
                    self._npc_waiting = False
            else:
                self._pid_drive(ROUNDABOUT_SPEED, bias=RIGHT_BIAS)

            # 실제 주행 시간(NPC 정지 제외)이 기준 초과 시 탈출
            driving_elapsed = elapsed - self._extra_npc_time
            if driving_elapsed >= RIGHT_TRAVERSE_SEC:
                self._transition(S.ROUNDABOUT_EXIT)

        # ── ROUNDABOUT_EXIT ────────────────────────────────────────────────
        elif self.state == S.ROUNDABOUT_EXIT:
            self._pid_drive(BASE_SPEED)
            if elapsed >= EXIT_SETTLE_SEC:
                self._det_buf.reset_cooldown()
                self._transition(S.LANE_FOLLOW)

        # ── STOP_SIGN ──────────────────────────────────────────────────────
        elif self.state == S.STOP_SIGN:
            self._stop()
            if elapsed >= STOP_WAIT_SEC:
                self._det_buf.reset_cooldown()
                self._transition(S.LANE_FOLLOW)

        # ── PEDESTRIAN_SIGN ────────────────────────────────────────────────
        elif self.state == S.PEDESTRIAN_SIGN:
            self._pid_drive(SLOW_SPEED)
            if elapsed >= PEDESTRIAN_SLOW_SEC:
                self._det_buf.reset_cooldown()
                self._transition(S.LANE_FOLLOW)

        # ── TRAFFIC_RED ────────────────────────────────────────────────────
        elif self.state == S.TRAFFIC_RED:
            self._stop()
            if confirmed == 'green':
                self._det_buf.reset_cooldown()
                self._transition(S.LANE_FOLLOW)

        # 매 루프마다 점선 선택 힌트 업데이트
        self._publish_select_x()

    # ─── 종료 ────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

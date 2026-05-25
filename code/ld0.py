#!/usr/bin/env python3
"""
Lane Detection for Dashed Center Line Following
================================================
트랙 위 양쪽 노란 실선 사이의 노란 점선을 검출하여
점선의 중심 경로를 따라가도록 offset을 계산하는 코드.

핵심 아이디어:
  1. HSV 색공간에서 노란색만 마스킹
  2. ROI(관심 영역)로 도로 영역만 남기기 (책상/의자 등 배경 노이즈 제거)
  3. Contour 분석으로 "실선(긴 contour)" vs "점선(짧고 여러 개)" 구분
  4. 점선 조각들의 중심점(centroid)을 모아 2차 다항식으로 fitting
  5. 차량 중심(이미지 가로 중앙)과 점선 기준선의 offset 계산
"""

import cv2
import numpy as np


# ===================== 파라미터 =====================
# 노란색 HSV 범위 (실내 조명에 따라 조정 필요)
# 실내 형광등 아래 트랙의 노란색은 채도가 낮고 H가 연두 쪽으로 치우침
YELLOW_HSV_LOWER = np.array([20, 25, 140], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([75, 200, 255], dtype=np.uint8)

# Contour 면적 필터
DASH_AREA_MIN = 50        # 너무 작은 잡음 제거
DASH_AREA_MAX = 7000      # 단일 점선 조각 최대 면적 (가까울수록 크게 보이므로 여유 있게)

# 점선 모양 필터: 종횡비 상한
# 5.0 → 7.0으로 완화: 커브에서 기울어진 점선(aspect ~3-5)도 허용
# 실선 파편은 훨씬 길쭉(aspect 8~15+)해서 이 기준으로 걸러짐
DASH_MAX_ASPECT = 7.0

# 점선 클러스터 판정 기준
DASH_MIN_PIECES = 2       # 이 이상 모여야 "점선 패턴"으로 인정
DASH_WINDOW     = 0.12    # cx 클러스터링 윈도우 (이미지 너비 비율)

# 곡선 피팅 차수 (2차면 충분)
POLY_ORDER = 2

# 중앙 허용 구간: 카메라가 차체 중심에서 벗어나 있는 경우 보정
# 이 범위(픽셀) 안에 점선이 있으면 offset=0 으로 처리 (직진 유지)
OFFSET_DEADZONE_PX = 30

# 화면 표시용 색
COLOR_DASH = (0, 255, 0)      # 검출된 점선 조각 (초록)
COLOR_SOLID = (0, 0, 255)     # 실선 (빨강)
COLOR_FIT = (255, 255, 0)     # 피팅된 점선 경로 (시안)
COLOR_CENTER = (0, 165, 255)  # 차량 중심선 (주황)


# ===================== 유틸 함수 =====================
def make_roi_mask(shape):
    """
    ROI 마스크 생성. 도로가 보이는 사다리꼴 영역만 남김.
    이미지 하단을 넓게, 상단(멀리 있는 책상/의자 영역)은 좁게.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # 사다리꼴 꼭지점 (좌하 → 좌상 → 우상 → 우하)
    # 도로가 화면 거의 전체를 차지하므로 ROI를 충분히 크게
    polygon = np.array([[
        (int(w * 0.00), h),
        (int(w * 0.10), int(h * 0.30)),
        (int(w * 0.95), int(h * 0.30)),
        (int(w * 1.00), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return mask


def extract_yellow_mask(bgr):
    """HSV 변환 후 노란색만 추출."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    # 작은 노이즈 제거 + 점선 조각이 너무 잘게 쪼개지지 않게 약간 closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def classify_contours(mask, img_w):
    """
    Contour를 분석하여 점선 조각과 실선으로 분류.

    전략:
      1) 면적 하한(DASH_AREA_MIN)으로 잡음 제거
      2) 면적 상한(DASH_AREA_MAX)으로 1차 분리
         → 실선 파편은 개별 면적이 크므로 여기서 걸러짐
      3) 남은 "소형 contour" 중 cx 슬라이딩 윈도우로 가장 많이 모인 클러스터 선택
      4) 해당 클러스터가 DASH_MIN_PIECES 이상이면 점선 → waypoint 후보
         (y-gap 체크 없음: 커브에서 점선이 가로로 퍼지면 y-gap이 줄어들어 실패함)

    Returns:
        dash_contours, solid_contours, dash_centroids
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_items = []   # 면적 통과한 점선 후보 (cnt, cx, cy, area)
    solid_cnts  = []   # 면적이 커서 확실히 실선

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < DASH_AREA_MIN:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 면적 상한 + 종횡비로 1차 분류
        # 실선 파편은 면적이 크거나 매우 길쭉함 → 둘 중 하나라도 걸리면 실선
        if area > DASH_AREA_MAX:
            solid_cnts.append(cnt)
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        aspect = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 999
        if aspect > DASH_MAX_ASPECT:
            solid_cnts.append(cnt)
        else:
            small_items.append((cnt, cx, cy, area))

    if not small_items:
        return [], solid_cnts, []

    # 소형·compact contour 중 가장 많이 모인 cx 클러스터 탐색
    # count가 같으면 이미지 중앙에 가까운 클러스터 우선 (center bias)
    cxs = np.array([it[1] for it in small_items])
    window_half = int(img_w * DASH_WINDOW)
    center_x = img_w // 2

    best_center, best_count, best_dist = center_x, 0, img_w
    for c in cxs:
        count = int(np.sum(np.abs(cxs - c) <= window_half))
        dist  = abs(int(c) - center_x)
        if count > best_count or (count == best_count and dist < best_dist):
            best_count  = count
            best_center = int(c)
            best_dist   = dist

    # 클러스터 조각 수가 충분하면 점선, 아니면 실선
    dash_contours, dash_centroids = [], []
    for cnt, cx, cy, area in small_items:
        if abs(cx - best_center) <= window_half and best_count >= DASH_MIN_PIECES:
            dash_contours.append(cnt)
            dash_centroids.append((cx, cy))
        else:
            solid_cnts.append(cnt)

    return dash_contours, solid_cnts, dash_centroids


def fit_dash_polynomial(centroids, img_h):
    """
    점선 조각들의 중심점에 2차 다항식 피팅.
    y를 독립변수로 두는 게 좋음 (도로는 세로로 길쭉하므로).
    Returns:
        poly_coeffs: np.polyfit 계수 (x = f(y))
        fit_points: 시각화용 (x, y) 포인트 리스트
    """
    if len(centroids) < 3:
        return None, []

    pts = np.array(centroids, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]

    # x = a*y^2 + b*y + c
    coeffs = np.polyfit(ys, xs, POLY_ORDER)

    # 시각화용: 점선이 분포한 y 범위에서 곡선 그리기
    y_min, y_max = int(ys.min()), int(ys.max())
    # 화면 하단까지 외삽 (차량 앞쪽까지 곡선 연장)
    y_max = min(img_h - 1, y_max + 30)
    y_range = np.linspace(y_min, y_max, 50)
    x_range = np.polyval(coeffs, y_range)
    fit_points = list(zip(x_range.astype(int), y_range.astype(int)))

    return coeffs, fit_points


def compute_offset(coeffs, img_w, img_h, lookahead_ratio=0.85):
    """
    차량 기준선(이미지 가로 중앙)과 점선 경로의 offset 계산.
    lookahead_ratio: 화면 하단에서 얼마나 떨어진 지점을 기준으로 할지
                     (1.0 = 가장 가까운 곳, 0.5 = 화면 중간)
    Returns:
        offset_px: 양수면 점선이 차량 오른쪽 / 음수면 왼쪽
        target_x: 점선상의 목표 x 좌표
        target_y: 목표 y 좌표
    """
    if coeffs is None:
        return None, None, None

    target_y = int(img_h * lookahead_ratio)
    target_x = int(np.polyval(coeffs, target_y))
    car_center_x = img_w // 2
    offset_px = target_x - car_center_x
    if abs(offset_px) <= OFFSET_DEADZONE_PX:
        offset_px = 0
    return offset_px, target_x, target_y


# ===================== 메인 처리 함수 =====================
def process_frame(frame):
    """
    한 프레임에 대해 lane detection 수행.
    Returns:
        result_img: 좌측 원본 + 검출 결과 오버레이
        debug_img: 우측 디버그 뷰 (마스크 + 곡선)
        offset_px: 점선 기준 offset (픽셀)
    """
    h, w = frame.shape[:2]

    # 1) 노란색 마스크
    yellow_mask = extract_yellow_mask(frame)

    # 2) ROI 적용
    roi_mask = make_roi_mask(frame.shape)
    masked = cv2.bitwise_and(yellow_mask, roi_mask)

    # 3) Contour 분류
    dash_cnts, solid_cnts, dash_centroids = classify_contours(masked, w)

    # 4) 점선에 곡선 피팅
    coeffs, fit_points = fit_dash_polynomial(dash_centroids, h)

    # 5) Offset 계산
    offset_px, target_x, target_y = compute_offset(coeffs, w, h)

    # ---------- 시각화 (왼쪽: 원본 위에 오버레이) ----------
    result = frame.copy()

    # 실선 표시
    cv2.drawContours(result, solid_cnts, -1, COLOR_SOLID, 2)
    # 점선 조각 표시
    cv2.drawContours(result, dash_cnts, -1, COLOR_DASH, 2)
    # 점선 중심점
    for (cx, cy) in dash_centroids:
        cv2.circle(result, (cx, cy), 4, COLOR_DASH, -1)
    # 피팅된 곡선
    if len(fit_points) >= 2:
        for i in range(len(fit_points) - 1):
            cv2.line(result, fit_points[i], fit_points[i + 1], COLOR_FIT, 3)
    # 중앙 허용 구간 (초록 띠) — 이 안에 점선이 있으면 offset=0
    band_l = w // 2 - OFFSET_DEADZONE_PX
    band_r = w // 2 + OFFSET_DEADZONE_PX
    cv2.rectangle(result, (band_l, h // 2), (band_r, h), COLOR_DASH, 1)
    # 차량 중심선 (세로)
    cv2.line(result, (w // 2, h // 2), (w // 2, h), COLOR_CENTER, 2)
    # 목표점 표시
    if target_x is not None:
        cv2.circle(result, (target_x, target_y), 10, (0, 255, 255), -1)
        cv2.line(result, (w // 2, target_y), (target_x, target_y), (0, 255, 255), 2)

    # Offset 텍스트
    offset_text = f"offset: {offset_px:+d} px" if offset_px is not None else "offset: N/A"
    cv2.putText(result, offset_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ---------- 디버그 뷰 (오른쪽: 마스크 + 피팅 결과) ----------
    debug = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    if len(fit_points) >= 2:
        for i in range(len(fit_points) - 1):
            cv2.line(debug, fit_points[i], fit_points[i + 1], COLOR_FIT, 3)
    for (cx, cy) in dash_centroids:
        cv2.circle(debug, (cx, cy), 4, COLOR_DASH, -1)
    cv2.putText(debug, f"dashes: {len(dash_centroids)}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug, f"solids: {len(solid_cnts)}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 좌우로 합치기
    combined = np.hstack([result, debug])
    return combined, offset_px


# ===================== 실행부 =====================
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=360, format=I420 ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink max-buffers=1 drop=True"
)


def main(source=None):
    """
    source: None → CSI 카메라(GStreamer), int → V4L2 인덱스, str → 비디오 파일 경로
    """
    if source is None:
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    elif isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"카메라/비디오를 열 수 없습니다: {source}")

    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        view, offset_px = process_frame(frame)
        cv2.imshow("Lane Detection", view)

        # 여기서 offset_px를 제어부(PID 등)로 publish하면 됨
        # 예: steering = Kp * offset_px

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        src = sys.argv[1]
        src = int(src) if src.isdigit() else src
    else:
        src = None  # CSI 카메라 (GStreamer)
    main(src)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sign_detection.py — TensorRT YOLO11n 표지판 인식 ROS2 노드 / 단독 카메라 테스트

ROS2 노드: rclpy.spin()으로 실행, best.engine + CSI cam1 → /detection 발행
단독 테스트: python3 sign_detection.py  (ROS2 불필요)

클래스 매핑:
  green → green     left  → left_turn
  red   → red       right → right_turn
  slow  → pedestrian  stop → stop
"""

import os
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  CUDA context 초기화

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'best.engine')
SENSOR_ID   = 1       # cam1 사용 (cam0 은 ld0 전용)
CONF_THRESH = 0.50
NMS_THRESH  = 0.45
LOOP_HZ     = 20

CLASS_NAMES = ['green', 'left', 'red', 'right', 'slow', 'stop']
CLASS_MAP   = {
    'green': 'green',
    'left':  'left',
    'red':   'red',
    'right': 'right',
    'slow':  'slow',
    'stop':  'stop',
}

def _gst_pipeline(sensor_id: int, w=640, h=360) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width={w}, height={h}, format=I420 ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=True"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT 추론 래퍼
# ─────────────────────────────────────────────────────────────────────────────
class TRTInference:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        self._host   = {}
        self._device = {}
        self._inputs  = []
        self._outputs = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            h = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            d = cuda.mem_alloc(h.nbytes)
            self._host[name]   = h
            self._device[name] = d
            self.context.set_tensor_address(name, int(d))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(name)
            else:
                self._outputs.append(name)

    def infer(self, img: np.ndarray) -> np.ndarray:
        """img: float32 NCHW [1,3,640,640] 0~1 정규화"""
        name_in = self._inputs[0]
        np.copyto(self._host[name_in], img.ravel())
        cuda.memcpy_htod_async(self._device[name_in], self._host[name_in], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        name_out = self._outputs[0]
        cuda.memcpy_dtoh_async(self._host[name_out], self._device[name_out], self.stream)
        self.stream.synchronize()
        shape = tuple(self.engine.get_tensor_shape(name_out))
        return self._host[name_out].reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 / 후처리
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])  # NCHW


def postprocess(output: np.ndarray, frame_hw: tuple = None):
    """
    output: [1, 10, 8400]  (4 box + 6 class scores)
    frame_hw: (h, w) 원본 프레임 크기 — 박스를 원본 좌표로 역변환할 때 사용
    반환: (class_id, confidence, box_xyxy) or (None, None, None)
    """
    preds     = output[0].T                              # [8400, 10]
    scores    = preds[:, 4:]                             # [8400, 6]
    class_ids = np.argmax(scores, axis=1)
    confs     = scores[np.arange(len(scores)), class_ids]

    mask = confs > CONF_THRESH
    if not mask.any():
        return None, None, None

    boxes_raw = preds[mask, :4]   # cx, cy, w, h (640px 기준)
    class_ids = class_ids[mask]
    confs     = confs[mask]

    x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    bboxes_xywh = np.stack([x1, y1, boxes_raw[:, 2], boxes_raw[:, 3]], axis=1)

    indices = cv2.dnn.NMSBoxes(
        bboxes_xywh.tolist(), confs.tolist(), CONF_THRESH, NMS_THRESH
    )
    if len(indices) == 0:
        return None, None, None

    flat = np.array(indices).flatten()
    best = flat[np.argmax(confs[flat])]

    box = bboxes_xywh[best]  # x, y, w, h in 640-space
    x1b, y1b, x2b, y2b = box[0], box[1], box[0] + box[2], box[1] + box[3]

    if frame_hw is not None:
        fh, fw = frame_hw
        sx, sy = fw / 640.0, fh / 640.0
        x1b, x2b = x1b * sx, x2b * sx
        y1b, y2b = y1b * sy, y2b * sy

    return int(class_ids[best]), float(confs[best]), (int(x1b), int(y1b), int(x2b), int(y2b))


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 노드
# ─────────────────────────────────────────────────────────────────────────────
class SignDetectionNode:
    """rclpy 없이도 import 가능하도록 조건부 정의."""

    def _build(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        class _Node(Node):
            def __init__(inner):
                super().__init__('sign_detection_node')
                inner._pub = inner.create_publisher(String, '/detection', 1)

                engine_path = os.path.realpath(ENGINE_PATH)
                inner.get_logger().info(f'Loading TRT engine: {engine_path}')
                inner._trt = TRTInference(engine_path)
                inner.get_logger().info('Engine loaded')

                inner._cap = cv2.VideoCapture(
                    _gst_pipeline(SENSOR_ID), cv2.CAP_GSTREAMER
                )
                if not inner._cap.isOpened():
                    raise RuntimeError(f'cam{SENSOR_ID} 열기 실패')

                inner.create_timer(1.0 / LOOP_HZ, inner._loop)
                inner.get_logger().info('SignDetection node started')

            def _loop(inner):
                ret, frame = inner._cap.read()
                if not ret:
                    return
                inp = preprocess(frame)
                out = inner._trt.infer(inp)
                cls_id, conf, _ = postprocess(out)
                if cls_id is not None:
                    label = CLASS_MAP.get(CLASS_NAMES[cls_id], '')
                    if label:
                        from std_msgs.msg import String as S
                        msg = S()
                        msg.data = label
                        inner._pub.publish(msg)
                        inner.get_logger().debug(
                            f'{CLASS_NAMES[cls_id]} → {label}  ({conf:.2f})'
                        )

            def destroy_node(inner):
                if hasattr(inner, '_cap'):
                    inner._cap.release()
                super().destroy_node()

        return _Node


def main(args=None):
    import rclpy
    NodeClass = SignDetectionNode()._build()
    rclpy.init(args=args)
    node = NodeClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 단독 카메라 테스트  (python3 sign_detection.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    print(f'[test] Loading TRT engine: {os.path.realpath(ENGINE_PATH)}')
    trt_infer = TRTInference(os.path.realpath(ENGINE_PATH))
    print('[test] Engine loaded. Press q to quit.')

    cap = cv2.VideoCapture(_gst_pipeline(SENSOR_ID), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(f'[ERROR] cam{SENSOR_ID} 열기 실패. SENSOR_ID를 확인하세요.')

    cv2.namedWindow('Sign Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        inp = preprocess(frame)
        out = trt_infer.infer(inp)
        cls_id, conf, box = postprocess(out, frame_hw=(fh, fw))

        vis = frame.copy()
        if cls_id is not None:
            label     = CLASS_MAP.get(CLASS_NAMES[cls_id], CLASS_NAMES[cls_id])
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f'{label} {conf:.2f}',
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, 'No detection', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Sign Detection', vis)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

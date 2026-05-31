# coding: utf-8
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort


class Face_Detect:
    """人脸检测模块。

    接口原则：
    - 本文件不读取 config.json。
    - 本文件不写死模型路径、provider、线程数。
    - detect_face/crop 所需阈值和尺寸由 main.py 从 config.py 读取后传入。
    """

    def __init__(self, model_path, provider, threads):
        self.face_model = None
        self.model_path = str(model_path)
        self.provider = provider
        self.threads = int(threads)
        self.load_model()

    def load_model(self):
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = self.threads
            session_options.inter_op_num_threads = 1

            if self.provider == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                available = ort.get_available_providers()
                providers = []
                if "SpaceMITExecutionProvider" in available:
                    providers.append("SpaceMITExecutionProvider")
                providers.append("CPUExecutionProvider")

            self.face_model = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=providers,
            )
            print(f"load face_model success: {self.model_path}", flush=True)
            print(f"using providers: {self.face_model.get_providers()}", flush=True)

        except Exception as e:
            print(f"加载模型失败: {e}", flush=True)
            self.face_model = None

    def img_convert(self, img, img_size):
        try:
            img_size = int(img_size)
            input_img = cv2.resize(img, (img_size, img_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.transpose(2, 0, 1).astype(np.float32)
            input_img *= 1.0 / 255.0
            input_img = np.expand_dims(input_img, axis=0)
            return input_img
        except Exception as e:
            print(f"图像预处理失败: {e}", flush=True)
            return None

    @staticmethod
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union != 0 else 0

    def nms(self, boxes, iou_threshold):
        if len(boxes) == 0:
            return []

        iou_threshold = float(iou_threshold)
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [
                box for box in boxes
                if self.iou(best, box) < iou_threshold
            ]

        return keep

    def detect_face(self, img, img_size, conf_threshold, iou_threshold):
        if img is None:
            return []
        if self.face_model is None:
            return []

        img_size = int(img_size)
        conf_threshold = float(conf_threshold)
        iou_threshold = float(iou_threshold)

        h, w = img.shape[:2]
        input_img = self.img_convert(img, img_size)
        if input_img is None:
            return []

        outputs = self.face_model.run(
            None,
            {self.face_model.get_inputs()[0].name: input_img},
        )

        preds = outputs[0][0]
        if preds.shape[0] == 5 and preds.shape[1] > 5:
            preds = preds.T

        boxes = []
        for p in preds:
            cx, cy, bw, bh, conf = p[:5]
            if conf < conf_threshold:
                continue

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            x1 = int(x1 * w / img_size)
            y1 = int(y1 * h / img_size)
            x2 = int(x2 * w / img_size)
            y2 = int(y2 * h / img_size)

            boxes.append([x1, y1, x2, y2, float(conf)])

        return self.nms(boxes, iou_threshold=iou_threshold)

    def crop(self, frame, boxes, pad, extra_ratio):
        faces = []
        if frame is None:
            return faces

        pad = int(pad)
        extra_ratio = float(extra_ratio)
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2, conf) in boxes:
            x1p = max(0, int(x1) - pad)
            y1p = max(0, int(y1) - pad)
            x2p = min(w, int(x2) + pad)
            y2p = min(h, int(y2) + pad)

            extra = int(max(x2p - x1p, y2p - y1p) * extra_ratio)
            x1p = max(0, x1p - extra)
            y1p = max(0, y1p - extra)
            x2p = min(w, x2p + extra)
            y2p = min(h, y2p + extra)

            face = frame[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue

            faces.append((x1p, y1p, x2p, y2p, conf, face))

        return faces

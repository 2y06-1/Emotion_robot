# coding: utf-8
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

class Face_Detect:
    """
    人脸检测模块。
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
        """
        根据人脸框裁剪正方形人脸ROI。

        与原版本相比：
        1. 减少背景区域；
        2. 保证ROI为正方形；
        3. 避免情绪模型将长方形人脸强行拉伸为正方形；
        4. 保留完整的额头、眼睛和嘴部。
        """
        faces = []

        if frame is None:
            return faces

        pad = int(pad)
        extra_ratio = float(extra_ratio)

        frame_h, frame_w = frame.shape[:2]

        for x1, y1, x2, y2, conf in boxes:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # 使用人脸框中心。
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # 取宽高中较大的值，构造正方形。
            base_size = max(box_w, box_h)

            # pad作为固定像素扩张，extra_ratio作为比例扩张。
            side_length = int(
                base_size * (1.0 + 2.0 * extra_ratio)
                + 2 * pad
            )

            side_length = max(side_length, 1)

            half = side_length / 2.0

            square_x1 = int(round(center_x - half))
            square_y1 = int(round(center_y - half))
            square_x2 = square_x1 + side_length
            square_y2 = square_y1 + side_length

            # 如果正方形超出图像范围，整体平移回来，
            # 而不是直接单边截断导致重新变成长方形。
            if square_x1 < 0:
                square_x2 -= square_x1
                square_x1 = 0

            if square_y1 < 0:
                square_y2 -= square_y1
                square_y1 = 0

            if square_x2 > frame_w:
                shift = square_x2 - frame_w
                square_x1 -= shift
                square_x2 = frame_w

            if square_y2 > frame_h:
                shift = square_y2 - frame_h
                square_y1 -= shift
                square_y2 = frame_h

            square_x1 = max(0, square_x1)
            square_y1 = max(0, square_y1)
            square_x2 = min(frame_w, square_x2)
            square_y2 = min(frame_h, square_y2)

            face = frame[
                square_y1:square_y2,
                square_x1:square_x2,
            ]

            if face.size == 0:
                continue

            faces.append(
                (
                    square_x1,
                    square_y1,
                    square_x2,
                    square_y2,
                    conf,
                    face,
                )
            )

        return faces

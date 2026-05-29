# coding: utf-8
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class Face_Detect:
    def __init__(self, model_path, provider="auto", threads=4):
        self.face_model = None
        self.model_path = str(model_path)
        self.provider = provider
        self.threads = threads
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
                avail = ort.get_available_providers()
                providers = []
                if "SpaceMITExecutionProvider" in avail:
                    providers.append("SpaceMITExecutionProvider")
                providers.append("CPUExecutionProvider")

            self.face_model = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=providers,
            )
            print(f"load face_model success: {self.model_path}")
            print(f"using providers: {self.face_model.get_providers()}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.face_model = None

    def img_convert(self, img, img_size):
        try:
            input_img = cv2.resize(img, (img_size, img_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.transpose(2, 0, 1).astype(np.float32)
            input_img *= 1.0 / 255.0
            input_img = np.expand_dims(input_img, axis=0)
            return input_img
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None

    @staticmethod
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union != 0 else 0

    def nms(self, boxes, iou_threshold=0.4):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [box for box in boxes if self.iou(best, box) < iou_threshold]
        return keep

    def detect_face(self, img, img_size=224, conf_threshold=0.5, iou_threshold=0.4):
        h, w = img.shape[:2]
        input_img = self.img_convert(img, img_size)
        if input_img is None or self.face_model is None:
            return []

        outputs = self.face_model.run(None, {self.face_model.get_inputs()[0].name: input_img})
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

    def crop(self, frame, boxes, pad=20, extra_ratio=0.25):
        faces = []
        h, w = frame.shape[:2]
        for (x1, y1, x2, y2, conf) in boxes:
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run uint8 face detection on camera frames.")
    parser.add_argument("--model", default="/home/bianbu/Emotion_robot/model/vision/best.onnx")
    parser.add_argument("--camera", default="/dev/video20")
    parser.add_argument("--provider", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--extra-ratio", type=float, default=0.25)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    face_model = Face_Detect(args.model, provider=args.provider, threads=args.threads)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("摄像头打开失败，退出")
        exit(1)

    frame_id = 0
    boxes = []
    frame_counter = 0
    start_time = time.time()
    fps = 0.0
    infer_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        frame_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            start_time = time.time()
            print(f"\r实时帧率: {fps:.1f} FPS  face: {infer_ms:.1f} ms  boxes: {len(boxes)}  ", end="")

        frame_id += 1
        if frame_id % args.detect_every == 0:
            t0 = time.perf_counter()
            boxes = face_model.detect_face(
                frame,
                img_size=args.img_size,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

        faces = face_model.crop(frame, boxes, pad=args.pad, extra_ratio=args.extra_ratio)
        if not args.no_show:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for (x1, y1, x2, y2, conf, face) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("face_crop", face)

            frame_show = cv2.resize(frame, (224, 224))
            cv2.imshow("face_emotion", frame_show)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

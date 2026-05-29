# coding: utf-8
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = ["angry", "happy", "neutral", "sad", "surprise"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def make_cpu_session(model_path: str, threads: int = 4) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])


class EmotionClassifier:
    """Bottom-level emotion classifier.

    Input: BGR image, usually a cropped face ROI.
    Output: emotion label and probability.
    """

    def __init__(
        self,
        model_path: str = "/home/bianbu/Emotion_robot/model/vision/emotion_best_uint8_static.onnx",
        img_size: int = 64,
        top_k: int = 1,
        threads: int = 4,
    ):
        self.img_size = img_size
        self.top_k = top_k
        self.session = make_cpu_session(model_path, threads)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[emotion] model: {model_path}")
        print(f"[emotion] providers: {self.session.get_providers()}")

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr_image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        return img[None]

    def predict_probs(self, bgr_image: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: self.preprocess(bgr_image)})[0]
        return softmax(logits)[0]

    def predict_topk(self, bgr_image: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        probs = self.predict_probs(bgr_image)
        k = self.top_k if top_k is None else top_k
        indices = np.argsort(probs)[::-1][:k]
        return [(CLASS_NAMES[int(i)], float(probs[int(i)])) for i in indices]

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]:
        return self.predict_topk(bgr_image, top_k=1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test bottom-level uint8 emotion classifier.")
    parser.add_argument("--model", default="/home/bianbu/Emotion_robot/model/vision/emotion_best.onnx")
    parser.add_argument("--image", default="", help="Test one image instead of camera.")
    parser.add_argument("--camera", default="/dev/video20")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = EmotionClassifier(args.model, top_k=args.top_k, threads=args.threads)

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"cannot read image: {args.image}")
        top = classifier.predict_topk(image, args.top_k)
        print("Top results:")
        for label, prob in top:
            print(f"  {label:8s} {prob:.4f}")
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("cannot open camera, try camera 0")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("camera is not available")

    frames = 0
    fps = 0.0
    infer_ms = 0.0
    last_time = time.perf_counter()

    print("press q or ESC to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        t0 = time.perf_counter()
        top = classifier.predict_topk(frame, args.top_k)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        frames += 1
        now = time.perf_counter()
        if now - last_time >= 1.0:
            fps = frames / (now - last_time)
            frames = 0
            last_time = now
            label, prob = top[0]
            print(f"\rFPS: {fps:.1f}  infer: {infer_ms:.1f} ms  {label} {prob:.2f}   ", end="")

        if not args.no_show:
            y = 30
            for label, prob in top:
                cv2.putText(frame, f"{label}: {prob:.3f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
            cv2.putText(frame, f"FPS {fps:.1f}  {infer_ms:.1f}ms", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("emotion classifier test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

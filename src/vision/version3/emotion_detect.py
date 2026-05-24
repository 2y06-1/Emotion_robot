import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path

class EmotionClassifier:
    def __init__(self, model_path, img_size=64, top_k=1):
        self.img_size = img_size
        self.top_k = top_k
        self.class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        providers = []
        if "SpaceMITExecutionProvider" in ort.get_available_providers():
            providers.append("SpaceMITExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[情绪分类] 激活的 EP: {self.session.get_providers()}")
    def preprocess(self, bgr_image):
        img = cv2.resize(bgr_image, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    def predict(self, bgr_image):
        """返回 (最高概率标签, 最高概率值)"""
        inp = self.preprocess(bgr_image)
        logits = self.session.run(None, {self.input_name: inp})[0]
        probs = self.softmax(logits)[0]
        top_indices = np.argsort(probs)[::-1][:self.top_k]
        top_label = self.class_names[top_indices[0]]
        top_prob  = float(probs[top_indices[0]])
        return top_label, top_prob


if __name__ == "__main__":
    MODEL_PATH = "/home/bianbu/Emotion_robot/model/vision/emotion_best.onnx"

    classifier = EmotionClassifier(MODEL_PATH, top_k=1)

    cap = cv2.VideoCapture("/dev/video20")
    if not cap.isOpened():
        print("无法打开摄像头，尝试系统默认设备(0)")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit("摄像头不可用")

    print("按 'q' 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        labels, probs = classifier.predict(frame)
        y0 = 30
        for i, (lab, prob) in enumerate(zip(labels, probs)):
            text = f"{lab}: {prob:.3f}"
            cv2.putText(frame, text, (10, y0 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Emotion Classification (full frame)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import time
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.vision.version3.face_detect import Face_Detect
from src.vision.version3.emotion_detect import EmotionClassifier

DET_MODEL_PATH = "/home/bianbu/Emotion_robot/model/vision/best.onnx"
EMO_MODEL_PATH = "/home/bianbu/Emotion_robot/model/vision/emotion_best.onnx"

def main():
    # 打开摄像头
    detector = Face_Detect(DET_MODEL_PATH)
    classifier = EmotionClassifier(EMO_MODEL_PATH, top_k=1)   # 只需要最高概率
    # 打开摄像头（K1 MUSE Pi Pro 通常为 /dev/video20）
    cap = cv2.VideoCapture("/dev/video20", cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头，尝试默认设备 /dev/video0")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头不可用，退出")
        exit(1)
    # 帧率统计
    frame_counter = 0
    start_time = time.time()
    fps = 0.0
    frame_id = 0
    boxes = []          # 保存上一次检测结果
    print("按 ESC 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 水平翻转（镜像），使画面更自然
        frame = cv2.flip(frame, 1)
        # 每隔3帧重新检测人脸（节省算力，可调整）
        frame_id += 1
        if frame_id % 3 == 0:
            boxes = detector.detect_face(frame, img_size=224, conf_threshold=0.5)
        # 裁剪人脸并对每个脸做情绪识别
        faces = detector.crop(frame, boxes)
        for (x1, y1, x2, y2, conf, face_img) in faces:
            # 情绪分类
            label, prob = classifier.predict(face_img)
            print(label)
            # 绘制人脸框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示检测置信度和情绪
            text = f"{label} ({prob:.2f})"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 帧率计算与显示
        frame_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # 显示（可缩放窗口便于查看）
        display = cv2.resize(frame, (224, 224))
        cv2.imshow("Face Emotion Recognition", display)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC 键退出
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
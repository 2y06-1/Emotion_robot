import sys
from pathlib import Path
import time
import cv2
from collections import deque, Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.vision.version2.new_face_detect import Face_Detect
from src.vision.version2.new_emotion_detect import Emotion_Detect

emotion_texts = {
    "Happy": "当下人物看起来很开心！",
    "Sad": "当下人物似乎有点难过。",
    "Angry": "当下人物有点生气哦！",
    "Surprise": "当下人物惊讶！",
    "Fear": "当下人物感觉有点害怕。",
    "Neutral": "当下人物保持平静",
    "Disgust": "当下人物有点反感的表情",
    "Contempt": "当下人物有点轻蔑的表情"
}

if __name__ == "__main__":

    face_model_path = BASE_DIR / "model" / "vision" / "yolov8n-face-lindevs.onnx"
    emotion_model_path = BASE_DIR / "model" / "vision" / "enet_backbone.onnx"
    weights_path = BASE_DIR / "model" / "vision" / "gemm_weights.npz"

    face_model = Face_Detect(str(face_model_path))
    emotion_model = Emotion_Detect(str(emotion_model_path), str(weights_path))

    cap = cv2.VideoCapture("/dev/video20", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_counter = 0
    start_time = time.time()
    fps = 0.0
    frame_id = 0
    boxes = []

    STABLE_FRAMES = 4
    emotion_buffer = deque(maxlen=STABLE_FRAMES)
    last_output_emotion = None

    FACE_INTERVAL = 3 # 3帧检测一次人脸
    EMOTION_INTERVAL = 10 # 10帧1次情绪

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_id += 1
        frame_counter += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            start_time = time.time()

        if frame_id % FACE_INTERVAL == 0:
            boxes = face_model.detect_face(frame, img_size=320)

        faces = face_model.crop(frame, boxes)

        largest_face = None
        largest_area = 0

        for (x1, y1, x2, y2, conf, face) in faces:
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_face = (x1, y1, x2, y2, conf, face)

        if largest_face is not None:
            x1, y1, x2, y2, conf, face = largest_face
            if frame_id % EMOTION_INTERVAL == 0:
                detected_emotion = emotion_model.detect_emotion(face, img_size=224)
                emotion_buffer.append(detected_emotion)
                print(emotion_buffer)

                most_common, count = Counter(emotion_buffer).most_common(1)[0]
                if count == STABLE_FRAMES and most_common != last_output_emotion:
                    current_emotion = most_common
                    text_to_show = emotion_texts.get(current_emotion)
                    print(text_to_show)
                    last_output_emotion = current_emotion

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
'''
import sys
from pathlib import Path
import time
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.vision.version2.new_face_detect import Face_Detect
from src.vision.version2.new_emotion_detect import Emotion_Detect
emotion_texts = {
        "Happy": "当下人物看起来很开心！",
        "Sad": "当下人物似乎有点难过。",
        "Angry": "当下人物有点生气哦！",
        "Surprise": "哇，好惊讶！",
        "Fear": "当下人物感觉有点害怕。",
        "Neutral": "当下人物保持平静",
        "Disgust": "当下人物有点反感的表情",
        "Contempt": "当下人物有点轻蔑的表情"
    }

if __name__ == "__main__":

    face_model_path = (BASE_DIR / "model" / "vision" / "yolov8n-face-lindevs.onnx")
    emotion_model_path = (BASE_DIR / "model" / "vision" / "enet_backbone.onnx")
    weights_path = (BASE_DIR / "model" / "vision" / "gemm_weights.npz")

    face_model = Face_Detect(str(face_model_path))
    emotion_model = Emotion_Detect(str(emotion_model_path),str(weights_path))

    cap = cv2.VideoCapture("/dev/video20", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_counter = 0
    start_time = time.time()
    fps = 0.0

    frame_id = 0

    boxes = []

    current_emotion = "Detecting..."

    FACE_INTERVAL = 3
    EMOTION_INTERVAL = 10

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_id += 1
        frame_counter += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            start_time = time.time()

        if frame_id % FACE_INTERVAL == 0:
            boxes = face_model.detect_face(frame,img_size=320)

        faces = face_model.crop(frame, boxes)

        largest_face = None
        largest_area = 0

        for (x1, y1, x2, y2, conf, face) in faces:
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_face = (x1,y1,x2,y2,conf,face)

        if largest_face is not None:
            x1, y1, x2, y2, conf, face = largest_face
            if frame_id % EMOTION_INTERVAL == 0:
                current_emotion = emotion_model.detect_emotion(face,img_size=224)
                
                
                text_to_show = emotion_texts.get(current_emotion)
                print(text_to_show)

        cv2.putText(frame,f"FPS: {fps:.1f}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
        cv2.imshow("Face Emotion",frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
'''
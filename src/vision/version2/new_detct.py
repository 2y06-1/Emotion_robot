import sys
from pathlib import Path
import time
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.vision.version2.new_face_detect import Face_Detect
from src.vision.version2.new_emotion_detect import Emotion_Detect

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
                print("emotion =", current_emotion)
            # cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
            # cv2.putText(frame,current_emotion,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255),2)

        cv2.putText(frame,f"FPS: {fps:.1f}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
        cv2.imshow("Face Emotion",frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
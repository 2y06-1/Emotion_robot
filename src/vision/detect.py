import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
BASE_DIR = Path(__file__).resolve().parent.parent

import cv2
from src.vision.face_detect import Face_Detect
from src.vision.emotion_detect import Emotion_Detect

if __name__ == "__main__":

    face_model_path = BASE_DIR / "model" / "vision" / "yolov8n-face-lindevs.onnx"
    emotion_model_path = BASE_DIR / "model" / "vision" / "enet_b0_8_best_afew.onnx"

    face_model = Face_Detect(str(face_model_path))
    emotion_model = Emotion_Detect(str(emotion_model_path))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_id = 0
    boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_id += 1

        if frame_id % 3 == 0:
            boxes = face_model.detect_face(frame, img_size=320)

        faces = face_model.crop(frame, boxes)


        for (x1, y1, x2, y2, conf, face) in faces:

            emotion = emotion_model.detect_emotion(face,224)

            cv2.putText(frame, emotion, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("face_emotion", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

class Emotion_Detect:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def __str__(self):
        message = []

        inputs = self.emotion_model.get_inputs()
        for i in inputs:
            message.append(f"输入名: {i.name}")
            message.append(f"shape: {i.shape}")
            message.append(f"类型: {i.type}")

        outputs = self.emotion_model.get_outputs()
        for out in outputs:
            message.append(f"输出名: {out.name}")
            message.append(f"shape: {out.shape}")

        return "\n".join(message)

    def load_model(self):
        try:
            self.emotion_model = ort.InferenceSession(self.model_path)
        except Exception as e:
            print(f"加载情绪模型失败: {e}")
            self.emotion_model = None

    def img_convert(self, img, img_size):
        try:
            input_img = cv2.resize(img, (img_size, img_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            input_img = input_img.transpose(2, 0, 1).astype(np.float32)
            input_img *= 1.0 / 255.0

            input_img = np.expand_dims(input_img, axis=0)

            return input_img

        except Exception as e:
            print(f"情绪预处理失败: {e}")
            return None

    def detect_emotion(self, face,img_size):
        input_img = self.img_convert(face,img_size)

        if input_img is None:
            return None

        outputs = self.emotion_model.run(None, {
                self.emotion_model.get_inputs()[0].name: input_img
            })

        pred = outputs[0][0]
        idx = np.argmax(pred)

        return self.get_label(idx)

    def get_label(self, idx):

        labels = [
            "Fear",
            "Contempt",
            "Angry",#
            "Surprise",#
            "Happy",#
            "Neutral",
            "sad",#
            "Disgust"#
        ]

        if 0 <= idx < len(labels):
            return labels[idx]
        return "Unknown"

if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent
    model_path = current_dir.parent.parent/ "model" / "vision" / "enet_b0_8_best_afew.onnx"

    emotion_model = Emotion_Detect(str(model_path))
    print(emotion_model)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        emotion = emotion_model.detect_emotion(frame,224)

        if emotion:
            cv2.putText(frame, emotion, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("emotion_test", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
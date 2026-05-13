from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort
import time

current_dir = Path(__file__).resolve().parent

model_path = current_dir.parent.parent.parent / "model" / "vision" / "enet_backbone.onnx"
weights_path = current_dir.parent.parent.parent / "model" / "vision" / "gemm_weights.npz"


class Emotion_Detect:
    def __init__(self, model_path, weights_path):
        self.model_path = model_path

        data = np.load(weights_path)
        self.W = data['W']
        self.B = data['B']

        self._init_session()

    def _init_session(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        sess_options.log_severity_level = 3

        providers = ["SpaceMITExecutionProvider","CPUExecutionProvider"]

        try:
            self.emotion_model = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )

            print("后端加载成功:")
            print(self.emotion_model.get_providers())

            self.input_name = self.emotion_model.get_inputs()[0].name

            dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

            for _ in range(3):
                self.emotion_model.run(None,{self.input_name: dummy})

        except Exception as e:
            print(f"模型加载失败: {e}")

    def img_convert(self, img, img_size):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(
                img_rgb,
                (img_size, img_size)
            )

            img_data = img_resized.astype(np.float32) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, axis=0)

            return img_data

        except Exception as e:
            print(f"预处理失败: {e}")
            return None

    def detect_emotion(self, face, img_size=224):
        input_tensor = self.img_convert(face, img_size)

        if input_tensor is None:
            return None

        outputs = self.emotion_model.run(
            None,
            {self.input_name: input_tensor}
        )

        features = outputs[0].flatten()
        logits = np.dot(self.W, features) + self.B
        idx = np.argmax(logits)
        
        return self.get_label(idx)

    def get_label(self, idx):

        labels = ["Angry","Contempt","Happy","Surprise","Happy","Neutral","Sad","Disgust"]

        if 0 <= idx < len(labels):
            return labels[idx]

        return "Unknown"


if __name__ == "__main__":

    emotion_model = Emotion_Detect(
        str(model_path),
        str(weights_path)
    )

    cap = cv2.VideoCapture(20)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start_time = time.time()
    frame_count = 0
    fps = 0.0

    infer_interval = 3

    frame_id = 0

    emotion = ""

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % infer_interval == 0:
            emotion = emotion_model.detect_emotion(frame,img_size=224)

        frame_count += 1
        curr_time = time.time()
        elapsed_time = curr_time - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = curr_time

        if emotion:
            cv2.putText(frame,f"Emotion: {emotion}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2)

        cv2.putText(frame,f"FPS: {fps:.1f}",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 255),2)

        cv2.imshow("SpaceMIT NPU Emotion Detect",frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
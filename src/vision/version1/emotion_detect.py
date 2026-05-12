from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort

current_dir = Path(__file__).resolve().parent
model_path = current_dir.parent.parent.parent / "model" / "vision" / "enet_b0_8_best_afew.onnx"

class Emotion_Detect:
    def __init__(self, model_path):
        self.model_path = model_path
        self._init_session()

    def _init_session(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available = ort.get_available_providers()

        providers = []
        if 'SpacemiTExecutionProvider' in available:
            providers.append('SpacemiTExecutionProvider')
        providers.append('CPUExecutionProvider')
    
        try:
            self.emotion_model = ort.InferenceSession(
                self.model_path, 
                sess_options=sess_options, 
                providers=providers
            )
            print(f"当前使用后端: {self.emotion_model.get_providers()}")
            print("load emotion_model success")
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.emotion_model = None

    def __str__(self):
        if not self.emotion_model: return "模型未加载"
        msg = []
        for inp in self.emotion_model.get_inputs():
            msg.append(f"输入: {inp.name} | shape: {inp.shape} | type: {inp.type}")
        for out in self.emotion_model.get_outputs():
            msg.append(f"输出: {out.name} | shape: {out.shape}")
        return "\n".join(msg)

    def img_convert(self, img, img_size):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            gray = gray.astype(np.float32) / 255.0
            gray = np.expand_dims(gray, axis=(0, 1))
            return np.repeat(gray, 3, axis=1)
        except Exception as e:
            print(f"预处理失败: {e}")
            return None

    def detect_emotion(self, face, img_size=224):
        input_tensor = self.img_convert(face, img_size)
        if input_tensor is None:
            return None

        outputs = self.emotion_model.run(
            None, 
            {self.emotion_model.get_inputs()[0].name: input_tensor}
        )
        pred = outputs[0][0]
        idx = np.argmax(pred)
        return self.get_label(idx)

    def get_label(self, idx):
        labels = ["Fear", "Contempt", "Angry", "Surprise", "Happy", "Neutral", "sad", "Disgust"]
        return labels[idx] if 0 <= idx < len(labels) else "Unknown"

if __name__ == "__main__":
    emotion_model = Emotion_Detect(str(model_path))
    
    cap = cv2.VideoCapture(20)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        emotion = emotion_model.detect_emotion(frame, img_size=224) 
        
        if emotion:
            cv2.putText(frame, emotion, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("emotion_test", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()
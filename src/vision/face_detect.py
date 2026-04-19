from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

class Face_Detect:
    def __init__(self,model_path):
        self.model_path = model_path
        self.load_model()

    def __str__(self):
        # 调试用
        message=[]

        inputs = self.face_model.get_inputs()
        for i in inputs:
            message.append(f"输入名: {i.name}")
            message.append(f"shape: {i.shape}")
            message.append(f"类型: {i.type}")

        outputs = self.face_model.get_outputs()
        for out in outputs:
            message.append(f"输入名: {out.name}")
            message.append(f"shape: {out.shape}")

        return "\n".join(message)

    def load_model(self):
        try:
            # 部署到板子上边 self.face_model = ort.InferenceSession(
            #         model_path,
            #         session_options,
            #         providers=["SpaceMITExecutionProvider"]  # 使用SpaceMIT硬件加速
            #     )
            self.face_model = ort.InferenceSession(self.model_path)

        except Exception as e:
            print(f"加载模型失败: {e}")
            self.emotion_model = None

    def img_convert(self, img,img_size):
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

    def iou(self, box1, box2):
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

            boxes = [
                box for box in boxes
                if self.iou(best, box) < iou_threshold
            ]

        return keep

    def detect_face(self, img, img_size,conf_threshold=0.5):
        h, w, _ = img.shape

        input_img = self.img_convert(img,img_size)

        if input_img is None:
            return []

        outputs = self.face_model.run(None, {
            self.face_model.get_inputs()[0].name: input_img
        })

        preds = outputs[0][0]  # (5, 8400)
        preds = preds.T  # (8400, 5)

        boxes = []

        for p in preds:
            x, y, bw, bh, conf = p

            if conf < conf_threshold:
                continue

            x1 = int((x - bw / 2) * w / img_size)
            y1 = int((y - bh / 2) * h / img_size)
            x2 = int((x + bw / 2) * w / img_size)
            y2 = int((y + bh / 2) * h / img_size)

            boxes.append([x1, y1, x2, y2, conf])

        # ===== 4. NMS =====
        boxes = self.nms(boxes, iou_threshold=0.4)

        return boxes

    def crop(self, frame, boxes, pad=20):
        faces = []
        h, w, _ = frame.shape

        for (x1, y1, x2, y2, conf) in boxes:
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            face = frame[y1p:y2p, x1p:x2p]

            if face.size == 0:
                continue

            faces.append((x1p, y1p, x2p, y2p, conf, face))

        return faces

if __name__ == "__main__":
    frame_id = 0
    boxes = []

    current_dir = Path(__file__).resolve().parent.parent
    model_path = current_dir.parent/ "model" / "yolov8n-face-lindevs.onnx"

    face_model = Face_Detect(str(model_path))
    print(face_model)
    cap = cv2.VideoCapture(0)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("face_crop", face)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
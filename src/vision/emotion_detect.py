# coding: utf-8
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def make_onnx_session(model_path: str, provider: str, threads: int) -> ort.InferenceSession:
    """创建 ONNXRuntime 推理会话。"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        available = ort.get_available_providers()
        providers = []
        if "SpaceMITExecutionProvider" in available:
            providers.append("SpaceMITExecutionProvider")
        providers.append("CPUExecutionProvider")

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )


class EmotionClassifier:
    """
    表情分类模块。
    """

    def __init__(
        self,
        model_path,
        img_size,
        top_k,
        threads,
        provider,
        class_names: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        auto_enhance: bool = False,
        enhance_dark_threshold: float = 70.0,
        gamma: float = 1.6,
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.top_k = int(top_k)
        self.class_names = list(class_names)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.auto_enhance = bool(auto_enhance)
        self.enhance_dark_threshold = float(enhance_dark_threshold)
        self.gamma = float(gamma)

        if len(self.class_names) == 0:
            raise ValueError("class_names 不能为空")
        if self.mean.shape[0] != 3 or self.std.shape[0] != 3:
            raise ValueError("mean 和 std 必须是长度为 3 的列表")

        self.session = make_onnx_session(
            model_path=self.model_path,
            provider=provider,
            threads=int(threads),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]


    @staticmethod
    def gray_mean(bgr_image: np.ndarray) -> float:
        if bgr_image is None or bgr_image.size == 0:
            return 0.0
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def enhance_dark_face(self, bgr_image: np.ndarray) -> np.ndarray:
        """暗光增强：只在 ROI 偏暗时启用，尽量不破坏正常光照图像。"""
        if not self.auto_enhance:
            return bgr_image

        mean_gray = self.gray_mean(bgr_image)
        if mean_gray >= self.enhance_dark_threshold:
            return bgr_image

        img = bgr_image.copy()

        # gamma > 1 时，通过 1/gamma 曲线提亮暗部。
        inv_gamma = 1.0 / max(self.gamma, 1e-6)
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        img = cv2.LUT(img, table)

        # 对亮度通道做 CLAHE，提升脸部局部对比度。
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        img = self.enhance_dark_face(bgr_image)

        img = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Ultralytics 分类默认使用 ImageNet mean/std。
        # 如果你导出/训练时确认没有 Normalize，可在 config.json 改成 mean=[0,0,0], std=[1,1,1] 对比测试。
        img = (img - self.mean) / self.std

        img = img.transpose(2, 0, 1).astype(np.float32)
        return img[None]

    def _to_probs(self, raw_output: np.ndarray) -> np.ndarray:
        out = np.asarray(raw_output)
        out = np.squeeze(out)

        # 防止某些模型输出多维，统一拉平成一维类别向量。
        if out.ndim != 1:
            out = out.reshape(-1)

        # 如果 ONNX 已经输出概率，就不要再次 softmax。
        s = float(out.sum())
        if np.all(out >= 0) and 0.98 <= s <= 1.02:
            probs = out.astype(np.float32)
        else:
            probs = softmax(out.astype(np.float32))

        # 如果模型输出类别数比 class_names 多/少，直接打印提醒。
        if len(probs) != len(self.class_names):
            print(
                f"[emotion warn] 模型输出类别数={len(probs)}，class_names数量={len(self.class_names)}，请检查 ONNX 是否导错或标签顺序是否错。",
                flush=True,
            )

        return probs

    def predict_probs(self, bgr_image: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            None,
            {self.input_name: self.preprocess(bgr_image)},
        )
        return self._to_probs(outputs[0])

    def predict_topk(self, bgr_image: np.ndarray, top_k=None) -> list[tuple[str, float]]:
        probs = self.predict_probs(bgr_image)
        k = self.top_k if top_k is None else int(top_k)
        k = max(1, min(k, len(self.class_names), len(probs)))

        indices = np.argsort(probs)[::-1][:k]
        return [
            (self.class_names[int(i)], float(probs[int(i)]))
            for i in indices
        ]

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]:
        return self.predict_topk(bgr_image, top_k=1)[0]

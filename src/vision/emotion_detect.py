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
    """创建 ONNXRuntime 推理会话。

    接口原则：
    - 本文件不读取 config.json。
    - 本文件不写死模型路径、线程数、provider。
    - 所有配置由 main.py 从 config.py 获取后传入。
    """
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
    """表情分类模块。

    输入：
        BGR 图像，通常是人脸检测裁剪出来的 face ROI。

    输出：
        emotion label 和 probability。

    接口原则：
    - 不读取 config.json。
    - 不写死模型路径。
    - 不写死 img_size、top_k、threads、provider。
    - class_names、mean、std 也由外部传入，方便以后换数据集/模型。
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
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.top_k = int(top_k)
        self.class_names = list(class_names)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

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

        print(f"[emotion] model: {self.model_path}", flush=True)
        print(f"[emotion] providers: {self.session.get_providers()}", flush=True)

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            bgr_image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return img[None]

    def predict_probs(self, bgr_image: np.ndarray) -> np.ndarray:
        logits = self.session.run(
            None,
            {self.input_name: self.preprocess(bgr_image)},
        )[0]
        return softmax(logits)[0]

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

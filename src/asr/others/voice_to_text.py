#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量化语音转文本模块 (ONNX Runtime + CTC 解码)
使用 SenseVoice 量化模型，无需额外解码器模型
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
import yaml

# 添加父目录到路径以便导入 frontend 和 postprocess_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend import WavFrontend
from postprocess_utils import rich_transcription_postprocess


class OnnxAsrModel:
    """基于 ONNX Runtime 的语音识别模型 (SenseVoice)"""

    def __init__(self, model_dir: str, device_id: str = "cpu", num_threads: int = 2):
        """
        初始化模型
        Args:
            model_dir: 包含 model_quant.onnx, am.mvn, tokens.txt, config.yaml 的目录
            device_id: "cpu" 或 "cuda:0" (需要 onnxruntime-gpu)
            num_threads: CPU 线程数
        """
        self.model_dir = model_dir
        self.device_id = device_id

        # 1. 加载 ONNX 模型
        model_path = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 配置 ONNX Runtime 选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.enable_cpu_mem_arena = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 设置 providers
        if device_id == "cpu":
            providers = ['CPUExecutionProvider']
        elif device_id.startswith("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

        # 获取输入输出名称（调试用）
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[INFO] 模型输入: {self.input_names}")
        print(f"[INFO] 模型输出: {self.output_names}")

        # 2. 加载 tokens.txt (词表)
        tokens_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"找不到词表文件: {tokens_path}")
        self.token_list = []
        with open(tokens_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: "token\t分数" 或 "token"
                token = line.split('\t')[0].split()[0]
                self.token_list.append(token)
        self.vocab_size = len(self.token_list)
        self.blank_id = 0  # <unk> 作为 blank
        print(f"[INFO] 词表大小: {self.vocab_size}, blank_id: {self.blank_id}")

        # 3. 加载配置文件 config.yaml
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 4. 初始化 WavFrontend (特征提取器)
        frontend_conf = self.config.get('frontend_conf', {})
        cmvn_file = os.path.join(model_dir, "am.mvn")
        if not os.path.exists(cmvn_file):
            raise FileNotFoundError(f"找不到 cmvn 文件: {cmvn_file}")

        # 合并配置，优先使用配置文件中的值
        frontend_args = {
            'cmvn_file': cmvn_file,
            'fs': frontend_conf.get('fs', 16000),
            'window': frontend_conf.get('window', 'hamming'),
            'n_mels': frontend_conf.get('n_mels', 80),
            'frame_length': frontend_conf.get('frame_length', 25),
            'frame_shift': frontend_conf.get('frame_shift', 10),
            'lfr_m': frontend_conf.get('lfr_m', 1),
            'lfr_n': frontend_conf.get('lfr_n', 1),
            'dither': frontend_conf.get('dither', 1.0),
        }
        self.frontend = WavFrontend(**frontend_args)
        print("[INFO] WavFrontend 初始化完成")

        # 5. 语言 ID 和 文本正则化 ID (固定)
        # 参考 sensevoice_bin.py: lid_dict = {"auto":0, "zh":3, "en":4, ...}
        self.language_id = 3   # zh
        self.textnorm_id = 14  # withitn

    def load_audio(self, wav_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        加载音频文件并重采样到目标采样率
        Returns: 1D numpy array, dtype=float32, 范围 [-1, 1]
        """
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # 转为单声道
        if sr != target_sr:
            # 使用 librosa 重采样（需要安装 librosa）
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                # 简单的线性重采样（不够精确，但作为备选）
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        return audio.astype(np.float32)

    def extract_feat(self, waveform: np.ndarray) -> tuple:
        """
        提取 FBank 特征
        Returns:
            feats: (1, T, 80) float32
            feats_len: (1,) int32
        """
        # 调用 frontend 处理
        feat, feat_len = self.frontend.fbank(waveform)
        feat, feat_len = self.frontend.lfr_cmvn(feat)
        # 增加 batch 维度
        feats = np.expand_dims(feat, axis=0).astype(np.float32)
        feats_len = np.array([feat_len], dtype=np.int32)
        return feats, feats_len

    def ctc_decode(self, logits: np.ndarray) -> str:
        """
        CTC 解码: argmax -> 去重 -> 去 blank -> 映射为文本
        Args:
            logits: (1, T, vocab_size) 模型输出
        Returns:
            解码后的原始字符串（未后处理）
        """
        # 取 argmax
        y = np.argmax(logits, axis=-1)[0]  # (T,)
        # 去重 + 去 blank
        prev = -1
        token_ids = []
        for idx in y:
            if idx != prev and idx != self.blank_id:
                token_ids.append(idx)
            prev = idx

        # 映射为字符
        chars = []
        for tid in token_ids:
            token = self.token_list[tid]
            # 处理特殊 token
            if token.startswith('<|') or token == '<unk>' or token == '<s>' or token == '</s>':
                continue  # 跳过特殊 token
            elif token == '▁':
                chars.append(' ')
            elif token.startswith('▁'):
                chars.append(token[1:])  # 去掉开头的 ▁
            else:
                chars.append(token)
        text = ''.join(chars).strip()
        return text

    def __call__(self, wav_path: str, language: str = "zh", use_itn: bool = True) -> str:
        """
        语音转文本主入口
        Args:
            wav_path: 音频文件路径 (16kHz 单声道最好，会自动重采样)
            language: 语言代码，目前支持 "zh", "en"
            use_itn: 是否使用逆文本正则化 (数字、标点转换)
        Returns:
            识别出的文本
        """
        # 设置语言 ID
        lang_map = {"zh": 3, "en": 4}
        self.language_id = lang_map.get(language, 3)
        # 设置 textnorm ID
        self.textnorm_id = 14 if use_itn else 15   # 14: withitn, 15: woitn

        # 1. 加载音频
        waveform = self.load_audio(wav_path)

        # 2. 提取特征
        feats, feats_len = self.extract_feat(waveform)

        # 3. 准备其他输入
        language = np.array([self.language_id], dtype=np.int32)
        textnorm = np.array([self.textnorm_id], dtype=np.int32)

        # 4. ONNX Runtime 推理
        # 注意：输入顺序必须与模型导出时一致，一般顺序为: feats, feats_len, language, textnorm
        # 使用 input_names 确保顺序正确
        feed_dict = {
            self.input_names[0]: feats,
            self.input_names[1]: feats_len,
            self.input_names[2]: language,
            self.input_names[3]: textnorm,
        }
        outputs = self.session.run(self.output_names, feed_dict)
        logits = outputs[0]  # (1, T, vocab_size)

        # 5. CTC 解码
        raw_text = self.ctc_decode(logits)

        # 6. 后处理 (表情、事件等)
        final_text = rich_transcription_postprocess(raw_text)

        return final_text

    def transcribe_batch(self, wav_paths: list, **kwargs) -> list:
        """
        批量识别（逐条处理，保持简洁）
        """
        results = []
        for path in wav_paths:
            text = self.__call__(path, **kwargs)
            results.append(text)
        return results


def main():
    """示例：直接运行脚本进行测试"""
    # 配置路径
    MODEL_DIR = "/home/bianbu/Emotion_robot/model/asr"
    TEST_WAV = "/home/bianbu/Emotion_robot/wav/zh.mp3"   # 替换为你的测试音频文件

    if not os.path.exists(TEST_WAV):
        print(f"测试音频不存在: {TEST_WAV}")
        print("请修改 TEST_WAV 变量指向一个实际的 wav 文件")
        return

    # 初始化模型
    print("正在加载模型...")
    asr = OnnxAsrModel(MODEL_DIR, device_id="cpu", num_threads=2)
    print("模型加载完成！")

    # 识别
    text = asr(TEST_WAV, language="zh", use_itn=True)
    print(f"识别结果: {text}")


if __name__ == "__main__":
    main()
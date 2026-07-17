import html
import json
import random
import re
import unicodedata
from pathlib import Path

import requests


class Ollama_chat:
    """Ollama 情绪共情对话模块。"""

    EMOTION_CN = {
        "happy": "开心",
        "angry": "生气",
        "sad": "难过",
        "surprise": "惊讶",
        "neutral": "平静",
        "no_face": "未知",
    }

    EMOTION_STATUS_REPLIES = {
        "happy": [
            "你现在看起来心情挺开心的。",
            "你现在的情绪比较轻松愉快。",
            "我感觉你现在心情很不错。",
            "你此刻看起来比较开心。",
        ],
        "angry": [
            "你现在的情绪是烦躁和生气。",
            "你现在看起来有些生气和烦躁。",
            "我感觉你此刻有些烦躁。",
            "你现在似乎有点生气。",
        ],
        "sad": [
            "你现在看起来有些难过。",
            "我感觉你此刻的心情有些低落。",
            "你现在的情绪似乎比较低落。",
            "你此刻看起来不太开心。",
        ],
        "surprise": [
            "你现在看起来有些惊讶。",
            "我感觉你此刻有些惊喜和意外。",
            "你现在的情绪似乎比较惊讶。",
            "你此刻看起来有些意外。",
        ],
        "neutral": [
            "你现在的情绪看起来比较平静。",
            "我感觉你此刻的状态比较平稳。",
            "你现在看起来心情比较平静。",
            "你此刻的情绪没有明显波动。",
        ],
        "no_face": [
            "我现在还无法判断你的心情。",
            "我暂时还不能准确判断你的情绪。",
            "目前的信息不足以判断你的心情。",
            "我现在无法准确判断你的情绪。",
        ],
    }

    FALLBACK_REPLIES = {
        "happy": [
            "这份开心真好，我也替你高兴。",
            "你的喜悦很珍贵，愿快乐常伴你。",
        ],
        "angry": [
            "这确实让人窝火，我理解你的感受。",
            "受了这样的委屈，生气也很正常。",
        ],
        "sad": [
            "今天确实很难熬，我会陪着你。",
            "你的难过我听见了，我会陪着你。",
        ],
        "surprise": [
            "这份惊喜太棒了，我也替你开心。",
            "这确实让人惊讶，我懂你的感受。",
        ],
        "neutral": [
            "今天就好好休息，让自己慢慢放松。",
            "你的感受很重要，我会认真听着。",
        ],
        "no_face": [
            "最近确实辛苦了，我会陪着你。",
            "你的感受很重要，我会认真听着。",
        ],
    }

    # 这些词出现在句尾，通常说明句子还没有说完。
    INCOMPLETE_ENDINGS = (
        "但",
        "但是",
        "不过",
        "而且",
        "因为",
        "所以",
        "因此",
        "然后",
        "同时",
        "或者",
        "以及",
        "并且",
        "可是",
        "却",
        "请",
        "希望",
        "希望你",
        "希望你会",
        "一个",
        "一种",
        "一些",
        "这样的",
        "美好的",
        "更好的",
        "的",
        "地",
        "得",
        "和",
        "与",
        "或",
        "把",
        "被",
        "给",
        "让",
        "向",
        "对",
    )

    def __init__(
        self,
        base_url,
        model_name,
        txt_path,
        stream=True,
        timeout=120,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model_name = str(model_name)
        self.txt_path = Path(txt_path)
        self.stream = bool(stream)
        self.timeout = 120 if timeout is None else float(timeout)
        self.api_url = f"{self.base_url}/api/chat"
        self.history = []

        # 记录每种情绪上一次使用的状态回复，
        # 随机选择时尽量避免连续说同一句。
        self._last_emotion_status_reply = {}

        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
        self.txt_path.touch(exist_ok=True)

    # =========================================================
    # 对话历史
    # =========================================================

    def history_append(self, role, content):
        message = {
            "role": str(role),
            "content": str(content),
        }
        self.history.append(message)

        # 最多保留最近6轮，避免小模型被长历史干扰。
        if len(self.history) > 12:
            self.history = self.history[-12:]

        with open(self.txt_path, "a", encoding="utf-8") as file:
            file.write(f"{role}: {content}\n")

    def history_clear(self):
        self.history = []

        with open(self.txt_path, "w", encoding="utf-8") as file:
            file.truncate(0)

        print("对话历史已清空", flush=True)

    def history_show(self):
        for message in self.history:
            role = message.get("role", "")
            content = message.get("content", "")
            print(f"{role}: {content}", flush=True)

    # =========================================================
    # 主对话接口
    # =========================================================

    def chat_ollama(self, user_message, system_prompt=None):
        user_message = str(user_message or "").strip()
        system_prompt = str(system_prompt or "").strip()

        emotion = self._infer_emotion(system_prompt)

        # 用户明确询问自己当前的心情、情绪或表情时，
        # 直接根据本轮视觉状态回答，不让历史对话干扰判断。
        direct_reply = self._direct_emotion_status_reply(
            user_message=user_message,
            emotion=emotion,
        )

        if direct_reply:
            print(f"{self.model_name} > {direct_reply}", flush=True)

            self.history_append("user", user_message)
            self.history_append("assistant", direct_reply)

            return direct_reply

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # 只发送最近3轮历史。
        messages.extend(self.history[-6:])

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # 第一次正常生成。
        raw_reply = self._request_model(
            messages=messages,
            temperature=0.25,
            num_predict=64,
        )
        reply = self._clean_reply(raw_reply)

        # 回复过长、过短或句子残缺时，再压缩重写一次。
        if not reply:
            print(
                f"[LLM] 首次回复不合格，尝试压缩重写：{raw_reply}",
                flush=True,
            )

            repair_raw_reply = self._repair_reply(
                draft=raw_reply,
                emotion=emotion,
                user_message=user_message,
            )
            reply = self._clean_reply(repair_raw_reply)

        # 第二次仍不合格，使用完整兜底句。
        if not reply:
            reply = self._fallback_reply(
                emotion=emotion,
                user_message=user_message,
            )

            print(
                f"[LLM] 使用情绪兜底回复：{reply}",
                flush=True,
            )

        print(f"{self.model_name} > {reply}", flush=True)

        self.history_append("user", user_message)
        self.history_append("assistant", reply)

        return reply

    # =========================================================
    # Ollama 请求
    # =========================================================

    def _request_model(
        self,
        messages,
        temperature=0.25,
        num_predict=64,
    ):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            "options": {
                "temperature": temperature,
                "top_p": 0.70,
                "top_k": 20,
                "repeat_penalty": 1.12,

                # 这里限制的是token，不是汉字数量。
                # 必须给足空间，让模型先生成完整句子。
                "num_predict": num_predict,
                "num_ctx": 2048,
            },
        }

        response = requests.post(
            self.api_url,
            json=payload,
            stream=self.stream,
            timeout=(5, self.timeout),
        )
        response.raise_for_status()

        if self.stream:
            return self._read_stream_response(response)

        data = response.json()
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _read_stream_response(response):
        full_reply = []

        for line in response.iter_lines():
            if not line:
                continue

            try:
                chunk = json.loads(
                    line.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                continue

            content = (
                chunk
                .get("message", {})
                .get("content", "")
            )

            if content:
                full_reply.append(content)

            if chunk.get("done"):
                break

        return "".join(full_reply)

    # =========================================================
    # 超长回复压缩重写
    # =========================================================

    def _repair_reply(
        self,
        draft,
        emotion,
        user_message,
    ):
        emotion_cn = self.EMOTION_CN.get(
            emotion,
            "平静",
        )

        repair_system_prompt = """
你是中文短句改写器。

你的任务是把原回复改写成一句完整、自然的中文共情回复。

必须遵守：
1. 只输出改写后的最终句子。
2. 必须是一句完整的话。
3. 总长度为8到30个可见字符。
4. 不得从中间截断句子。
5. 不得以“但、不过、因为、所以、请、一个、的”等词结尾。
6. 可以自然提问，但不要连续追问或给用户压力。
7. 不说教，不使用表情符号。
8. 不输出解释、字数或角色名称。
""".strip()

        repair_user_prompt = f"""
用户情绪：{emotion_cn}
用户原话：{user_message}
原始回复：{draft}

请将原始回复压缩改写为一句10到30字的完整共情回复。
""".strip()

        repair_messages = [
            {
                "role": "system",
                "content": repair_system_prompt,
            },
            {
                "role": "user",
                "content": repair_user_prompt,
            },
        ]

        try:
            return self._request_model(
                messages=repair_messages,
                temperature=0.10,
                num_predict=48,
            )
        except Exception as exc:
            print(
                f"[LLM] 压缩重写失败：{exc}",
                flush=True,
            )
            return ""

    # =========================================================
    # 回复清洗和完整性检测
    # =========================================================

    @classmethod
    def _clean_reply(cls, text):
        """
        清洗模型回复。

        重要：
        这里绝对不再使用 text[:30] 强制截断。
        不合格就返回空字符串，交给重写或兜底处理。
        """
        text = str(text or "")

        # 删除思考标签。
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        text = html.unescape(text).strip()
        text = cls._remove_emojis(text)

        # 删除角色前缀。
        text = re.sub(
            r"^(助手|机器人|AI|回复|回答|"
            r"assistant)\s*[：:]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # 删除Markdown列表符号。
        text = re.sub(
            r"^\s*[-*•#]+\s*",
            "",
            text,
        )

        # 删除换行和多余空格。
        text = re.sub(r"\s+", "", text)

        # 保留正常的疑问语气。
        # 例如“你好！有什么可以帮助你的吗？”属于自然回复。

        # 删除外层引号。
        text = cls._strip_outer_quotes(text)

        if not text:
            return ""

        forbidden_phrases = [
            "当前情绪",
            "情绪识别",
            "视觉情绪",
            "通过摄像头",
            "检测到你",
            "置信度",
            "系统提示",
            "作为一个人工智能",
            "作为AI",
            "作为一个AI",
            "我无法感受",
            "用户：",
            "助手：",
            "回复：",
        ]

        if any(
            phrase in text
            for phrase in forbidden_phrases
        ):
            return ""

        # 如果整段是一至两句完整短句，并且总长度合格，
        # 直接保留整段，例如“你好！有什么可以帮助你的吗？”。
        sentence_end_count = len(
            re.findall(r"[。！？!?]", text)
        )

        if (
            text.endswith(("。", "！", "？", "!", "?"))
            and 1 <= sentence_end_count <= 2
            and cls._is_valid_sentence(text)
        ):
            return text

        # 整段不合格时，再尝试提取其中一条完整句子。
        complete_sentences = re.findall(
            r"[^。！？!?\n]+[。！？!?]",
            text,
        )

        for sentence in complete_sentences:
            sentence = sentence.strip()

            if cls._is_valid_sentence(sentence):
                return sentence

        # 模型有时不输出标点，但内容本身已经完整。
        if not any(mark in text for mark in "。！？!?"):
            candidate = text.strip("，、；：")

            # 加句号之后仍不得超过30字。
            if len(candidate) <= 29:
                candidate += "。"

            if cls._is_valid_sentence(candidate):
                return candidate

        # 太长、太短或句子残缺，返回空。
        # 由上层执行压缩重写，而不是强制截断。
        return ""

    @classmethod
    def _is_valid_sentence(cls, sentence):
        sentence = str(sentence or "").strip()

        if not sentence:
            return False

        length = len(re.sub(r"\s+", "", sentence))

        if not 8 <= length <= 30:
            return False

        if "\n" in sentence:
            return False

        if cls._is_incomplete_sentence(sentence):
            return False

        return True

    @classmethod
    def _is_incomplete_sentence(cls, sentence):
        # 先去掉结尾标点。
        content = sentence.rstrip(
            "。！？，、；：,.!?"
        ).strip()

        if not content:
            return True

        for ending in cls.INCOMPLETE_ENDINGS:
            if content.endswith(ending):
                return True

        return False

    @staticmethod
    def _strip_outer_quotes(text):
        quote_pairs = [
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("「", "」"),
            ("『", "』"),
        ]

        text = str(text or "").strip()
        changed = True

        while changed and len(text) >= 2:
            changed = False

            for left, right in quote_pairs:
                if (
                    text.startswith(left)
                    and text.endswith(right)
                ):
                    text = text[1:-1].strip()
                    changed = True
                    break

        return text

    @staticmethod
    def _remove_emojis(text):
        result = []

        for char in str(text or ""):
            code = ord(char)
            category = unicodedata.category(char)

            if category == "So":
                continue

            if (
                0x1F300 <= code <= 0x1FAFF
                or 0x2600 <= code <= 0x27BF
            ):
                continue

            result.append(char)

        return "".join(result)

    # =========================================================
    # 情绪识别和兜底
    # =========================================================

    @staticmethod
    def _infer_emotion(system_prompt):
        """
        从本轮 system prompt 中提取当前情绪。

        优先读取“当前用户情绪：xxx”这一行，避免风格示例、
        历史内容或其他说明文字中的情绪词影响判断。
        """
        prompt = str(system_prompt or "")

        match = re.search(
            r"当前用户情绪\s*[：:]\s*([^。\n]+)",
            prompt,
        )
        current_emotion_text = (
            match.group(1).strip()
            if match
            else prompt
        )

        checks = [
            ("生气烦躁", "angry"),
            ("烦躁和生气", "angry"),
            ("生气", "angry"),
            ("烦躁", "angry"),
            ("难过低落", "sad"),
            ("难过", "sad"),
            ("低落", "sad"),
            ("伤心", "sad"),
            ("开心", "happy"),
            ("高兴", "happy"),
            ("惊喜惊讶", "surprise"),
            ("惊喜", "surprise"),
            ("惊讶", "surprise"),
            ("平静", "neutral"),
            ("未知", "no_face"),
        ]

        for keyword, emotion in checks:
            if keyword in current_emotion_text:
                return emotion

        return "neutral"

    @staticmethod
    def _is_emotion_status_query(user_message):
        """
        判断用户是否在询问自己的当前心情、情绪或表情。

        只匹配明确的状态查询，不会把“我心情不好怎么办”
        这类普通倾诉误判成视觉状态查询。
        """
        text = re.sub(
            r"[\s，。！？、,.!?；;：:]",
            "",
            str(user_message or ""),
        )

        if not text:
            return False

        patterns = [
            r"(你觉得|你看|你认为|帮我看|看看)?"
            r"我(现在|目前)?(的)?"
            r"(心情|情绪|表情)"
            r"(是)?(什么|怎么样|怎样|如何|什么样)",

            r"我(现在|目前)?"
            r"(是|属于)?"
            r"(什么|哪种|怎样的?)"
            r"(心情|情绪|表情)",

            r"(你能|能不能|可以)?"
            r"(看出|判断|识别)(一下)?"
            r"我(现在|目前)?(的)?"
            r"(心情|情绪|表情)",
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    def _direct_emotion_status_reply(
        self,
        user_message,
        emotion,
    ):
        """
        对明确的当前情绪查询，从对应模板中随机选择一句。

        该分支不调用大模型，因此不会受历史对话干扰；
        同一情绪连续查询时，尽量避免重复上一句。
        """
        if not self._is_emotion_status_query(user_message):
            return None

        if emotion not in self.EMOTION_STATUS_REPLIES:
            emotion = "no_face"

        replies = self.EMOTION_STATUS_REPLIES[emotion]
        last_reply = self._last_emotion_status_reply.get(emotion)

        candidates = [
            reply
            for reply in replies
            if reply != last_reply
        ]

        if not candidates:
            candidates = replies

        reply = random.choice(candidates)
        self._last_emotion_status_reply[emotion] = reply

        return reply

    @classmethod
    def _fallback_reply(
        cls,
        emotion="neutral",
        user_message="",
    ):
        if emotion not in cls.FALLBACK_REPLIES:
            emotion = "neutral"

        replies = cls.FALLBACK_REPLIES[emotion]

        index = (
            sum(ord(char) for char in str(user_message))
            % len(replies)
        )

        return replies[index]
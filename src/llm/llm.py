import json
import re
import html
from pathlib import Path

import requests


class Ollama_chat:
    """Ollama 对话模块。"""

    def __init__(self, base_url, model_name, txt_path, stream, timeout):
        self.base_url = str(base_url).rstrip("/")
        self.model_name = str(model_name)
        self.txt_path = Path(txt_path)
        self.stream = bool(stream)
        self.timeout = None if timeout is None else float(timeout)

        self.api_url = f"{self.base_url}/api/chat"
        self.history = []

        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
        self.txt_path.touch(exist_ok=True)

    def history_append(self, role, content):
        self.history.append({"role": role, "content": content})

        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"{role}: {content}\n")

    def chat_ollama(self, user_message, system_prompt=None):
        """发送对话给 Ollama。

        user_message：用户真正说的话。
        system_prompt：机器人身份、情绪策略等系统提示。

        """
        user_message = str(user_message or "").strip()
        system_prompt = str(system_prompt or "").strip()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
        }

        response = requests.post(
            self.api_url,
            json=payload,
            stream=self.stream,
            timeout=self.timeout,
        )
        response.raise_for_status()

        print(f"{self.model_name} > ", end="", flush=True)

        if not self.stream:
            data = response.json()
            reply = data.get("message", {}).get("content", "")
            reply = self._clean_reply(reply)
            print(reply, flush=True)
            self.history_append("user", user_message)
            self.history_append("assistant", reply)
            return reply

        full_reply = ""
        in_think = False
        buffer = ""

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            content = chunk.get("message", {}).get("content", "")
            if content:
                buffer += content

                if not in_think:
                    think_start = buffer.find("<think>")
                    if think_start != -1:
                        part = buffer[:think_start]
                        if part:
                            print(part, end="", flush=True)
                            full_reply += part
                        in_think = True
                        buffer = buffer[think_start + len("<think>"):]
                    else:
                        print(buffer, end="", flush=True)
                        full_reply += buffer
                        buffer = ""
                else:
                    think_end = buffer.find("</think>")
                    if think_end != -1:
                        part = buffer[think_end + len("</think>"):]
                        if part:
                            print(part, end="", flush=True)
                            full_reply += part
                        in_think = False
                        buffer = ""
                    else:
                        buffer = ""

            if chunk.get("done"):
                break

        full_reply = self._clean_reply(full_reply)
        print(flush=True)
        self.history_append("user", user_message)
        self.history_append("assistant", full_reply)
        return full_reply

    def history_clear(self):
        self.history = []
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.truncate(0)
        print("对话历史已清空", flush=True)

    def history_show(self):
        for msg in self.history:
            print(f"{msg['role']}: {msg['content'][:100]}...", flush=True)

    @staticmethod
    def _remove_think_content(text):
        while True:
            start = text.find("<think>")
            end = text.find("</think>")
            if start == -1 or end == -1 or end < start:
                break
            text = text[:start] + text[end + len("</think>"):]
        return text

    @classmethod
    def _clean_reply(cls, text):
        text = cls._remove_think_content(str(text or "")).strip()
        text = html.unescape(text).strip()

        # 去掉模型可能输出的外层引号。
        quote_pairs = [
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("「", "」"),
            ("『", "』"),
        ]
        changed = True
        while changed and len(text) >= 2:
            changed = False
            for left, right in quote_pairs:
                if text.startswith(left) and text.endswith(right):
                    text = text[1:-1].strip()
                    changed = True

        # 如果模型仍然把系统提示词开头复述出来，做一个兜底截断。
        bad_starts = [
            "你是一个智能情感陪伴机器人",
            "基本要求：",
            "当前可靠的视觉情绪信息：",
            "当前视觉情绪识别结果：",
        ]
        if any(text.startswith(s) for s in bad_starts):
            # 这种回复不可用，返回空，让 main.py 使用兜底句。
            return ""

        # 去掉 Markdown 列表符号。
        text = re.sub(r"^\s*[-*•]\s*", "", text).strip()
        return text

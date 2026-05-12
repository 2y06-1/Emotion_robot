import requests
import json
from pathlib import Path
import os

current_dir = Path(__file__).resolve().parent
base_url = "http://localhost:11434"
model_name = "my_test:latest"
txt_path = current_dir.parent.parent/"model"/"llm"/"chat_history.txt"

class Ollama_chat:
    def __init__(self, base_url, model_name, txt_path):
        self.base_url = base_url
        self.model_name = model_name
        self.txt_path = txt_path
        self.api_url = f"{base_url}/api/chat"
        self.history = []

    def history_append(self, role, content):
        self.history.append({"role": role, "content": content})
        
        with open(self.txt_path, "a") as f:
            f.write(f"{role}: {content}\n")

    def chat_ollama(self, user_message):
        self.history_append("user", user_message)  

        payload = {
            "model": self.model_name,
            "messages": self.history,
            "stream": True
        }

        response = requests.post(self.api_url, json=payload, stream=True)
        print(f"{self.model_name} > ", end="", flush=True)
        full_reply = ""
        in_think = False
        buffer = ""

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue
                buffer += content
                if not in_think:
                    think_start = buffer.find("<think>")
                    if think_start != -1:
                        part = buffer[:think_start]
                        if part:
                            print(part, end="", flush=True)
                            full_reply += part
                        in_think = True
                        buffer = buffer[think_start + 7:] 
                    else:
                        print(buffer, end="", flush=True)
                        full_reply += buffer
                        buffer = ""
                else:
                    think_end = buffer.find("</think>")
                    if think_end != -1:
                        part = buffer[think_end + 8:]  
                        if part:
                            print(part, end="", flush=True)
                            full_reply += part
                        in_think = False
                        buffer = ""
                    else:
                        buffer = ""

                if chunk.get("done"):
                    break

        print()
        self.history_append("assistant", full_reply)
        return full_reply 

    def history_clear(self):
        self.history = []
        with open(self.txt_path, "w") as f:  
            f.truncate(0)
        print("对话历史已清空")

    def history_show(self):
        for msg in self.history:
            print(f"{msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    bot = Ollama_chat(base_url, model_name, txt_path)
    print("情感陪伴机器人已启动），输入 quit 退出,clear 清空历史,show 展示历史s\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("quit")
            break
        elif user_input.lower() == "clear":
            bot.history_clear()  
            continue
        elif user_input.lower() == "show":
            bot.history_show()  
            continue
        full_reply=bot.chat_ollama(user_input)  


'''
import requests
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent
base_url = "http://localhost:11434"
model_name = "Emotiom1:latest"
txt_path = current_dir.parent.parent / "model" / "llm" / "chat_history.txt"

class Ollama_chat:
    def __init__(self, base_url, model_name, txt_path):
        self.base_url = base_url
        self.model_name = model_name
        self.txt_path = txt_path
        self.api_url = f"{base_url}/api/chat"

        # 系统提示固定为情感陪伴机器人
        self.system_prompt= {
            "role": "system",
            "content": (
                "你是一个智能情感陪伴机器人。\n"
                "回复规则：每轮对话你【只输出一句】温暖、共情的话语。\n"
                "绝对禁止：输出分析、解释、自问自答、代码、多余标点或换行。\n"
                "示例：用户说'今天好累'，你回复'你辛苦了，给自己一点休息时间吧。'"
            )
        }

        self.history = [self.system_prompt]

    def history_append(self, role, content):
        self.history.append({"role": role, "content": content})
        with open(self.txt_path, "a") as f:
            f.write(f"{role}: {content}\n")
 
    def chat_ollama(self, user_message):
        self.history_append("user", user_message)

        payload = {
            "model": self.model_name,
            "messages": self.history,
            "stream": True,
            "options": 
            {                
            "num_predict": 40,     
            "temperature": 0.7,    
            }
        }  

        response = requests.post(self.api_url, json=payload, stream=True)
        print(f"{self.model_name} > ", end="", flush=True)

        full_reply = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_reply += content
                if chunk.get("done"):
                    break

        print()  
        self.history_append("assistant", full_reply)

    def history_clear(self):
        self.history = [self.system_prompt]  
        with open(self.txt_path, "w") as f:
            f.truncate(0)
        print("对话历史已清空")

    def history_show(self):
        for msg in self.history[1:]:
            print(f"{msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    bot = Ollama_chat(base_url, model_name, txt_path)
    print("情感陪伴机器人已启动，输入 quit 退出, clear 清空历史, show 展示历史\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("quit")
            break
        elif user_input.lower() == "clear":
            bot.history_clear()
            continue
        elif user_input.lower() == "show":
            bot.history_show()
            continue
        bot.chat_ollama(user_input)
'''
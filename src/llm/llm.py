import requests
import json

class Ollama_chat:
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/api/chat"
        self.history = []

    def history_append(self, role, content):
        self.history.append({"role": role, "content": content})
        
        with open("/home/bianbu/mycode/src/llm/chat_history.txt", "a") as f:
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

    def clear_history(self):
        self.history = []
        print("对话历史已清空")

    def show_history(self):
        for msg in self.history:
            print(f"{msg['role']}: {msg['content'][:100]}...")


if __name__ == "__main__":
    # 只需要初始化一次
    bot = Ollama_chat("http://localhost:11434", "qwen3:0.6b")
    print("情感陪伴机器人已启动（流式多轮对话），输入 quit 退出，clear 清空历史\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("quit")
            break
        elif user_input.lower() == "clear":
            bot.clear_history()
            continue
        bot.chat_ollama(user_input)
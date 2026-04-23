import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"

conversation_history = []

def chat_with_ollama(prompt: str):
    conversation_history.append({"role": "user", "content": prompt})

    payload = {
        "model": "qwen3:0.6b",  
        "messages": conversation_history,
        "stream": True  
    }

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status() 
            full_response = ""
            print("\n助手: ", end="", flush=True)

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content

                    if chunk.get('done', False):
                        conversation_history.append({"role": "assistant", "content": full_response})
                        print() 

    except requests.exceptions.RequestException as e:
        print(f"无法连接到Ollama API: {e}")
    except json.JSONDecodeError as e:
        print(f"解析JSON响应失败: {e}")


if __name__ == "__main__":
    print("Ollama 命令行聊天程序")
    print("模型: qwen3:0.6b | 输入 'exit' 或 'quit' 退出。")
    print("-" * 50)

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["exit", "quit"]:
            print("再见!")
            break
        
        chat_with_ollama(user_input)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
import tempfile
import subprocess
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

from ui import MainWindow


# ============================================
# 配置
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.asr.new_voice_collect import Voice_Collect, has_voice
from src.asr.voice_tranform import Voice_Transform
from src.llm.llm import Ollama_chat
from src.emotion.emotion_detector import EmotionDetector

CONFIG = {
    "voice_path": Path(tempfile.gettempdir()) / "emotion_robot_wav",
    "device_id": 2,
    "base_url": "http://localhost:11434",
    "model_name": "my_test:latest",
    "txt_path": BASE_DIR / "model" / "llm" / "chat_history.txt",
    "tts_script": "/home/bianbu/Emotion_robot/src/asr/tts_worker.py",
    "tts_python": "/home/bianbu/Emotion_robot/venv_tts/bin/python",
    "init_wav": "/home/bianbu/Emotion_robot/wav/init.wav",
}


# ============================================
# 工具函数
# ============================================
def is_valid_text(text):
    """文本有效性检测"""
    invalid_texts = {"", " ", "嗯", "啊", "哦", "呃", "额", "哈", "测试", "字幕", "空", "谢谢观看"}
    if len(text) < 2:
        return False
    if text in invalid_texts:
        return False
    if len(set(text)) <= 1:
        return False
    return True


def speak(tts_process, text):
    """TTS播放"""
    if not text.strip() or tts_process is None:
        return
    try:
        if tts_process.stdin and tts_process.poll() is None:
            tts_process.stdin.write(text + "\n")
            tts_process.stdin.flush()
            while True:
                line = tts_process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                print(line)
                if line == "TTS_DONE":
                    break
    except Exception as e:
        print(f"TTS播放错误: {e}")


# ============================================
# 主程序
# ============================================
def main():
    app = QApplication(sys.argv)
    
    # 1. 创建UI
    window = MainWindow()
    
    # 2. 初始化后端模块
    print("初始化后端模块...")
    
    # 语音采集
    vc = Voice_Collect(CONFIG["voice_path"], CONFIG["device_id"])
    print("语音采集初始化成功")
    
    # ASR
    asr = Voice_Transform()
    print("ASR初始化成功")
    
    # LLM
    bot = Ollama_chat(CONFIG["base_url"], CONFIG["model_name"], str(CONFIG["txt_path"]))
    print("LLM初始化成功")
    
    # TTS进程
    tts_process = subprocess.Popen(
        [CONFIG["tts_python"], CONFIG["tts_script"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print("TTS初始化成功")
    
    # 情绪检测
    emotion_detector = EmotionDetector()
    emotion_detecting = False  # 标记是否在进行情绪检测
    print("情绪检测初始化成功")
    
    # 3. 状态变量
    in_chat_mode = False      # 是否在聊天模式
    listening_for_response = False  # 是否在等待用户回复
    current_emotion = "--"    # 当前情绪
    
    # 4. 播放启动音
    try:
        subprocess.run(
            ["aplay", "-D", "plughw:0,0", CONFIG["init_wav"]],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception as e:
        print(f"播放启动音失败: {e}")
    
    # ========================================
    # 情绪检测线程（独立运行）
    # ========================================
    def run_emotion_detection():
        nonlocal current_emotion
        
        while True:
            try:
                if emotion_detecting and not in_chat_mode:
                    emotion, description, is_strong = emotion_detector.detect()
                    if emotion:
                        current_emotion = emotion
                        window.set_emotion(emotion, description, is_strong)
                        
                        # 检测到强烈情绪，自动聊天
                        if is_strong and not in_chat_mode:
                            print(f"检测到强烈情绪：{emotion}")
                            window.set_state_emotion_chat(emotion, description)
                            window.set_status(f"检测到强烈情绪：{emotion}，主动聊天中")
                            
                            # 生成AI回复
                            prompt = f"检测到用户的情绪是{emotion}，请基于此情绪给予适当的安慰和回应。"
                            reply = bot.chat_ollama(prompt)
                            window.append_ai_message(reply)
                            speak(tts_process, reply)
                            
                            # 重置状态，继续检测
                            window.set_state_emotion_detecting()
                time.sleep(0.1)
            except Exception as e:
                print(f"情绪检测错误: {e}")
                time.sleep(1)
    
    # 启动情绪检测线程
    import threading
    emotion_thread = threading.Thread(target=run_emotion_detection, daemon=True)
    emotion_thread.start()
    
    # 开始情绪检测
    emotion_detecting = True
    
    # ========================================
    # 录音和聊天处理函数
    # ========================================
    def handle_record_toggle():
        """处理录音开始/停止切换"""
        nonlocal in_chat_mode, listening_for_response
        
        if not in_chat_mode:
            # 第一次点击录音：进入聊天模式
            print("进入聊天模式")
            in_chat_mode = True
            listening_for_response = True
            
            # 关闭情绪检测
            nonlocal emotion_detecting
            emotion_detecting = False
            print("情绪检测已关闭")
            
            # 开始录音
            window.set_state_listening()
            record_and_recognize()
            
        else:
            # 第二次点击录音：停止录音，处理语音
            if listening_for_response:
                vc.stop_recording()
                listening_for_response = False
                window.set_state_thinking()
    
    def record_and_recognize():
        """录音并识别"""
        def _record():
            nonlocal listening_for_response
            
            try:
                wav_file = vc.record_audio()
                
                if not listening_for_response:
                    return
                
                if not has_voice(wav_file, threshold=500, min_voice_sec=2):
                    print("录音全程静音，跳过")
                    window.set_state_chatting()
                    return
                
                # ASR识别
                text = asr.speech_to_text(wav_file)
                text = text.strip()
                print(f"识别结果: {text}")
                
                if not is_valid_text(text):
                    print("文本无效，跳过")
                    window.set_state_chatting()
                    return
                
                # 显示用户消息
                window.append_user_message(text)
                
                # LLM处理
                window.set_state_thinking()
                reply = bot.chat_ollama(text)
                reply = reply.strip()
                print(f"AI回复: {reply}")
                
                if not reply:
                    window.set_state_chatting()
                    return
                
                # 显示AI消息并播放
                window.set_state_speaking()
                window.append_ai_message(reply)
                speak(tts_process, reply)
                
                # 恢复聊天状态，等待下次录音
                window.set_state_chatting()
                
            except Exception as e:
                print(f"录音识别错误: {e}")
                window.set_state_error(str(e))
        
        # 在新线程中执行录音
        threading.Thread(target=_record, daemon=True).start()
    
    def handle_exit_chat():
        """处理退出聊天"""
        nonlocal in_chat_mode, emotion_detecting
        
        print("退出聊天模式")
        in_chat_mode = False
        
        # 关闭录音
        if hasattr(vc, 'stop_recording'):
            vc.stop_recording()
        
        # 恢复情绪检测
        emotion_detecting = True
        window.set_state_emotion_detecting()
        window.set_emotion(current_emotion, f"当前情绪：{current_emotion}")
    
    def handle_exit_program():
        """处理退出程序"""
        print("退出程序")
        
        # 停止所有
        if hasattr(vc, 'stop_recording'):
            vc.stop_recording()
        
        try:
            tts_process.terminate()
            tts_process.wait(2)
        except:
            pass
        
        try:
            emotion_detector.release()
        except:
            pass
        
        QApplication.instance().quit()
    
    # ========================================
    # 绑定信号
    # ========================================
    window.record_button_clicked.connect(handle_record_toggle)
    window.exit_chat_clicked.connect(handle_exit_chat)
    window.exit_program_clicked.connect(handle_exit_program)
    
    # ========================================
    # 显示窗口
    # ========================================
    window.showFullScreen()
    print("系统启动完成")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
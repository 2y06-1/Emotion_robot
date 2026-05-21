import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QFrame, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal


class MainWindow(QWidget):
    # 信号定义
    record_button_clicked = pyqtSignal()          # 录音按钮被点击（开始/停止切换）
    exit_chat_clicked = pyqtSignal()              # 退出聊天按钮被点击
    exit_program_clicked = pyqtSignal()           # 退出程序按钮被点击
    
    def __init__(self):
        super().__init__()
        self.current_state = "emotion"  # emotion / listening / chatting / thinking / speaking
        self.is_recording = False       # 是否正在录音
        self.setWindowTitle("Emotion Robot")
        self.setMinimumSize(800, 480)
        
        self._build_ui()
        self._connect_signals()
        
        # 初始化状态
        self.set_state_emotion_detecting()
        self.set_emotion("--", "等待检测人物情绪")
    
    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        self.setStyleSheet(self._style_sheet())
        
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)
        
        # ========== 顶部状态栏 ==========
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)
        
        self.title_label = QLabel("Emotion Robot")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        
        self.status_label = QLabel("状态：情绪检测中")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        
        top_bar.addWidget(self.title_label, 1)
        top_bar.addWidget(self.status_label, 2)
        
        # ========== 中间区域 ==========
        center_layout = QHBoxLayout()
        center_layout.setSpacing(14)
        
        # ----- 左侧情绪卡片 -----
        self.emotion_card = QFrame()
        self.emotion_card.setObjectName("emotionCard")
        self.emotion_card.setFrameShape(QFrame.StyledPanel)
        
        emotion_layout = QVBoxLayout()
        emotion_layout.setContentsMargins(18, 18, 18, 18)
        emotion_layout.setSpacing(12)
        
        self.emotion_icon_label = QLabel("🙂")
        self.emotion_icon_label.setObjectName("emotionIcon")
        self.emotion_icon_label.setAlignment(Qt.AlignCenter)
        
        self.emotion_name_label = QLabel("当前情绪：--")
        self.emotion_name_label.setObjectName("emotionName")
        self.emotion_name_label.setAlignment(Qt.AlignCenter)
        
        self.emotion_text_label = QLabel("等待检测人物情绪")
        self.emotion_text_label.setObjectName("emotionText")
        self.emotion_text_label.setAlignment(Qt.AlignCenter)
        self.emotion_text_label.setWordWrap(True)
        
        # 状态指示灯
        self.emotion_indicator = QLabel("🟢 情绪检测中")
        self.emotion_indicator.setObjectName("emotionIndicator")
        self.emotion_indicator.setAlignment(Qt.AlignCenter)
        
        emotion_layout.addWidget(self.emotion_icon_label, 3)
        emotion_layout.addWidget(self.emotion_name_label, 1)
        emotion_layout.addWidget(self.emotion_text_label, 2)
        emotion_layout.addWidget(self.emotion_indicator, 1)
        
        self.emotion_card.setLayout(emotion_layout)
        
        # ----- 右侧聊天卡片 -----
        self.chat_card = QFrame()
        self.chat_card.setObjectName("chatCard")
        self.chat_card.setFrameShape(QFrame.StyledPanel)
        
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(18, 18, 18, 18)
        chat_layout.setSpacing(12)
        
        self.chat_title_label = QLabel("聊天窗口")
        self.chat_title_label.setObjectName("chatTitle")
        self.chat_title_label.setAlignment(Qt.AlignLeft)
        
        self.chat_box = QTextEdit()
        self.chat_box.setObjectName("chatBox")
        self.chat_box.setReadOnly(True)
        self.chat_box.setText("AI：你好，我会在这里陪你聊天。")
        self.chat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        chat_layout.addWidget(self.chat_title_label)
        chat_layout.addWidget(self.chat_box)
        
        self.chat_card.setLayout(chat_layout)
        
        center_layout.addWidget(self.emotion_card, 4)
        center_layout.addWidget(self.chat_card, 6)
        
        # ========== 底部按钮栏 ==========
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(14)
        
        # 录音按钮（核心按钮，切换开始/停止）
        self.record_button = QPushButton("🎙 开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setMinimumHeight(76)
        
        # 退出聊天按钮（录音开始后出现）
        self.exit_chat_button = QPushButton("❌ 退出聊天")
        self.exit_chat_button.setObjectName("exitChatButton")
        self.exit_chat_button.setMinimumHeight(76)
        self.exit_chat_button.hide()  # 初始隐藏
        
        # 退出程序按钮
        self.exit_program_button = QPushButton("⏻ 退出程序")
        self.exit_program_button.setObjectName("exitProgramButton")
        self.exit_program_button.setMinimumHeight(76)
        
        bottom_bar.addWidget(self.record_button, 4)
        bottom_bar.addWidget(self.exit_chat_button, 3)
        bottom_bar.addWidget(self.exit_program_button, 3)
        
        # ========== 组装布局 ==========
        root_layout.addLayout(top_bar)
        root_layout.addLayout(center_layout, 1)
        root_layout.addLayout(bottom_bar)
        
        self.setLayout(root_layout)
    
    # ---------------- 信号连接 ----------------
    def _connect_signals(self):
        self.record_button.clicked.connect(self._on_record_button_clicked)
        self.exit_chat_button.clicked.connect(self.exit_chat_clicked.emit)
        self.exit_program_button.clicked.connect(self.exit_program_clicked.emit)
    
    def _on_record_button_clicked(self):
        """录音按钮点击处理（内部逻辑 + 发送信号）"""
        if not self.is_recording:
            # 开始录音
            self.is_recording = True
            self.record_button.setText("🛑 停止录音")
            self.record_button.setObjectName("recordButtonActive")
            self.record_button.setStyleSheet(self._style_sheet())
            
            # 显示退出聊天按钮
            self.exit_chat_button.show()
            
            # 切换状态
            self.set_state_listening()
        else:
            # 停止录音
            self.is_recording = False
            self.record_button.setText("🎙 开始录音")
            self.record_button.setObjectName("recordButton")
            self.record_button.setStyleSheet(self._style_sheet())
            
            # 切换到思考状态（等待处理结果）
            self.set_state_thinking()
        
        # 发送信号给主程序处理
        self.record_button_clicked.emit()
    
    # ---------------- UI更新方法（供外部调用） ----------------
    def set_state_emotion_detecting(self):
        """设置为情绪检测状态"""
        self.current_state = "emotion"
        self.set_status("情绪检测中")
        self.record_button.setEnabled(True)
        self.record_button.setText("🎙 开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = False
        self.exit_chat_button.hide()
        
        self.emotion_indicator.setText("🟢 情绪检测中")
        self.emotion_indicator.setStyleSheet("color: #4CAF50; font-size: 20px;")
    
    def set_state_listening(self):
        """设置为录音状态"""
        self.current_state = "listening"
        self.set_status("正在录音中")
        self.record_button.setEnabled(True)
        self.record_button.setText("🛑 停止录音")
        self.record_button.setObjectName("recordButtonActive")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = True
        self.exit_chat_button.show()
        
        self.emotion_indicator.setText("🔴 录音中...")
        self.emotion_indicator.setStyleSheet("color: #FF5722; font-size: 20px;")
    
    def set_state_thinking(self):
        """设置为AI思考状态"""
        self.current_state = "thinking"
        self.set_status("AI 正在思考")
        self.record_button.setEnabled(False)
        self.record_button.setText("⏳ AI思考中...")
        
        self.emotion_indicator.setText("🟡 AI思考中")
        self.emotion_indicator.setStyleSheet("color: #FFC107; font-size: 20px;")
    
    def set_state_speaking(self):
        """设置为AI说话状态"""
        self.current_state = "speaking"
        self.set_status("AI 正在说话")
        self.record_button.setEnabled(False)
        self.record_button.setText("🔊 AI说话中...")
        
        self.emotion_indicator.setText("🟡 AI说话中")
        self.emotion_indicator.setStyleSheet("color: #FFC107; font-size: 20px;")
    
    def set_state_chatting(self):
        """设置为正常聊天状态（非录音、非思考、非说话）"""
        self.current_state = "chatting"
        self.set_status("聊天中")
        self.record_button.setEnabled(True)
        self.record_button.setText("🎙 开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = False
        
        self.emotion_indicator.setText("🟢 聊天中")
        self.emotion_indicator.setStyleSheet("color: #4CAF50; font-size: 20px;")
    
    def set_state_error(self, message):
        """设置错误状态"""
        self.current_state = "error"
        self.set_status("发生错误")
        self.record_button.setEnabled(True)
        self.record_button.setText("🎙 开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = False
        
        self.emotion_indicator.setText("🔴 发生错误")
        self.emotion_indicator.setStyleSheet("color: #F44336; font-size: 20px;")
        self.append_system_message(f"错误：{message}")
    
    def set_state_emotion_chat(self, emotion, description):
        """设置情绪触发的主动聊天状态"""
        self.current_state = "emotion_chat"
        self.set_status(f"检测到强烈情绪：{emotion}")
        self.record_button.setEnabled(True)
        self.record_button.setText("🎙 开始录音")
        
        self.emotion_indicator.setText(f"🔴 检测到{emotion}情绪")
        self.emotion_indicator.setStyleSheet("color: #FF5722; font-size: 20px;")
        
        # 显示情绪触发的消息
        self.append_ai_message(f"我注意到你现在的情绪是{emotion}，{description}。想和我聊聊吗？")
    
    def set_status(self, text):
        """更新状态标签"""
        self.status_label.setText(f"状态：{text}")
    
    def set_emotion(self, emotion, text=None, fps=None, strong=False):
        """更新情绪显示"""
        icon_map = {
            "Happy": "😊", "Sad": "😢", "Angry": "😠",
            "Surprise": "😮", "Fear": "😨", "Neutral": "😐",
            "Disgust": "🤢", "Contempt": "😒", "--": "🙂"
        }
        
        self.emotion_icon_label.setText(icon_map.get(emotion, "🙂"))
        self.emotion_name_label.setText(f"当前情绪：{emotion}")
        self.emotion_text_label.setText(text if text else "")
        
        # 更新卡片样式
        self.emotion_card.setObjectName("emotionCardStrong" if strong else "emotionCard")
        self.emotion_card.setStyleSheet(self.emotion_card.styleSheet())
    
    def clear_chat(self):
        """清空聊天记录"""
        self.chat_box.setText("AI：你好，我会在这里陪你聊天。")
    
    def append_user_message(self, text):
        """添加用户消息"""
        self.chat_box.append(f"\n你：{text}")
        self._scroll_chat_to_bottom()
    
    def append_ai_message(self, text):
        """添加AI消息"""
        self.chat_box.append(f"\nAI：{text}")
        self._scroll_chat_to_bottom()
    
    def append_system_message(self, text):
        """添加系统消息"""
        self.chat_box.append(f"\n系统：{text}")
        self._scroll_chat_to_bottom()
    
    def append_emotion_message(self, emotion, text):
        """添加情绪检测消息"""
        self.chat_box.append(f"\n🤖 检测到情绪：{emotion} - {text}")
        self._scroll_chat_to_bottom()
    
    def _scroll_chat_to_bottom(self):
        """滚动聊天框到底部"""
        cursor = self.chat_box.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_box.setTextCursor(cursor)
    
    # ---------------- 事件处理 ----------------
    def closeEvent(self, event):
        """关闭事件"""
        self.exit_program_clicked.emit()
        event.accept()
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Escape:
            self.exit_program_clicked.emit()
        else:
            super().keyPressEvent(event)
    
    # ---------------- 样式表 ----------------
    def _style_sheet(self):
        return """
        QWidget { 
            background-color: #101820; 
            color: #F5F7FA; 
            font-family: "Microsoft YaHei"; 
        }
        
        QLabel { 
            color: #F5F7FA; 
        }
        
        #titleLabel { 
            font-size: 32px; 
            font-weight: bold; 
            color: #FFFFFF; 
        }
        
        #statusLabel { 
            font-size: 26px; 
            color: #B8C7D9; 
        }
        
        #emotionCard { 
            background-color: #172331; 
            border-radius: 24px; 
            border: 2px solid #26384A; 
        }
        
        #emotionCardStrong { 
            background-color: #2A1B1B; 
            border-radius: 24px; 
            border: 3px solid #E25D5D; 
        }
        
        #chatCard { 
            background-color: #172331; 
            border-radius: 24px; 
            border: 2px solid #26384A; 
        }
        
        #emotionIcon { 
            font-size: 96px; 
        }
        
        #emotionName { 
            font-size: 34px; 
            font-weight: bold; 
        }
        
        #emotionText { 
            font-size: 26px; 
            color: #DCE6F2; 
        }
        
        #emotionIndicator {
            font-size: 20px;
            font-weight: bold;
        }
        
        #chatTitle { 
            font-size: 28px; 
            font-weight: bold; 
        }
        
        #chatBox { 
            background-color: #0D141C; 
            color: #F5F7FA; 
            border-radius: 18px; 
            border: 1px solid #26384A; 
            padding: 16px; 
            font-size: 26px; 
            line-height: 1.5; 
        }
        
        QPushButton { 
            border: none; 
            border-radius: 20px; 
            font-size: 30px; 
            font-weight: bold; 
            padding: 18px; 
        }
        
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        
        #recordButton { 
            background-color: #2F80ED; 
            color: white; 
        }
        
        #recordButton:hover {
            background-color: #1a6fd9;
        }
        
        #recordButtonActive { 
            background-color: #FF5722; 
            color: white; 
        }
        
        #recordButtonActive:hover {
            background-color: #E64A19;
        }
        
        #exitChatButton { 
            background-color: #FF9800; 
            color: white; 
        }
        
        #exitChatButton:hover {
            background-color: #F57C00;
        }
        
        #exitProgramButton { 
            background-color: #B33A3A; 
            color: white; 
        }
        
        #exitProgramButton:hover {
            background-color: #992e2e;
        }
        """
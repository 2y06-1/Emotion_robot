import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QTextEdit, QSizePolicy, QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QPainterPath,
    QLinearGradient, QRadialGradient, QPixmap, QImage
)


class RobotEyesWidget(QWidget):
    """
    第一个 UI：纯黑背景 + 一组发光机器人眼睛。
    不画蓝色边框、不画面罩、不显示状态文字。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.emotion = "no_face"
        self.strong = False
        self.tick = 0
        self.setMinimumSize(800, 480)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(80)

    def set_emotion(self, emotion: str, strong: bool = False):
        key = (emotion or "no_face").lower()
        if key in ["--", "none", "unknown"]:
            key = "no_face"
        self.emotion = key
        self.strong = strong
        self.update()

    def _animate(self):
        self.tick = (self.tick + 1) % 120
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        # 只保留极弱环境光，让屏幕不死黑，但不出现边框和装饰。
        glow = QRadialGradient(w * 0.5, h * 0.36, min(w, h) * 0.55)
        glow.setColorAt(0.0, QColor(0, 110, 180, 24))
        glow.setColorAt(0.55, QColor(0, 40, 80, 8))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow))

        self._draw_current_expression(painter, w, h)

    def _draw_current_expression(self, painter, w, h):
        # 眼睛整体上移，不放在屏幕正中心。
        cx = w * 0.50
        cy = h * 0.34
        gap = w * 0.145

        pulse = abs(60 - self.tick) / 60.0
        breath = 1.0 + pulse * 0.045

        # 基础眼睛尺寸，比上一版更大、更干净。
        eye_w = min(w * 0.095, 82) * breath
        eye_h = min(h * 0.22, 108) * breath

        # 情绪只改变眼睛形状，不增加其它 UI 元素。
        if self.emotion == "happy":
            self._draw_smile_eye(painter, cx - gap, cy, eye_w * 1.18, eye_h * 0.58)
            self._draw_smile_eye(painter, cx + gap, cy, eye_w * 1.18, eye_h * 0.58)
        elif self.emotion == "sad":
            self._draw_sad_eye(painter, cx - gap, cy + 8, eye_w * 1.08, eye_h * 0.55)
            self._draw_sad_eye(painter, cx + gap, cy + 8, eye_w * 1.08, eye_h * 0.55)
        elif self.emotion == "angry":
            self._draw_bar_eye(painter, cx - gap, cy, eye_w * 1.10, eye_h * 0.34, -14)
            self._draw_bar_eye(painter, cx + gap, cy, eye_w * 1.10, eye_h * 0.34, 14)
        elif self.emotion == "surprise":
            self._draw_oval_eye(painter, cx - gap, cy, eye_w * 1.02, eye_h * 1.18)
            self._draw_oval_eye(painter, cx + gap, cy, eye_w * 1.02, eye_h * 1.18)
        elif self.emotion == "neutral":
            self._draw_oval_eye(painter, cx - gap, cy, eye_w, eye_h * 0.95)
            self._draw_oval_eye(painter, cx + gap, cy, eye_w, eye_h * 0.95)
        else:
            # no_face：待机/扫描感，仍然是一种机器人表情，不写文字。
            self._draw_oval_eye(painter, cx - gap, cy, eye_w * 0.62, eye_h * 0.74, dim=True)
            self._draw_oval_eye(painter, cx + gap, cy, eye_w * 0.62, eye_h * 0.74, dim=True)

    def _eye_colors(self, dim=False):
        if dim:
            return QColor(55, 190, 235), QColor(0, 120, 210), QColor(0, 45, 110)
        if self.strong:
            return QColor(130, 250, 255), QColor(0, 205, 255), QColor(0, 95, 255)
        return QColor(95, 235, 255), QColor(0, 175, 255), QColor(0, 80, 230)

    def _draw_eye_glow(self, painter, x, y, ew, eh, dim=False):
        core, mid, outer = self._eye_colors(dim)
        rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)

        # 大范围柔光。
        for i in range(9, 0, -1):
            grow_x = i * ew * 0.11
            grow_y = i * eh * 0.10
            alpha = 7 + i * (3 if dim else 5)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(outer.red(), outer.green(), outer.blue(), alpha))
            painter.drawRoundedRect(
                rect.adjusted(-grow_x, -grow_y, grow_x, grow_y),
                ew, ew
            )

        # 主体渐变，让眼睛更像 LED/机器人屏幕光。
        grad = QRadialGradient(x, y - eh * 0.18, max(ew, eh) * 0.75)
        grad.setColorAt(0.0, QColor(core.red(), core.green(), core.blue(), 255 if not dim else 210))
        grad.setColorAt(0.45, QColor(mid.red(), mid.green(), mid.blue(), 230 if not dim else 170))
        grad.setColorAt(1.0, QColor(outer.red(), outer.green(), outer.blue(), 45 if not dim else 24))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, ew / 2, ew / 2)

        # 小高光。
        highlight = QRadialGradient(x - ew * 0.18, y - eh * 0.22, ew * 0.28)
        highlight.setColorAt(0.0, QColor(220, 255, 255, 115 if not dim else 60))
        highlight.setColorAt(1.0, QColor(220, 255, 255, 0))
        painter.setBrush(QBrush(highlight))
        painter.drawEllipse(QRectF(x - ew * 0.34, y - eh * 0.38, ew * 0.34, eh * 0.28))

    def _draw_oval_eye(self, painter, x, y, ew, eh, dim=False):
        self._draw_eye_glow(painter, x, y, ew, eh, dim)

    def _draw_bar_eye(self, painter, x, y, ew, eh, rotate=0):
        painter.save()
        painter.translate(x, y)
        painter.rotate(rotate)
        self._draw_eye_glow(painter, 0, 0, ew, eh, False)
        painter.restore()

    def _draw_smile_eye(self, painter, x, y, ew, eh):
        core, mid, outer = self._eye_colors(False)
        painter.save()
        # 弧线外发光。
        for width, alpha in [(20, 22), (14, 38), (9, 220)]:
            pen = QPen(QColor(core.red(), core.green(), core.blue(), alpha), width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)
            painter.drawArc(rect, 205 * 16, 130 * 16)
        painter.restore()

    def _draw_sad_eye(self, painter, x, y, ew, eh):
        core, mid, outer = self._eye_colors(False)
        painter.save()
        for width, alpha in [(20, 20), (14, 36), (9, 215)]:
            pen = QPen(QColor(core.red(), core.green(), core.blue(), alpha), width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)
            painter.drawArc(rect, 25 * 16, 130 * 16)
        painter.restore()


class MainWindow(QWidget):
    """
    UI 结构：
    1. 机器人动态表情页：全黑背景，只显示一组发光眼睛，右下角小字。
    2. 聊天页：录音按钮、结束聊天、退出程序、聊天窗口。
    3. 用户表情页：显示采集到的人脸窗口和用户情绪图标。
    """

    record_button_clicked = pyqtSignal()
    exit_chat_clicked = pyqtSignal()
    exit_program_clicked = pyqtSignal()
    page_changed = pyqtSignal(str)  # robot / chat / face

    def __init__(self):
        super().__init__()
        self.current_state = "emotion"
        self.current_page = "robot"
        self.is_recording = False

        self.setWindowTitle("Emotion Robot")
        self.setMinimumSize(800, 480)
        self._build_ui()
        self._connect_signals()

        self.set_state_emotion_detecting()
        self.set_emotion("no_face", "", False)
        self.show_robot_ui()

    def _build_ui(self):
        self.setStyleSheet(self._style_sheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        self.robot_page = self._build_robot_page()
        self.chat_page = self._build_chat_page()
        self.face_page = self._build_face_page()
        self.stack.addWidget(self.robot_page)
        self.stack.addWidget(self.chat_page)
        self.stack.addWidget(self.face_page)
        root.addWidget(self.stack)

    def _build_robot_page(self):
        page = QWidget()
        page.setObjectName("robotPage")

        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.eyes_widget = RobotEyesWidget()
        root.addWidget(self.eyes_widget, 1)

        self.enter_hint_label = QLabel("双击进入聊天界面", page)
        self.enter_hint_label.setObjectName("enterHintLabel")
        self.enter_hint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.enter_hint_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        return page

    def _build_chat_page(self):
        page = QWidget()
        page.setObjectName("chatPage")

        root_layout = QVBoxLayout(page)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("Emotion Robot Chat")
        self.title_label.setObjectName("titleLabel")
        self.status_label = QLabel("状态：准备聊天")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_bar.addWidget(self.title_label, 1)
        top_bar.addWidget(self.status_label, 2)

        self.chat_card = QFrame()
        self.chat_card.setObjectName("chatCard")
        chat_layout = QVBoxLayout(self.chat_card)
        chat_layout.setContentsMargins(18, 18, 18, 18)
        chat_layout.setSpacing(12)

        self.chat_title_label = QLabel("聊天窗口")
        self.chat_title_label.setObjectName("chatTitle")
        self.chat_box = QTextEdit()
        self.chat_box.setObjectName("chatBox")
        self.chat_box.setReadOnly(True)
        self.chat_box.setPlaceholderText("对话记录会显示在这里")
        self.chat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        chat_layout.addWidget(self.chat_title_label)
        chat_layout.addWidget(self.chat_box, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(14)

        self.record_button = QPushButton("开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setMinimumHeight(76)

        self.exit_chat_button = QPushButton("结束聊天")
        self.exit_chat_button.setObjectName("exitChatButton")
        self.exit_chat_button.setMinimumHeight(76)
        self.exit_chat_button.hide()

        self.exit_program_button = QPushButton("退出程序")
        self.exit_program_button.setObjectName("exitProgramButton")
        self.exit_program_button.setMinimumHeight(76)

        bottom_bar.addWidget(self.record_button, 4)
        bottom_bar.addWidget(self.exit_chat_button, 3)
        bottom_bar.addWidget(self.exit_program_button, 3)

        root_layout.addLayout(top_bar)
        root_layout.addWidget(self.chat_card, 1)
        root_layout.addLayout(bottom_bar)
        return page

    def _build_face_page(self):
        page = QWidget()
        page.setObjectName("facePage")

        root_layout = QVBoxLayout(page)
        root_layout.setContentsMargins(26, 24, 26, 22)
        root_layout.setSpacing(16)

        top_bar = QHBoxLayout()
        self.face_title_label = QLabel("用户表情")
        self.face_title_label.setObjectName("faceTitleLabel")
        self.face_status_label = QLabel("双击返回聊天界面")
        self.face_status_label.setObjectName("faceStatusLabel")
        self.face_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_bar.addWidget(self.face_title_label, 1)
        top_bar.addWidget(self.face_status_label, 1)

        content = QHBoxLayout()
        content.setSpacing(20)

        self.face_preview_card = QFrame()
        self.face_preview_card.setObjectName("facePreviewCard")
        preview_layout = QVBoxLayout(self.face_preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(8)

        self.face_image_label = QLabel("等待画面")
        self.face_image_label.setObjectName("faceImageLabel")
        self.face_image_label.setAlignment(Qt.AlignCenter)
        self.face_image_label.setMinimumSize(500, 320)
        self.face_image_label.setScaledContents(False)
        preview_layout.addWidget(self.face_image_label, 1)

        self.face_emotion_card = QFrame()
        self.face_emotion_card.setObjectName("faceEmotionCard")
        emotion_layout = QVBoxLayout(self.face_emotion_card)
        emotion_layout.setContentsMargins(18, 18, 18, 18)
        emotion_layout.setSpacing(14)

        self.user_emotion_icon_label = QLabel("◌")
        self.user_emotion_icon_label.setObjectName("userEmotionIconLabel")
        self.user_emotion_icon_label.setAlignment(Qt.AlignCenter)

        self.user_emotion_name_label = QLabel("未检测")
        self.user_emotion_name_label.setObjectName("userEmotionNameLabel")
        self.user_emotion_name_label.setAlignment(Qt.AlignCenter)

        self.user_emotion_prob_label = QLabel("")
        self.user_emotion_prob_label.setObjectName("userEmotionProbLabel")
        self.user_emotion_prob_label.setAlignment(Qt.AlignCenter)

        emotion_layout.addStretch(1)
        emotion_layout.addWidget(self.user_emotion_icon_label)
        emotion_layout.addWidget(self.user_emotion_name_label)
        emotion_layout.addWidget(self.user_emotion_prob_label)
        emotion_layout.addStretch(1)

        content.addWidget(self.face_preview_card, 7)
        content.addWidget(self.face_emotion_card, 3)

        bottom_hint = QLabel("双击返回聊天界面")
        bottom_hint.setObjectName("faceBottomHintLabel")
        bottom_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        root_layout.addLayout(top_bar)
        root_layout.addLayout(content, 1)
        root_layout.addWidget(bottom_hint)
        return page

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "enter_hint_label"):
            hint_w = 260
            hint_h = 34
            margin_right = 28
            margin_bottom = 20
            self.enter_hint_label.setGeometry(
                self.width() - hint_w - margin_right,
                self.height() - hint_h - margin_bottom,
                hint_w,
                hint_h,
            )

    def _connect_signals(self):
        self.record_button.clicked.connect(self._on_record_button_clicked)
        self.exit_chat_button.clicked.connect(self.exit_chat_clicked.emit)
        self.exit_program_button.clicked.connect(self.exit_program_clicked.emit)

    def mouseDoubleClickEvent(self, event):
        if self.current_page == "robot":
            self.show_chat_ui()
            event.accept()
            return
        if self.current_page == "chat":
            self.show_face_ui()
            event.accept()
            return
        if self.current_page == "face":
            self.show_chat_ui()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _on_record_button_clicked(self):
        if not self.is_recording:
            self.is_recording = True
            self.exit_chat_button.show()
            self.set_state_listening()
        else:
            self.is_recording = False
            self.set_state_thinking()
        self.record_button_clicked.emit()

    def show_robot_ui(self):
        self.current_page = "robot"
        self.stack.setCurrentWidget(self.robot_page)
        self.page_changed.emit("robot")

    def show_chat_ui(self):
        self.current_page = "chat"
        self.stack.setCurrentWidget(self.chat_page)
        if self.current_state == "emotion":
            self._reset_record_button()
            self.exit_chat_button.hide()
            self.set_status("准备聊天")
        self.page_changed.emit("chat")

    def show_face_ui(self):
        self.current_page = "face"
        self.stack.setCurrentWidget(self.face_page)
        self.page_changed.emit("face")

    def set_state_emotion_detecting(self):
        self.current_state = "emotion"
        self.set_status("准备聊天")
        self._reset_record_button()
        self.exit_chat_button.hide()

    def set_state_listening(self):
        self.current_state = "listening"
        self.set_status("正在录音中")
        self.record_button.setEnabled(True)
        self.record_button.setText("结束录音")
        self.record_button.setObjectName("recordButtonActive")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = True
        self.exit_chat_button.show()

    def set_state_thinking(self):
        self.current_state = "thinking"
        self.set_status("AI 正在思考")
        self.record_button.setEnabled(False)
        self.record_button.setText("AI思考中...")
        self.exit_chat_button.show()

    def set_state_speaking(self):
        self.current_state = "speaking"
        self.set_status("AI 正在说话")
        self.record_button.setEnabled(False)
        self.record_button.setText("AI说话中...")
        self.exit_chat_button.show()

    def set_state_chatting(self):
        self.current_state = "chatting"
        self.set_status("聊天中")
        self._reset_record_button()
        self.exit_chat_button.show()

    def set_state_error(self, message):
        self.current_state = "error"
        self.set_status("发生错误")
        self._reset_record_button()
        self.exit_chat_button.show()
        self.append_system_message(f"错误：{message}")

    def _reset_record_button(self):
        self.record_button.setEnabled(True)
        self.record_button.setText("开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = False

    def set_status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.setText(f"状态：{text}")

    def set_emotion(self, emotion, text=None, strong=False):
        # 第一个 UI 不显示任何状态文字，只通过眼睛变化表达状态。
        key = (emotion or "no_face").lower()
        if key in ["--", "none", "unknown"]:
            key = "no_face"
        self.eyes_widget.set_emotion(key, strong)

    def update_user_face(self, frame, emotion="no_face", prob=0.0):
        """第三个 UI 更新：frame 为 OpenCV BGR 图像。"""
        key = (emotion or "no_face").lower()
        icon_map = {
            "angry": "▰▰",
            "happy": "⌒  ⌒",
            "sad": "╯  ╰",
            "neutral": "●  ●",
            "surprise": "○  ○",
            "no_face": "◌"
        }
        name_map = {
            "angry": "生气",
            "happy": "开心",
            "sad": "难过",
            "neutral": "平稳",
            "surprise": "惊讶",
            "no_face": "未检测"
        }
        if key in ["--", "none", "unknown"]:
            key = "no_face"

        self.user_emotion_icon_label.setText(icon_map.get(key, "◌"))
        self.user_emotion_name_label.setText(name_map.get(key, key))
        self.user_emotion_prob_label.setText("" if key == "no_face" else f"置信度 {prob:.2f}")

        if frame is None:
            self.face_image_label.setText("暂无画面")
            self.face_image_label.setPixmap(QPixmap())
            return

        try:
            rgb = frame[:, :, ::-1].copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target_size = self.face_image_label.size()
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.face_image_label.setText("")
            self.face_image_label.setPixmap(pix)
        except Exception as e:
            self.face_image_label.setText(f"画面显示失败：{e}")
            self.face_image_label.setPixmap(QPixmap())

    def clear_chat(self):
        self.chat_box.clear()

    def append_user_message(self, text):
        self.chat_box.append(f"\n你：{text}")
        self._scroll_chat_to_bottom()

    def append_ai_message(self, text):
        self.chat_box.append(f"\nAI：{text}")
        self._scroll_chat_to_bottom()

    def append_system_message(self, text):
        self.chat_box.append(f"\n系统：{text}")
        self._scroll_chat_to_bottom()

    def append_emotion_message(self, emotion, text):
        self.chat_box.append(f"\n系统：检测到情绪 {emotion} {text}")
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self):
        cursor = self.chat_box.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_box.setTextCursor(cursor)

    def closeEvent(self, event):
        self.exit_program_clicked.emit()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.exit_program_clicked.emit()
        else:
            super().keyPressEvent(event)

    def _style_sheet(self):
        return """
        QWidget {
            background-color: #000000;
            color: #F5F7FA;
            font-family: "Microsoft YaHei";
        }
        #robotPage {
            background-color: #000000;
        }
        #enterHintLabel {
            color: rgba(145, 225, 255, 135);
            font-size: 16px;
            letter-spacing: 1px;
            background: transparent;
        }
        #chatPage {
            background-color: #0B1118;
        }
        #titleLabel {
            font-size: 30px;
            font-weight: bold;
            color: #FFFFFF;
        }
        #statusLabel {
            font-size: 24px;
            color: #B8C7D9;
        }
        #chatCard {
            background-color: #111B26;
            border-radius: 24px;
            border: 1px solid #26384A;
        }
        #chatTitle {
            font-size: 26px;
            font-weight: bold;
            color: #FFFFFF;
        }
        #chatBox {
            background-color: #071018;
            color: #F5F7FA;
            border-radius: 18px;
            border: 1px solid #24394D;
            padding: 16px;
            font-size: 25px;
            line-height: 1.5;
        }
        #facePage {
            background-color: #05080D;
        }
        #faceTitleLabel {
            font-size: 30px;
            font-weight: bold;
            color: #FFFFFF;
        }
        #faceStatusLabel, #faceBottomHintLabel {
            font-size: 16px;
            color: rgba(145, 225, 255, 140);
        }
        #facePreviewCard, #faceEmotionCard {
            background-color: #0B1118;
            border-radius: 26px;
            border: 1px solid #24394D;
        }
        #faceImageLabel {
            background-color: #000000;
            border-radius: 20px;
            color: #53606B;
            font-size: 24px;
        }
        #userEmotionIconLabel {
            font-size: 76px;
            font-weight: bold;
            color: #67E8F9;
            letter-spacing: 6px;
        }
        #userEmotionNameLabel {
            font-size: 32px;
            font-weight: bold;
            color: #F5F7FA;
        }
        #userEmotionProbLabel {
            font-size: 20px;
            color: #9FB7C9;
        }
        QPushButton {
            border: none;
            border-radius: 20px;
            font-size: 28px;
            font-weight: bold;
            padding: 18px;
        }
        QPushButton:disabled {
            background-color: #4A5562;
            color: #9AA3AD;
        }
        #recordButton {
            background-color: #1E88E5;
            color: white;
        }
        #recordButtonActive {
            background-color: #F05A28;
            color: white;
        }
        #exitChatButton {
            background-color: #F59E0B;
            color: white;
        }
        #exitProgramButton {
            background-color: #B33A3A;
            color: white;
        }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec_())

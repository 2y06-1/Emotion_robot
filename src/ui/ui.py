import sys
import html

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QStackedWidget,
    QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QRadialGradient,
    QPixmap,
    QImage,
)


class RobotEyesWidget(QWidget):
    """
    第一个 UI：纯黑背景 + 一组发光机器人眼睛。
    这个页面保持极简，只通过眼睛表达机器人状态。
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

        glow = QRadialGradient(w * 0.5, h * 0.36, min(w, h) * 0.58)
        glow.setColorAt(0.0, QColor(0, 135, 210, 28))
        glow.setColorAt(0.58, QColor(0, 45, 95, 9))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow))

        self._draw_current_expression(painter, w, h)

    def _draw_current_expression(self, painter, w, h):
        cx = w * 0.50
        cy = h * 0.34
        gap = w * 0.145

        pulse = abs(60 - self.tick) / 60.0
        breath = 1.0 + pulse * 0.045

        eye_w = min(w * 0.095, 82) * breath
        eye_h = min(h * 0.22, 108) * breath

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
            self._draw_oval_eye(painter, cx - gap, cy, eye_w * 0.62, eye_h * 0.74, dim=True)
            self._draw_oval_eye(painter, cx + gap, cy, eye_w * 0.62, eye_h * 0.74, dim=True)

    def _eye_colors(self, dim=False):
        if dim:
            return QColor(55, 190, 235), QColor(0, 120, 210), QColor(0, 45, 110)
        if self.strong:
            return QColor(145, 255, 255), QColor(0, 220, 255), QColor(0, 100, 255)
        return QColor(100, 240, 255), QColor(0, 178, 255), QColor(0, 82, 230)

    def _draw_eye_glow(self, painter, x, y, ew, eh, dim=False):
        core, mid, outer = self._eye_colors(dim)
        rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)

        for i in range(9, 0, -1):
            grow_x = i * ew * 0.11
            grow_y = i * eh * 0.10
            alpha = 7 + i * (3 if dim else 5)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(outer.red(), outer.green(), outer.blue(), alpha))
            painter.drawRoundedRect(rect.adjusted(-grow_x, -grow_y, grow_x, grow_y), ew, ew)

        grad = QRadialGradient(x, y - eh * 0.18, max(ew, eh) * 0.75)
        grad.setColorAt(0.0, QColor(core.red(), core.green(), core.blue(), 255 if not dim else 210))
        grad.setColorAt(0.45, QColor(mid.red(), mid.green(), mid.blue(), 230 if not dim else 170))
        grad.setColorAt(1.0, QColor(outer.red(), outer.green(), outer.blue(), 45 if not dim else 24))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, ew / 2, ew / 2)

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
        core, _, _ = self._eye_colors(False)
        painter.save()
        rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)
        for width, alpha in [(20, 22), (14, 38), (9, 220)]:
            painter.setPen(QPen(QColor(core.red(), core.green(), core.blue(), alpha), width, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(rect, 25 * 16, 130 * 16)
        painter.restore()

    def _draw_sad_eye(self, painter, x, y, ew, eh):
        core, _, _ = self._eye_colors(False)
        painter.save()
        rect = QRectF(x - ew / 2, y - eh / 2, ew, eh)
        for width, alpha in [(20, 20), (14, 36), (9, 215)]:
            painter.setPen(QPen(QColor(core.red(), core.green(), core.blue(), alpha), width, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(rect, 205 * 16, 130 * 16)
        painter.restore()


class SoftGlowPage(QWidget):
    """黑色柔光背景页，用于聊天页和表情页。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()
        painter.fillRect(self.rect(), QColor(3, 7, 12))

        glow1 = QRadialGradient(w * 0.20, h * 0.10, min(w, h) * 0.70)
        glow1.setColorAt(0.0, QColor(0, 115, 255, 45))
        glow1.setColorAt(0.55, QColor(0, 60, 150, 13))
        glow1.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow1))

        glow2 = QRadialGradient(w * 0.86, h * 0.76, min(w, h) * 0.58)
        glow2.setColorAt(0.0, QColor(0, 210, 255, 28))
        glow2.setColorAt(0.60, QColor(0, 80, 120, 10))
        glow2.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow2))

        super().paintEvent(event)


class ChatBubble(QFrame):
    """聊天气泡。"""

    def __init__(self, text, role="robot", parent=None):
        super().__init__(parent)
        self.role = role
        self.setObjectName({
            "user": "userBubble",
            "robot": "robotBubble",
            "system": "systemBubble",
        }.get(role, "robotBubble"))

        layout = QVBoxLayout(self)
        if role == "system":
            layout.setContentsMargins(12, 7, 12, 7)
        else:
            layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        if role == "robot":
            name = QLabel("机器人")
            name.setObjectName("bubbleName")
            layout.addWidget(name)

        # 这里只做普通文本显示。
        # 原来使用 html.escape(text)，会把英文双引号转成 &quot;，
        # QLabel 按普通文本显示时就会直接看到 &quot;。
        display_text = html.unescape(str(text or ""))
        label = QLabel(display_text)
        label.setTextFormat(Qt.PlainText)
        label.setObjectName("bubbleText")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(label)


class MainWindow(QWidget):
    """
    三个界面：
    1. 机器人动态表情页
    2. 极简聊天页
    3. 用户表情检测页
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
        self._message_count = 0

        self.setWindowTitle("Emotion Robot")
        self.setMinimumSize(800, 480)
        self._build_ui()
        self._connect_signals()

        self.set_state_emotion_detecting()
        self.set_emotion("no_face", "", False)
        self.show_robot_ui()

    # ---------- UI 构建 ----------
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

        self.enter_hint_label = QLabel("双击进入聊天", page)
        self.enter_hint_label.setObjectName("enterHintLabel")
        self.enter_hint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.enter_hint_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        return page

    def _build_chat_page(self):
        page = SoftGlowPage()
        page.setObjectName("chatPage")

        root = QVBoxLayout(page)
        root.setContentsMargins(26, 20, 26, 22)
        root.setSpacing(12)

        # 顶部：只保留标题、状态和一个弱提示，不再像软件后台。
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = QLabel("和我聊聊")
        self.title_label.setObjectName("titleLabel")
        self.subtitle_label = QLabel("我会认真听你说")
        self.subtitle_label.setObjectName("subtitleLabel")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)

        self.status_label = QLabel("可以继续说")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addLayout(title_box, 1)
        top_bar.addWidget(self.status_label, 1)

        # 中间：无明显外框的对话区域，只显示气泡。
        self.chat_panel = QFrame()
        self.chat_panel.setObjectName("chatPanel")
        panel_layout = QVBoxLayout(self.chat_panel)
        panel_layout.setContentsMargins(10, 4, 10, 4)
        panel_layout.setSpacing(0)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.chat_content = QWidget()
        self.chat_content.setObjectName("chatContent")
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.setContentsMargins(4, 4, 4, 4)
        self.chat_layout.setSpacing(10)

        self.empty_hint_label = QLabel("我在这里，按下开始说话")
        self.empty_hint_label.setObjectName("emptyHintLabel")
        self.empty_hint_label.setAlignment(Qt.AlignCenter)
        self.empty_hint_label.setWordWrap(True)
        self.chat_layout.addStretch(1)
        self.chat_layout.addWidget(self.empty_hint_label)
        self.chat_layout.addStretch(1)

        self.chat_scroll.setWidget(self.chat_content)
        panel_layout.addWidget(self.chat_scroll, 1)

        # 底部：只保留主操作按钮和弱化的结束聊天按钮。
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(14)

        self.record_button = QPushButton("开始说话")
        self.record_button.setObjectName("recordButton")
        self.record_button.setMinimumHeight(74)
        self.record_button.setCursor(Qt.PointingHandCursor)

        self.exit_chat_button = QPushButton("结束聊天")
        self.exit_chat_button.setObjectName("exitChatButton")
        self.exit_chat_button.setMinimumHeight(74)
        self.exit_chat_button.setCursor(Qt.PointingHandCursor)
        self.exit_chat_button.hide()

        bottom_bar.addWidget(self.record_button, 5)
        bottom_bar.addWidget(self.exit_chat_button, 2)

        root.addLayout(top_bar)
        root.addWidget(self.chat_panel, 1)
        root.addLayout(bottom_bar)
        return page

    def _build_face_page(self):
        page = SoftGlowPage()
        page.setObjectName("facePage")

        root = QVBoxLayout(page)
        root.setContentsMargins(24, 18, 24, 20)
        root.setSpacing(12)

        # 顶部：只保留页面名称和返回提示，避免像调试后台。
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        self.face_title_label = QLabel("表情观察")
        self.face_title_label.setObjectName("faceTitleLabel")

        self.face_status_label = QLabel("双击返回")
        self.face_status_label.setObjectName("faceStatusLabel")
        self.face_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addWidget(self.face_title_label, 1)
        top_bar.addWidget(self.face_status_label, 1)

        # 中间：摄像头画面占主要空间，不再右侧放大卡片。
        self.face_preview_card = QFrame()
        self.face_preview_card.setObjectName("facePreviewCard")
        preview_layout = QVBoxLayout(self.face_preview_card)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(0)

        self.face_image_label = QLabel("等待画面")
        self.face_image_label.setObjectName("faceImageLabel")
        self.face_image_label.setAlignment(Qt.AlignCenter)
        self.face_image_label.setMinimumSize(680, 300)
        self.face_image_label.setScaledContents(False)
        preview_layout.addWidget(self.face_image_label, 1)

        # 底部：情绪结果条。比右侧大卡片更省空间，也更像产品界面。
        self.face_result_bar = QFrame()
        self.face_result_bar.setObjectName("faceResultBar")
        result_layout = QHBoxLayout(self.face_result_bar)
        result_layout.setContentsMargins(22, 8, 22, 8)
        result_layout.setSpacing(14)

        self.user_emotion_icon_label = QLabel("◌")
        self.user_emotion_icon_label.setObjectName("userEmotionIconLabel")
        self.user_emotion_icon_label.setAlignment(Qt.AlignCenter)
        self.user_emotion_icon_label.setFixedWidth(96)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        self.user_emotion_name_label = QLabel("未检测到人脸")
        self.user_emotion_name_label.setObjectName("userEmotionNameLabel")
        self.user_emotion_name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.user_emotion_prob_label = QLabel("请正对摄像头")
        self.user_emotion_prob_label.setObjectName("userEmotionProbLabel")
        self.user_emotion_prob_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        text_box.addWidget(self.user_emotion_name_label)
        text_box.addWidget(self.user_emotion_prob_label)

        self.face_tip_label = QLabel("实时识别中")
        self.face_tip_label.setObjectName("faceTipLabel")
        self.face_tip_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        result_layout.addWidget(self.user_emotion_icon_label, 0)
        result_layout.addLayout(text_box, 1)
        result_layout.addWidget(self.face_tip_label, 1)

        root.addLayout(top_bar)
        root.addWidget(self.face_preview_card, 1)
        root.addWidget(self.face_result_bar, 0)
        return page

    # ---------- 信号连接 ----------
    def _connect_signals(self):
        self.record_button.clicked.connect(self._on_record_button_clicked)
        self.exit_chat_button.clicked.connect(self.exit_chat_clicked.emit)

    # ---------- 页面切换 ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "enter_hint_label"):
            hint_w = 210
            hint_h = 34
            margin_right = 28
            margin_bottom = 20
            self.enter_hint_label.setGeometry(
                self.width() - hint_w - margin_right,
                self.height() - hint_h - margin_bottom,
                hint_w,
                hint_h,
            )

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

    def show_robot_ui(self):
        self.current_page = "robot"
        self.stack.setCurrentWidget(self.robot_page)
        self.page_changed.emit("robot")

    def show_chat_ui(self):
        self.current_page = "chat"
        self.stack.setCurrentWidget(
            self.chat_page
        )

        if self.current_state == "emotion":
            self._reset_record_button()
            self.exit_chat_button.hide()
            self.set_status("可以继续说")

        self.page_changed.emit("chat")

        # 每次进入聊天页面时显示最新消息。
        QTimer.singleShot(
            0,
            self._scroll_chat_to_bottom,
        )

    def show_face_ui(self):
        self.current_page = "face"
        self.stack.setCurrentWidget(self.face_page)
        self.page_changed.emit("face")

    # ---------- 按钮与状态 ----------
    def _on_record_button_clicked(self):
        if not self.is_recording:
            self.is_recording = True
            self.exit_chat_button.show()
            self.set_state_listening()
        else:
            self.is_recording = False
            self.set_state_thinking()
        self.record_button_clicked.emit()

    def set_state_emotion_detecting(self):
        self.current_state = "emotion"
        self.set_status("可以继续说")
        self._reset_record_button()
        self.exit_chat_button.hide()

    def set_state_listening(self):
        self.current_state = "listening"
        self.set_status("正在聆听")
        self.record_button.setEnabled(True)
        self.record_button.setText("结束录音")
        self.record_button.setObjectName("recordButtonActive")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = True
        self.exit_chat_button.show()

    def set_state_thinking(self):
        self.current_state = "thinking"
        self.set_status("正在思考")
        self.record_button.setEnabled(False)
        self.record_button.setText("思考中...")
        self.exit_chat_button.show()

    def set_state_speaking(self):
        self.current_state = "speaking"
        self.set_status("正在回应")
        self.record_button.setEnabled(False)
        self.record_button.setText("回应中...")
        self.exit_chat_button.show()

    def set_state_chatting(self):
        self.current_state = "chatting"
        self.set_status("可以继续说")
        self._reset_record_button()
        self.exit_chat_button.show()

    def set_state_error(self, message):
        self.current_state = "error"
        self.set_status("出现错误")
        self._reset_record_button()
        self.exit_chat_button.show()
        self.append_system_message(f"错误：{message}")

    def _reset_record_button(self):
        self.record_button.setEnabled(True)
        self.record_button.setText("开始说话")
        self.record_button.setObjectName("recordButton")
        self.record_button.setStyleSheet(self._style_sheet())
        self.is_recording = False

    def set_status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.setText(text)

    # ---------- 情绪显示 ----------
    def set_emotion(self, emotion, text=None, strong=False):
        key = (emotion or "no_face").lower()
        if key in ["--", "none", "unknown"]:
            key = "no_face"
        self.eyes_widget.set_emotion(key, strong)

    def update_user_face(self, frame, emotion="no_face", prob=0.0):
        """第三个 UI 更新：frame 为 OpenCV BGR 图像。"""
        key = (emotion or "no_face").lower()
        if key in ["--", "none", "unknown"]:
            key = "no_face"

        icon_map = {
            "angry": "▰▰",
            "happy": "⌒  ⌒",
            "sad": "╯  ╰",
            "neutral": "●  ●",
            "surprise": "○  ○",
            "fear": "◇  ◇",
            "disgust": "—  —",
            "no_face": "◌",
        }
        name_map = {
            "angry": "生气",
            "happy": "开心",
            "sad": "难过",
            "neutral": "平静",
            "surprise": "惊讶",
            "fear": "害怕",
            "disgust": "厌恶",
            "no_face": "未检测",
        }

        self.user_emotion_icon_label.setText(icon_map.get(key, "◌"))
        if key == "no_face":
            self.user_emotion_name_label.setText("未检测到人脸")
            self.user_emotion_prob_label.setText("请正对摄像头")
            if hasattr(self, "face_tip_label"):
                self.face_tip_label.setText("等待人脸")
        else:
            self.user_emotion_name_label.setText(name_map.get(key, key))
            self.user_emotion_prob_label.setText(f"置信度 {prob:.2f}")
            if hasattr(self, "face_tip_label"):
                self.face_tip_label.setText("实时识别中")

        if frame is None:
            self.face_image_label.setText("暂无画面")
            self.face_image_label.setPixmap(QPixmap())
            return

        try:
            rgb = frame[:, :, ::-1].copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(self.face_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.face_image_label.setText("")
            self.face_image_label.setPixmap(pix)
        except Exception as e:
            self.face_image_label.setText(f"画面显示失败：{e}")
            self.face_image_label.setPixmap(QPixmap())

    # ---------- 聊天消息 ----------
    def clear_chat(self):
        """
        清空聊天内容。
        """
        self._message_count = 0

        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue

            if widget is self.empty_hint_label:
                widget.hide()
                widget.setParent(self.chat_content)
            else:
                widget.deleteLater()

        self.chat_layout.addStretch(1)
        self._ensure_empty_hint_label()
        self.empty_hint_label.show()
        self.chat_layout.addWidget(self.empty_hint_label)
        self.chat_layout.addStretch(1)
        self._scroll_chat_to_bottom()

    def _ensure_empty_hint_label(self):
        """确保空聊天提示 QLabel 可用。"""
        try:
            self.empty_hint_label.setText("我在这里，按下开始说话")
            self.empty_hint_label.setParent(self.chat_content)
        except RuntimeError:
            self.empty_hint_label = QLabel("我在这里，按下开始说话")
            self.empty_hint_label.setObjectName("emptyHintLabel")
            self.empty_hint_label.setAlignment(Qt.AlignCenter)
            self.empty_hint_label.setWordWrap(True)
            self.empty_hint_label.setParent(self.chat_content)

    def append_user_message(self, text):
        self._append_message(text, role="user")

    def append_ai_message(self, text):
        self._append_message(text, role="robot")

    def append_system_message(self, text):
        self._append_message(text, role="system")

    def append_emotion_message(self, emotion, text):
        self._append_message(f"检测到情绪 {emotion} {text}", role="system")

    def _append_message(self, text, role="robot"):
        if not hasattr(self, "chat_layout"):
            return

        if self._message_count == 0:
            self._remove_empty_hint()
        self._message_count += 1

        row = QWidget()
        row.setObjectName("messageRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        bubble = ChatBubble(text, role=role)
        bubble.setMaximumWidth(520 if role != "system" else 440)

        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignRight)
        elif role == "system":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignCenter)
            row_layout.addStretch(1)
        else:
            row_layout.addWidget(bubble, 0, Qt.AlignLeft)
            row_layout.addStretch(1)

        insert_index = max(
            0,
            self.chat_layout.count() - 1,
        )

        self.chat_layout.insertWidget(
            insert_index,
            row,
        )

        # 将本次新增的消息传入，自动定位到最新气泡。
        self._scroll_chat_to_bottom(row)

    def _remove_empty_hint(self):
        """移除空提示，但不要销毁它。"""
        self._ensure_empty_hint_label()
        self.empty_hint_label.hide()

        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget is self.empty_hint_label:
                widget.hide()
                widget.setParent(self.chat_content)
            else:
                widget.deleteLater()

        self.chat_layout.addStretch(1)

    def _scroll_chat_to_bottom(
        self,
        target_widget=None,
    ):
        """
        将聊天区域自动滚动到最新消息。

        第一次延迟等待控件插入布局；
        第二次延迟等待 QLabel 完成自动换行和高度计算。
        """

        def do_scroll():
            if not hasattr(
                self,
                "chat_scroll",
            ):
                return

            # 强制刷新聊天布局及内容高度。
            self.chat_layout.activate()
            self.chat_content.updateGeometry()

            # 优先确保本次新消息进入可见区域。
            if target_widget is not None:
                self.chat_scroll.ensureWidgetVisible(
                    target_widget,
                    0,
                    16,
                )

            # 最终强制移动到滚动条底部。
            scroll_bar = (
                self.chat_scroll
                .verticalScrollBar()
            )
            scroll_bar.setValue(
                scroll_bar.maximum()
            )

        # 等待新控件加入布局。
        QTimer.singleShot(
            0,
            do_scroll,
        )

        # 等待文字自动换行和气泡高度计算完成。
        QTimer.singleShot(
            60,
            do_scroll,
        )

        # 板端性能较慢时再校正一次。
        QTimer.singleShot(
            150,
            do_scroll,
        )

    # ---------- 退出逻辑 ----------
    def closeEvent(self, event):
        self.exit_program_clicked.emit()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.exit_program_clicked.emit()
        else:
            super().keyPressEvent(event)

    # ---------- 样式 ----------
    def _style_sheet(self):
        return """
        QWidget {
            background: transparent;
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

        #chatPage, #facePage {
            background: transparent;
        }

        #titleLabel, #faceTitleLabel {
            font-size: 34px;
            font-weight: 800;
            color: #FFFFFF;
            background: transparent;
        }

        #subtitleLabel {
            font-size: 16px;
            color: rgba(180, 220, 255, 155);
            background: transparent;
        }

        #statusLabel {
            font-size: 22px;
            font-weight: 700;
            color: #68E1FF;
            background: transparent;
        }

        #chatPanel {
            background-color: rgba(5, 12, 20, 120);
            border-radius: 28px;
            border: 1px solid rgba(100, 210, 255, 55);
        }

        #chatScroll, #chatContent {
            background: transparent;
            border: none;
        }

        QScrollBar:vertical {
            background: rgba(255, 255, 255, 18);
            width: 10px;
            margin: 4px 0 4px 0;
            border-radius: 5px;
        }

        QScrollBar::handle:vertical {
            background: rgba(104, 225, 255, 145);
            min-height: 42px;
            border-radius: 5px;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
            background: transparent;
            border: none;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: transparent;
        }

        #emptyHintLabel {
            color: rgba(185, 220, 240, 130);
            font-size: 26px;
            font-weight: 700;
            background: transparent;
        }

        #messageRow {
            background: transparent;
        }

        #userBubble {
            background-color: rgba(0, 126, 255, 215);
            border-radius: 18px;
            border: 1px solid rgba(125, 215, 255, 95);
        }

        #robotBubble {
            background-color: rgba(15, 28, 43, 225);
            border-radius: 18px;
            border: 1px solid rgba(92, 160, 210, 75);
        }

        #systemBubble {
            background-color: rgba(110, 130, 150, 65);
            border-radius: 14px;
            border: 1px solid rgba(180, 220, 255, 35);
        }

        #bubbleName {
            color: #68E1FF;
            font-size: 15px;
            font-weight: 700;
            background: transparent;
        }

        #bubbleText {
            color: #FFFFFF;
            font-size: 24px;
            line-height: 1.45;
            background: transparent;
        }

        QPushButton {
            border: none;
            border-radius: 24px;
            font-size: 28px;
            font-weight: 800;
            padding: 16px;
        }

        QPushButton:disabled {
            background-color: rgba(80, 95, 110, 140);
            color: rgba(210, 220, 230, 120);
        }

        #recordButton {
            color: white;
            background-color: #087BFF;
            border: 1px solid rgba(130, 225, 255, 135);
        }

        #recordButton:hover {
            background-color: #1590FF;
        }

        #recordButtonActive {
            color: white;
            background-color: #F0642D;
            border: 1px solid rgba(255, 205, 170, 150);
        }

        #exitChatButton {
            color: rgba(230, 245, 255, 230);
            background-color: rgba(13, 28, 45, 200);
            border: 1px solid rgba(120, 200, 255, 95);
        }

        #exitChatButton:hover {
            background-color: rgba(18, 42, 66, 230);
        }

        #faceStatusLabel {
            font-size: 18px;
            color: rgba(145, 225, 255, 145);
            background: transparent;
        }

        #facePreviewCard {
            background-color: rgba(4, 10, 18, 150);
            border-radius: 28px;
            border: 1px solid rgba(100, 210, 255, 58);
        }

        #faceImageLabel {
            background-color: rgba(0, 0, 0, 205);
            border-radius: 22px;
            color: rgba(160, 180, 200, 130);
            font-size: 25px;
            font-weight: 700;
        }

        #faceResultBar {
            background-color: rgba(7, 18, 30, 175);
            border-radius: 24px;
            border: 1px solid rgba(100, 210, 255, 55);
        }

        #userEmotionIconLabel {
            font-size: 50px;
            font-weight: 900;
            color: #67E8F9;
            letter-spacing: 4px;
            background: transparent;
        }

        #userEmotionNameLabel {
            font-size: 28px;
            font-weight: 850;
            color: #F5F7FA;
            background: transparent;
        }

        #userEmotionProbLabel {
            font-size: 18px;
            color: rgba(185, 210, 225, 190);
            background: transparent;
        }

        #faceTipLabel {
            font-size: 18px;
            font-weight: 700;
            color: rgba(104, 225, 255, 175);
            background: transparent;
        }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec_())
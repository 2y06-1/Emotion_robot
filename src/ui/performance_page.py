from __future__ import annotations

import math
from typing import Iterable, List, Optional, TYPE_CHECKING

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    try:
        from monitor.performance_monitor import PerformanceSnapshot
    except ImportError:
        from performance_monitor import PerformanceSnapshot


class MetricCard(QFrame):
    """上半部分的紧凑指标卡。"""

    def __init__(
        self,
        title: str,
        value: str = "等待首次交互",
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("performanceMetricCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(88)
        self.setMaximumHeight(96)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 9, 13, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("performanceMetricTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("performanceMetricValue")
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value_label.setTextInteractionFlags(Qt.NoTextInteraction)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("performanceMetricSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label, 1)
        layout.addWidget(self.subtitle_label)

        self.set_value(value)

    def set_value(self, text: str) -> None:
        value = str(text)
        self.value_label.setText(value)

        # “等待首次交互”等长文本自动缩小，数字保持醒目。
        font = QFont(self.value_label.font())
        font.setBold(True)
        if len(value) >= 6:
            font.setPointSize(15)
        elif len(value) >= 4:
            font.setPointSize(19)
        else:
            font.setPointSize(22)
        self.value_label.setFont(font)

    def set_subtitle(self, text: str) -> None:
        self.subtitle_label.setText(str(text or ""))


class TrendChart(QWidget):
    """不依赖第三方库的轻量折线图。"""

    def __init__(
        self,
        title: str,
        unit: str,
        fixed_maximum: Optional[float] = None,
        display_divisor: float = 1.0,
        minimum_auto_maximum: float = 1.0,
        empty_text: str = "等待数据",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.title = str(title)
        self.unit = str(unit)
        self.fixed_maximum = fixed_maximum
        self.display_divisor = max(float(display_divisor), 1e-9)
        self.minimum_auto_maximum = max(float(minimum_auto_maximum), 1e-9)
        self.empty_text = str(empty_text)
        self.values: List[float] = []

        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_values(self, values: Iterable[float]) -> None:
        sanitized: List[float] = []
        for item in values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                sanitized.append(value)
        self.values = sanitized
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt命名约定
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        self._draw_background(painter, rect)

        title_rect = QRectF(rect.left() + 13, rect.top() + 7, rect.width() - 26, 24)
        painter.setPen(QColor(236, 247, 255))
        title_font = QFont("Microsoft YaHei", 11)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title)

        if self.values:
            latest_text = self._format_display_value(self.values[-1])
            painter.setPen(QColor(104, 225, 255))
            value_font = QFont("Microsoft YaHei", 10)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.drawText(
                title_rect,
                Qt.AlignRight | Qt.AlignVCenter,
                latest_text,
            )

        plot_rect = QRectF(
            rect.left() + 42,
            rect.top() + 38,
            max(10.0, rect.width() - 56),
            max(10.0, rect.height() - 58),
        )

        if not self.values:
            painter.setPen(QColor(170, 205, 225, 130))
            empty_font = QFont("Microsoft YaHei", 11)
            painter.setFont(empty_font)
            painter.drawText(plot_rect, Qt.AlignCenter, self.empty_text)
            return

        maximum = self._calculate_maximum()
        self._draw_grid(painter, plot_rect, maximum)
        self._draw_curve(painter, plot_rect, maximum)

    @staticmethod
    def _draw_background(painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(100, 210, 255, 55), 1))
        painter.setBrush(QColor(5, 15, 25, 185))
        painter.drawRoundedRect(rect, 18, 18)

    def _calculate_maximum(self) -> float:
        if self.fixed_maximum is not None:
            return max(float(self.fixed_maximum), 1e-9)

        data_max = max(self.values) if self.values else 0.0
        # 留20%顶部空间，且端到端曲线最低按1秒范围显示。
        return max(self.minimum_auto_maximum, data_max * 1.2, 1e-9)

    def _draw_grid(self, painter: QPainter, plot_rect: QRectF, maximum: float) -> None:
        grid_pen = QPen(QColor(150, 205, 235, 35), 1, Qt.DashLine)
        axis_pen = QPen(QColor(160, 220, 245, 70), 1)

        label_font = QFont("Microsoft YaHei", 8)
        painter.setFont(label_font)

        for index in range(3):
            ratio = index / 2.0
            y = plot_rect.bottom() - ratio * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(
                QPointF(plot_rect.left(), y),
                QPointF(plot_rect.right(), y),
            )

            raw_value = maximum * ratio
            label = self._format_display_value(raw_value, compact=True)
            painter.setPen(QColor(170, 205, 225, 120))
            label_rect = QRectF(
                plot_rect.left() - 39,
                y - 9,
                34,
                18,
            )
            painter.drawText(label_rect, Qt.AlignRight | Qt.AlignVCenter, label)

        painter.setPen(axis_pen)
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())

    def _draw_curve(self, painter: QPainter, plot_rect: QRectF, maximum: float) -> None:
        count = len(self.values)
        if count == 1:
            x_positions = [plot_rect.center().x()]
        else:
            step = plot_rect.width() / max(1, count - 1)
            x_positions = [plot_rect.left() + step * i for i in range(count)]

        points: List[QPointF] = []
        for x, value in zip(x_positions, self.values):
            normalized = max(0.0, min(1.0, value / maximum))
            y = plot_rect.bottom() - normalized * plot_rect.height()
            points.append(QPointF(x, y))

        if not points:
            return

        if len(points) >= 2:
            line_path = QPainterPath(points[0])
            for point in points[1:]:
                line_path.lineTo(point)

            fill_path = QPainterPath(line_path)
            fill_path.lineTo(points[-1].x(), plot_rect.bottom())
            fill_path.lineTo(points[0].x(), plot_rect.bottom())
            fill_path.closeSubpath()

            gradient = QLinearGradient(0, plot_rect.top(), 0, plot_rect.bottom())
            gradient.setColorAt(0.0, QColor(0, 205, 255, 70))
            gradient.setColorAt(1.0, QColor(0, 110, 255, 4))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawPath(fill_path)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(92, 224, 255), 2.2))
            painter.drawPath(line_path)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(215, 255, 255))
        for point in points[-3:]:
            painter.drawEllipse(point, 2.6, 2.6)

    def _format_display_value(self, raw_value: float, compact: bool = False) -> str:
        value = raw_value / self.display_divisor
        if self.unit == "%":
            return f"{value:.0f}%" if compact else f"{value:.1f}%"
        if self.unit == "s":
            return f"{value:.1f}s" if compact else f"{value:.2f} s"
        if compact:
            return f"{value:.1f}"
        return f"{value:.2f} {self.unit}".strip()


class PerformancePage(QWidget):
    """系统状态第4页。"""

    back_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("performancePage")
        self.setMinimumSize(800, 480)
        self._build_ui()
        self.setStyleSheet(self._style_sheet())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 14, 20, 13)
        root.setSpacing(8)

        # 顶部栏
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        self.back_button = QPushButton("← 返回聊天")
        self.back_button.setObjectName("performanceBackButton")
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.setFixedHeight(38)
        self.back_button.clicked.connect(self.back_clicked.emit)

        self.title_label = QLabel("系统运行状态")
        self.title_label.setObjectName("performancePageTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.provider_badge = QLabel("本地推理 · CPU")
        self.provider_badge.setObjectName("performanceProviderBadge")
        self.provider_badge.setAlignment(Qt.AlignCenter)
        self.provider_badge.setMinimumWidth(130)
        self.provider_badge.setFixedHeight(34)

        top_bar.addWidget(self.back_button, 0)
        top_bar.addStretch(1)
        top_bar.addWidget(self.title_label, 0)
        top_bar.addStretch(1)
        top_bar.addWidget(self.provider_badge, 0)

        # 4个主指标
        cards = QHBoxLayout()
        cards.setSpacing(8)

        self.fps_card = MetricCard("视觉帧率", "等待视觉", "完整视觉链路")
        self.llm_card = MetricCard("LLM耗时", "等待首次交互", "完整回复")
        self.e2e_card = MetricCard("语音交互延迟", "等待首次交互", "说完到播报")
        self.cpu_card = MetricCard("CPU占用", "采集中", "整机负载")

        cards.addWidget(self.fps_card, 1)
        cards.addWidget(self.llm_card, 1)
        cards.addWidget(self.e2e_card, 1)
        cards.addWidget(self.cpu_card, 1)

        # 最近一次交互分解
        self.interaction_bar = QFrame()
        self.interaction_bar.setObjectName("performanceInteractionBar")
        interaction_layout = QHBoxLayout(self.interaction_bar)
        interaction_layout.setContentsMargins(14, 6, 14, 6)
        interaction_layout.setSpacing(8)

        self.interaction_detail_label = QLabel("最近交互：等待首次交互")
        self.interaction_detail_label.setObjectName("performanceInteractionText")
        self.interaction_detail_label.setAlignment(Qt.AlignCenter)
        interaction_layout.addWidget(self.interaction_detail_label, 1)

        # 两条趋势曲线
        charts = QHBoxLayout()
        charts.setSpacing(9)

        self.cpu_chart = TrendChart(
            title="CPU负载 · 最近60秒",
            unit="%",
            fixed_maximum=100.0,
            display_divisor=1.0,
            minimum_auto_maximum=100.0,
            empty_text="正在采集CPU数据",
        )
        self.e2e_chart = TrendChart(
            title="交互延迟 · 最近10轮",
            unit="s",
            fixed_maximum=None,
            display_divisor=1000.0,
            minimum_auto_maximum=1000.0,
            empty_text="等待完成首次交互",
        )

        charts.addWidget(self.cpu_chart, 1)
        charts.addWidget(self.e2e_chart, 1)

        # 底部状态栏
        self.footer = QFrame()
        self.footer.setObjectName("performanceFooter")
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(12, 3, 12, 3)
        footer_layout.setSpacing(8)

        self.memory_label = QLabel("程序内存 --")
        self.uptime_label = QLabel("连续运行 00:00:00")
        self.backend_label = QLabel("视觉后端 CPU")
        self.status_label = QLabel("系统正常")

        for label in (
            self.memory_label,
            self.uptime_label,
            self.backend_label,
            self.status_label,
        ):
            label.setObjectName("performanceFooterText")
            label.setAlignment(Qt.AlignCenter)

        footer_layout.addWidget(self.memory_label, 1)
        footer_layout.addWidget(self.uptime_label, 1)
        footer_layout.addWidget(self.backend_label, 1)
        footer_layout.addWidget(self.status_label, 1)

        root.addLayout(top_bar)
        root.addLayout(cards)
        root.addWidget(self.interaction_bar)
        root.addLayout(charts, 1)
        root.addWidget(self.footer)

    def update_snapshot(self, snapshot: "PerformanceSnapshot") -> None:
        """使用 PerformanceMonitor.snapshot() 的结果刷新界面。"""
        if snapshot is None:
            return

        # 视觉帧率
        vision_has_data = bool(getattr(snapshot, "vision_has_data", False))
        vision_paused = bool(getattr(snapshot, "vision_paused", False))
        vision_fps = getattr(snapshot, "vision_fps", None)

        if vision_paused:
            self.fps_card.set_value("视觉暂停")
            self.fps_card.set_subtitle("等待视觉恢复")
        elif vision_fps is not None:
            self.fps_card.set_value(f"{float(vision_fps):.1f} FPS")
            self.fps_card.set_subtitle("完整视觉链路")
        elif vision_has_data:
            self.fps_card.set_value("计算中")
            self.fps_card.set_subtitle("正在形成滑动窗口")
        else:
            self.fps_card.set_value("等待视觉")
            self.fps_card.set_subtitle("完整视觉链路")

        # 最近一次完整交互
        has_completed = bool(
            getattr(snapshot, "has_completed_interaction", False)
        )
        llm_ms = getattr(snapshot, "llm_ms", None)
        e2e_ms = getattr(snapshot, "end_to_end_ms", None)

        if has_completed and llm_ms is not None:
            self.llm_card.set_value(self._format_seconds(llm_ms))
            self.llm_card.set_subtitle("完整回复")
        else:
            self.llm_card.set_value("等待首次交互")
            self.llm_card.set_subtitle("完整回复")

        if has_completed and e2e_ms is not None:
            self.e2e_card.set_value(self._format_seconds(e2e_ms))
            self.e2e_card.set_subtitle("说完到播报")
        else:
            self.e2e_card.set_value("等待首次交互")
            self.e2e_card.set_subtitle("说完到播报")

        # 整机CPU
        cpu_percent = getattr(snapshot, "cpu_percent", None)
        if cpu_percent is None:
            self.cpu_card.set_value("采集中")
        else:
            self.cpu_card.set_value(f"{float(cpu_percent):.1f}%")

        # 最近交互分解
        if has_completed:
            asr_text = self._format_seconds_or_dash(
                getattr(snapshot, "asr_ms", None)
            )
            llm_text = self._format_seconds_or_dash(llm_ms)
            tts_text = self._format_seconds_or_dash(
                getattr(snapshot, "tts_wait_ms", None)
            )
            self.interaction_detail_label.setText(
                "最近交互："
                f"ASR {asr_text}  ｜  "
                f"LLM {llm_text}  ｜  "
                f"TTS等待 {tts_text}"
            )
        else:
            self.interaction_detail_label.setText("最近交互：等待首次交互")

        # 曲线
        self.cpu_chart.set_values(getattr(snapshot, "cpu_history", ()))
        self.e2e_chart.set_values(
            getattr(snapshot, "end_to_end_history", ())
        )

        # 底部状态
        rss_mb = max(0.0, float(getattr(snapshot, "program_rss_mb", 0.0)))
        self.memory_label.setText(f"程序内存 {self._format_memory(rss_mb)}")

        uptime = max(0.0, float(getattr(snapshot, "uptime_seconds", 0.0)))
        self.uptime_label.setText(
            f"连续运行 {self._format_duration(uptime)}"
        )

        provider = str(getattr(snapshot, "vision_provider", "CPU") or "CPU")
        self.backend_label.setText(f"视觉后端 {provider}")
        self.provider_badge.setText(f"本地推理 · {provider}")

        system_status = str(
            getattr(snapshot, "system_status", "系统正常") or "系统正常"
        )
        self.status_label.setText(system_status)

    @staticmethod
    def _format_seconds(milliseconds: float) -> str:
        return f"{float(milliseconds) / 1000.0:.2f} s"

    @classmethod
    def _format_seconds_or_dash(cls, milliseconds: Optional[float]) -> str:
        if milliseconds is None:
            return "--"
        return cls._format_seconds(float(milliseconds))

    @staticmethod
    def _format_memory(rss_mb: float) -> str:
        if rss_mb >= 1024.0:
            return f"{rss_mb / 1024.0:.2f} GB"
        return f"{rss_mb:.0f} MB"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt命名约定
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(3, 7, 12))

        w = self.width()
        h = self.height()

        glow1 = QRadialGradient(w * 0.18, h * 0.08, min(w, h) * 0.72)
        glow1.setColorAt(0.0, QColor(0, 115, 255, 42))
        glow1.setColorAt(0.58, QColor(0, 60, 150, 12))
        glow1.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow1))

        glow2 = QRadialGradient(w * 0.88, h * 0.80, min(w, h) * 0.62)
        glow2.setColorAt(0.0, QColor(0, 210, 255, 24))
        glow2.setColorAt(0.62, QColor(0, 80, 120, 8))
        glow2.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), QBrush(glow2))

        super().paintEvent(event)

    @staticmethod
    def _style_sheet() -> str:
        return """
        QWidget {
            background: transparent;
            color: #F5F7FA;
            font-family: "Microsoft YaHei";
        }

        #performancePage {
            background: transparent;
        }

        #performancePageTitle {
            color: #FFFFFF;
            font-size: 25px;
            font-weight: 800;
            background: transparent;
        }

        #performanceBackButton {
            color: rgba(225, 245, 255, 235);
            background-color: rgba(13, 31, 50, 220);
            border: 1px solid rgba(110, 205, 255, 100);
            border-radius: 16px;
            padding: 5px 13px;
            font-size: 15px;
            font-weight: 700;
        }

        #performanceBackButton:hover {
            background-color: rgba(20, 55, 84, 235);
            border-color: rgba(104, 225, 255, 175);
        }

        #performanceProviderBadge {
            color: #7DE9FF;
            background-color: rgba(0, 125, 205, 45);
            border: 1px solid rgba(104, 225, 255, 90);
            border-radius: 16px;
            padding: 3px 10px;
            font-size: 14px;
            font-weight: 700;
        }

        #performanceMetricCard {
            background-color: rgba(5, 15, 25, 195);
            border: 1px solid rgba(100, 210, 255, 58);
            border-radius: 18px;
        }

        #performanceMetricTitle {
            color: rgba(185, 220, 240, 190);
            font-size: 14px;
            font-weight: 650;
            background: transparent;
        }

        #performanceMetricValue {
            color: #FFFFFF;
            background: transparent;
        }

        #performanceMetricSubtitle {
            color: rgba(125, 210, 240, 150);
            font-size: 11px;
            background: transparent;
        }

        #performanceInteractionBar {
            background-color: rgba(5, 18, 30, 190);
            border: 1px solid rgba(100, 210, 255, 48);
            border-radius: 15px;
        }

        #performanceInteractionText {
            color: rgba(225, 245, 255, 225);
            font-size: 15px;
            font-weight: 650;
            background: transparent;
        }

        #performanceFooter {
            background-color: rgba(4, 13, 22, 185);
            border: 1px solid rgba(100, 210, 255, 42);
            border-radius: 12px;
        }

        #performanceFooterText {
            color: rgba(185, 220, 240, 190);
            font-size: 12px;
            font-weight: 600;
            background: transparent;
        }
        """
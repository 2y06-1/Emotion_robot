from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PerformanceSnapshot:
    """提供给 PyQt UI 的只读性能快照。"""

    # 视觉链路
    vision_fps: Optional[float]
    vision_has_data: bool
    vision_paused: bool
    vision_frame_age_s: Optional[float]

    # 最近一次已完成交互
    asr_ms: Optional[float]
    llm_ms: Optional[float]
    tts_wait_ms: Optional[float]
    end_to_end_ms: Optional[float]
    has_completed_interaction: bool

    # 系统资源
    cpu_percent: Optional[float]
    program_rss_mb: float
    uptime_seconds: float

    # 状态信息
    vision_provider: str
    system_status: str

    # 趋势数据
    cpu_history: Tuple[float, ...]
    end_to_end_history: Tuple[float, ...]


class PerformanceMonitor:
    """线程安全的轻量级性能监控器。"""

    def __init__(
        self,
        vision_provider: str = "CPU",
        cpu_history_size: int = 60,
        end_to_end_history_size: int = 10,
        vision_window_seconds: float = 2.0,
        vision_pause_seconds: float = 2.5,
    ) -> None:
        self._lock = threading.RLock()
        self._start_monotonic = time.monotonic()

        self._vision_provider = str(vision_provider or "CPU")
        self._system_status = "系统正常"

        self._vision_window_seconds = max(0.5, float(vision_window_seconds))
        self._vision_pause_seconds = max(
            self._vision_window_seconds,
            float(vision_pause_seconds),
        )

        # 只需保留一个短窗口中的帧时间。240足以覆盖高于100 FPS的情况。
        self._vision_frame_times: Deque[float] = deque(maxlen=240)
        self._last_vision_frame_at: Optional[float] = None

        self._cpu_history: Deque[float] = deque(
            maxlen=max(2, int(cpu_history_size))
        )
        self._end_to_end_history: Deque[float] = deque(
            maxlen=max(2, int(end_to_end_history_size))
        )

        self._cpu_percent: Optional[float] = None
        self._program_rss_mb: float = 0.0

        # /proc/stat 的上一组累计值，用于计算相邻采样间隔内的CPU占用率。
        self._previous_cpu_total: Optional[int] = None
        self._previous_cpu_idle: Optional[int] = None

        # TTS 子进程 PID。未设置或进程已退出时只统计主进程。
        self._tts_pid: Optional[int] = None

        # 最近一次已经完整走到“扬声器开始播放”的交互数据。
        self._last_asr_ms: Optional[float] = None
        self._last_llm_ms: Optional[float] = None
        self._last_tts_wait_ms: Optional[float] = None
        self._last_end_to_end_ms: Optional[float] = None
        self._has_completed_interaction = False

        # 当前正在进行的交互。采用 pending 字段，避免把半轮数据和上一轮混在一起。
        self._interaction_active = False
        self._recording_finished_at: Optional[float] = None
        self._tts_submitted_at: Optional[float] = None
        self._pending_asr_ms: Optional[float] = None
        self._pending_llm_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # 视觉性能
    # ------------------------------------------------------------------
    def record_vision_frame(self, timestamp: Optional[float] = None) -> None:
        """在一次完整视觉循环结束后调用一次。

        完整视觉循环应包含：取帧、人脸检测、表情分类、情绪平滑和结果提交。
        不要在同一帧的多个内部步骤重复调用。
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            return

        with self._lock:
            self._vision_frame_times.append(now)
            self._last_vision_frame_at = now
            self._trim_vision_frames_locked(now)

    def _trim_vision_frames_locked(self, now: float) -> None:
        cutoff = now - self._vision_window_seconds
        while self._vision_frame_times and self._vision_frame_times[0] < cutoff:
            self._vision_frame_times.popleft()

    def _vision_state_locked(
        self,
        now: float,
    ) -> Tuple[Optional[float], bool, bool, Optional[float]]:
        if self._last_vision_frame_at is None:
            return None, False, False, None

        age = max(0.0, now - self._last_vision_frame_at)
        paused = age > self._vision_pause_seconds
        if paused:
            return None, True, True, age

        self._trim_vision_frames_locked(now)
        count = len(self._vision_frame_times)
        if count < 2:
            return None, True, False, age

        elapsed = self._vision_frame_times[-1] - self._vision_frame_times[0]
        if elapsed <= 1e-9:
            return None, True, False, age

        # N个时间点之间有N-1个帧间隔。
        fps = (count - 1) / elapsed
        if not math.isfinite(fps) or fps < 0:
            fps = None
        return fps, True, False, age

    # ------------------------------------------------------------------
    # 一轮对话性能
    # ------------------------------------------------------------------
    def mark_recording_finished(self, timestamp: Optional[float] = None) -> None:
        """标记用户语音真正结束，作为端到端延迟的起点。"""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            return

        with self._lock:
            self._interaction_active = True
            self._recording_finished_at = now
            self._tts_submitted_at = None
            self._pending_asr_ms = None
            self._pending_llm_ms = None

    def set_asr_latency(self, elapsed_ms: Optional[float]) -> None:
        value = self._sanitize_latency(elapsed_ms)
        if value is None:
            return

        with self._lock:
            if self._interaction_active:
                self._pending_asr_ms = value
            else:
                # 便于单独测试ASR时直接看到结果。
                self._last_asr_ms = value

    def set_llm_latency(self, elapsed_ms: Optional[float]) -> None:
        value = self._sanitize_latency(elapsed_ms)
        if value is None:
            return

        with self._lock:
            if self._interaction_active:
                self._pending_llm_ms = value
            else:
                self._last_llm_ms = value

    def mark_tts_submitted(self, timestamp: Optional[float] = None) -> None:
        """标记回复文本已提交给TTS Worker。"""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            return

        with self._lock:
            if not self._interaction_active:
                # 防止调用顺序异常时端到端起点缺失。
                self._interaction_active = True
            self._tts_submitted_at = now

    def mark_tts_playback_started(
        self,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """标记扬声器真正开始播放。

        返回值：
            (tts_wait_ms, end_to_end_ms)

        该调用会把本轮 pending 数据提交为“最近一次已完成交互”，并把端到端
        延迟加入最近10轮历史。
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            return None, None

        with self._lock:
            tts_wait_ms: Optional[float] = None
            end_to_end_ms: Optional[float] = None

            if self._tts_submitted_at is not None:
                tts_wait_ms = max(0.0, (now - self._tts_submitted_at) * 1000.0)

            if self._recording_finished_at is not None:
                end_to_end_ms = max(
                    0.0,
                    (now - self._recording_finished_at) * 1000.0,
                )

            if self._pending_asr_ms is not None:
                self._last_asr_ms = self._pending_asr_ms
            if self._pending_llm_ms is not None:
                self._last_llm_ms = self._pending_llm_ms
            if tts_wait_ms is not None:
                self._last_tts_wait_ms = tts_wait_ms
            if end_to_end_ms is not None:
                self._last_end_to_end_ms = end_to_end_ms
                self._end_to_end_history.append(end_to_end_ms)

            # 至少得到了端到端时间，才算一轮可以用于答辩展示的完整交互。
            if end_to_end_ms is not None:
                self._has_completed_interaction = True

            self._interaction_active = False
            self._recording_finished_at = None
            self._tts_submitted_at = None
            self._pending_asr_ms = None
            self._pending_llm_ms = None

            return tts_wait_ms, end_to_end_ms

    def cancel_interaction(self) -> None:
        """取消或失败时清除本轮未完成数据，但保留上一轮完整结果。"""
        with self._lock:
            self._interaction_active = False
            self._recording_finished_at = None
            self._tts_submitted_at = None
            self._pending_asr_ms = None
            self._pending_llm_ms = None

    @staticmethod
    def _sanitize_latency(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or result < 0:
            return None
        return result

    # ------------------------------------------------------------------
    # 系统资源采样
    # ------------------------------------------------------------------
    def set_tts_pid(self, pid: Optional[int]) -> None:
        """设置TTS子进程PID；传None或非正数可清除。"""
        normalized: Optional[int]
        try:
            pid_value = int(pid) if pid is not None else 0
            normalized = pid_value if pid_value > 0 else None
        except (TypeError, ValueError):
            normalized = None

        with self._lock:
            self._tts_pid = normalized

    def sample_system(self) -> None:
        """采样整机CPU和程序内存，建议每秒调用一次。"""
        cpu_percent = self._sample_cpu_percent()

        main_rss_kb = self._read_process_rss_kb(os.getpid()) or 0
        with self._lock:
            tts_pid = self._tts_pid

        tts_rss_kb = 0
        if tts_pid is not None and tts_pid != os.getpid():
            tts_rss_kb = self._read_process_rss_kb(tts_pid) or 0

        program_rss_mb = (main_rss_kb + tts_rss_kb) / 1024.0

        with self._lock:
            if cpu_percent is not None:
                self._cpu_percent = cpu_percent
                self._cpu_history.append(cpu_percent)
            self._program_rss_mb = max(0.0, program_rss_mb)

    def _sample_cpu_percent(self) -> Optional[float]:
        current = self._read_cpu_totals()
        if current is None:
            return None

        total, idle = current
        with self._lock:
            previous_total = self._previous_cpu_total
            previous_idle = self._previous_cpu_idle
            self._previous_cpu_total = total
            self._previous_cpu_idle = idle

        if previous_total is None or previous_idle is None:
            return None

        total_delta = total - previous_total
        idle_delta = idle - previous_idle
        if total_delta <= 0:
            return None

        busy_delta = max(0, total_delta - idle_delta)
        percent = busy_delta * 100.0 / total_delta
        return max(0.0, min(100.0, percent))

    @staticmethod
    def _read_cpu_totals() -> Optional[Tuple[int, int]]:
        """读取/proc/stat第一行，返回(total, idle+iowait)。"""
        try:
            with open("/proc/stat", "r", encoding="utf-8") as file:
                first_line = file.readline().strip()
        except (OSError, UnicodeError):
            return None

        parts = first_line.split()
        if not parts or parts[0] != "cpu":
            return None

        try:
            values = [int(item) for item in parts[1:]]
        except ValueError:
            return None

        if len(values) < 4:
            return None

        total = sum(values)
        idle = values[3]
        if len(values) > 4:
            idle += values[4]  # iowait
        return total, idle

    @staticmethod
    def _read_process_rss_kb(pid: int) -> Optional[int]:
        """读取/proc/<pid>/status中的VmRSS，单位kB。"""
        path = f"/proc/{int(pid)}/status"
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.startswith("VmRSS:"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        return None
                    return max(0, int(parts[1]))
        except (OSError, ValueError, UnicodeError):
            return None
        return None

    # ------------------------------------------------------------------
    # 状态和快照
    # ------------------------------------------------------------------
    def set_system_status(self, status: str) -> None:
        text = str(status or "系统正常").strip() or "系统正常"
        with self._lock:
            self._system_status = text

    def set_vision_provider(self, provider: str) -> None:
        text = str(provider or "CPU").strip() or "CPU"
        with self._lock:
            self._vision_provider = text

    def snapshot(self) -> PerformanceSnapshot:
        now = time.monotonic()
        with self._lock:
            vision_fps, vision_has_data, vision_paused, frame_age = (
                self._vision_state_locked(now)
            )

            return PerformanceSnapshot(
                vision_fps=vision_fps,
                vision_has_data=vision_has_data,
                vision_paused=vision_paused,
                vision_frame_age_s=frame_age,
                asr_ms=self._last_asr_ms,
                llm_ms=self._last_llm_ms,
                tts_wait_ms=self._last_tts_wait_ms,
                end_to_end_ms=self._last_end_to_end_ms,
                has_completed_interaction=self._has_completed_interaction,
                cpu_percent=self._cpu_percent,
                program_rss_mb=self._program_rss_mb,
                uptime_seconds=max(0.0, now - self._start_monotonic),
                vision_provider=self._vision_provider,
                system_status=self._system_status,
                cpu_history=tuple(self._cpu_history),
                end_to_end_history=tuple(self._end_to_end_history),
            )

    def reset_histories(self) -> None:
        """仅清空趋势曲线；不会重置最近一次交互的指标。"""
        with self._lock:
            self._cpu_history.clear()
            self._end_to_end_history.clear()

    def seed_cpu_history(self, values: Sequence[float]) -> None:
        """测试UI时可注入CPU历史，正式运行通常不需要调用。"""
        with self._lock:
            self._cpu_history.clear()
            for value in values:
                sanitized = self._sanitize_percentage(value)
                if sanitized is not None:
                    self._cpu_history.append(sanitized)

    @staticmethod
    def _sanitize_percentage(value: object) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return max(0.0, min(100.0, result))
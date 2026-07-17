import math
import os
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd


class Voice_Collect:
    """
    自适应能量 VAD 录音器（V4：边校准边检测）。

    保持原调用方式不变：
        path = recorder.record_audio(max_duration=60)

    V4 重点修复：
    1. 不再等待校准完成后才检测语音；
    2. 用户按下录音后可立即讲话；
    3. 保存最近一次底噪，下一次录音可直接使用；
    4. 仅丢弃很短的声卡启动瞬态；
    5. 预录音缓存保留起始字；
    6. 安静帧在后台持续修正底噪；
    7. 连续静音后自动结束。
    """

    def __init__(
        self,
        voice_path,
        device_id,
        max_keep_files,
        voice_threshold,
        min_voice_sec,
        channels,
        chunk_size,
        dtype,
        *,
        frame_ms=30,
        startup_discard_ms=120,
        start_trigger_ms=150,
        end_silence_ms=1100,
        pre_roll_ms=600,
        post_roll_ms=250,
        min_speech_ms=300,
        no_speech_timeout=8.0,
        noise_window_sec=2.0,
        calibration_ms=240,
        default_noise_rms=300.0,
        start_energy_ratio=2.0,
        continue_energy_ratio=1.35,
        start_margin=350.0,
        continue_margin=160.0,
        max_noise_rms=1600.0,
        min_rms=None,
    ):
        self.voice_path = Path(voice_path)
        self.device_id = int(device_id)
        self.max_keep_files = int(max_keep_files)
        self.voice_threshold = max(0, int(voice_threshold))

        # 保留旧参数，兼容原 main.py/config.py。
        self.min_voice_sec = float(min_voice_sec)
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.dtype = str(dtype)

        if self.dtype != "int16":
            raise ValueError("当前录音器只支持 dtype='int16'")
        if self.channels < 1:
            raise ValueError("channels 必须大于等于 1")
        if not 10 <= int(frame_ms) <= 50:
            raise ValueError("frame_ms 建议设置在 10～50 ms")

        self.frame_ms = int(frame_ms)
        self.startup_discard_ms = int(startup_discard_ms)
        self.start_trigger_ms = int(start_trigger_ms)
        self.end_silence_ms = int(end_silence_ms)
        self.pre_roll_ms = int(pre_roll_ms)
        self.post_roll_ms = int(post_roll_ms)
        self.min_speech_ms = int(min_speech_ms)
        self.no_speech_timeout = float(no_speech_timeout)
        self.noise_window_sec = float(noise_window_sec)
        self.calibration_ms = int(calibration_ms)

        self.default_noise_rms = float(default_noise_rms)
        self.start_energy_ratio = float(start_energy_ratio)
        self.continue_energy_ratio = float(continue_energy_ratio)
        self.start_margin = float(start_margin)
        self.continue_margin = float(continue_margin)
        self.max_noise_rms = float(max_noise_rms)

        if min_rms is None:
            min_rms = max(
                100.0,
                min(220.0, self.voice_threshold * 0.20),
            )
        self.min_rms = float(min_rms)

        self._stop_recording = threading.Event()
        self._recording_lock = threading.Lock()
        self._is_recording = False

        # 关键：跨多次录音保存上一次可靠底噪。
        self._last_noise_floor = None

        self.last_result_reason = "idle"

    def clean_files(self):
        try:
            files = sorted(
                self.voice_path.glob("*.wav"),
                key=lambda path: path.stat().st_mtime,
            )
        except OSError as exc:
            print(f"[VAD] 扫描录音目录失败: {exc}", flush=True)
            return

        if len(files) <= self.max_keep_files:
            return

        for file_path in files[:-self.max_keep_files]:
            try:
                os.remove(file_path)
                print(f"[VAD] 删除旧文件: {file_path.name}", flush=True)
            except OSError as exc:
                print(f"[VAD] 删除失败 {file_path}: {exc}", flush=True)

    def stop_recording(self):
        self._stop_recording.set()
        print("[VAD] 收到手动停止请求", flush=True)

    def is_recording(self):
        return self._is_recording

    @staticmethod
    def _to_mono_int16(chunk):
        data = np.asarray(chunk)

        if data.ndim == 2:
            if data.shape[1] == 1:
                data = data[:, 0]
            else:
                data = np.mean(data.astype(np.float32), axis=1)

        return np.clip(data, -32768, 32767).astype(
            np.int16,
            copy=False,
        )

    @staticmethod
    def _frame_features(frame):
        samples = frame.astype(np.float32, copy=False)
        if samples.size == 0:
            return 0.0, 0.0

        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        peak = float(np.max(np.abs(samples)))
        return rms, peak

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0

        array = np.asarray(values, dtype=np.float32)
        return float(np.percentile(array, percentile))

    def _noise_floor(self, history, fallback):
        if not history:
            return fallback

        return max(
            80.0,
            min(
                self.max_noise_rms,
                self._percentile(history, 20.0),
            ),
        )

    def _thresholds(self, noise_floor):
        start_rms = max(
            self.min_rms,
            noise_floor * self.start_energy_ratio,
            noise_floor + self.start_margin,
        )
        continue_rms = max(
            self.min_rms,
            noise_floor * self.continue_energy_ratio,
            noise_floor + self.continue_margin,
        )

        start_peak = max(
            float(self.voice_threshold),
            start_rms * 1.30,
        )
        continue_peak = max(
            float(self.voice_threshold) * 0.75,
            continue_rms * 1.10,
        )

        return start_rms, start_peak, continue_rms, continue_peak

    @staticmethod
    def _save_wav(wav_path, frames, sample_rate):
        if not frames:
            return False

        audio = np.concatenate(frames).astype(np.int16, copy=False)

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio.tobytes())
            return True
        except Exception as exc:
            print(f"[VAD] 保存录音失败: {exc}", flush=True)
            return False

    def record_audio(self, max_duration):
        with self._recording_lock:
            self._is_recording = True
            self._stop_recording.clear()
            self.last_result_reason = "recording"

            try:
                return self._record_audio_impl(float(max_duration))
            finally:
                self._is_recording = False
                self._stop_recording.clear()

    def _record_audio_impl(self, max_duration):
        self.voice_path.mkdir(parents=True, exist_ok=True)
        wav_path = self.voice_path / (
            f"audio_{int(time.time() * 1000)}.wav"
        )

        try:
            device_info = sd.query_devices(self.device_id, "input")
            sample_rate = int(device_info["default_samplerate"])
        except Exception as exc:
            self.last_result_reason = "device_error"
            print(f"[VAD] 获取录音设备失败: {exc}", flush=True)
            return None

        frame_samples = max(
            1,
            int(round(sample_rate * self.frame_ms / 1000.0)),
        )

        discard_frames = max(
            0,
            math.ceil(self.startup_discard_ms / self.frame_ms),
        )
        start_frames = max(
            1,
            math.ceil(self.start_trigger_ms / self.frame_ms),
        )
        end_frames = max(
            1,
            math.ceil(self.end_silence_ms / self.frame_ms),
        )
        pre_frames = max(
            1,
            math.ceil(self.pre_roll_ms / self.frame_ms),
        )
        post_frames = max(
            0,
            math.ceil(self.post_roll_ms / self.frame_ms),
        )
        min_speech_frames = max(
            1,
            math.ceil(self.min_speech_ms / self.frame_ms),
        )
        calibration_frames = max(
            1,
            math.ceil(self.calibration_ms / self.frame_ms),
        )
        noise_history_size = max(
            calibration_frames,
            math.ceil(
                self.noise_window_sec * 1000.0 / self.frame_ms
            ),
            10,
        )

        # 首次录音使用保守默认值，之后直接使用上一次可靠底噪。
        initial_noise = (
            self._last_noise_floor
            if self._last_noise_floor is not None
            else self.default_noise_rms
        )
        initial_noise = max(
            80.0,
            min(self.max_noise_rms, float(initial_noise)),
        )

        pre_roll = deque(maxlen=pre_frames)
        noise_history = deque(maxlen=noise_history_size)
        quiet_samples = []
        recorded_frames = []

        speech_started = False
        speech_run = 0
        speech_frames = 0
        silence_frames = 0
        total_frames = 0
        overflow_count = 0
        calibration_announced = False

        begin_time = time.monotonic()
        last_debug_time = 0.0

        print(
            "[VAD] 开始监听："
            f"采样率={sample_rate}Hz，帧长={self.frame_ms}ms，"
            f"仅丢弃启动瞬态={self.startup_discard_ms}ms，"
            f"初始底噪={initial_noise:.0f}",
            flush=True,
        )

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=self.channels,
                device=self.device_id,
                dtype=self.dtype,
                blocksize=frame_samples,
            ) as stream:
                while True:
                    now = time.monotonic()
                    elapsed = now - begin_time

                    if self._stop_recording.is_set():
                        self.last_result_reason = "manual_stop"
                        print("[VAD] 手动停止录音", flush=True)
                        break

                    if elapsed >= max_duration:
                        self.last_result_reason = "max_duration"
                        print("[VAD] 达到最大录音时长", flush=True)
                        break

                    if (
                        not speech_started
                        and self.no_speech_timeout > 0
                        and elapsed >= self.no_speech_timeout
                    ):
                        self.last_result_reason = "no_speech_timeout"
                        print(
                            "[VAD] 等待超时，未检测到连续人声",
                            flush=True,
                        )
                        break

                    chunk, overflowed = stream.read(frame_samples)
                    total_frames += 1

                    if overflowed:
                        overflow_count += 1

                    frame = self._to_mono_int16(chunk)
                    rms, peak = self._frame_features(frame)

                    # 只丢弃非常短的声卡启动瞬态。
                    if total_frames <= discard_frames:
                        continue

                    pre_roll.append(frame.copy())

                    fallback_noise = (
                        self._last_noise_floor
                        if self._last_noise_floor is not None
                        else initial_noise
                    )
                    noise_floor = self._noise_floor(
                        noise_history,
                        fallback_noise,
                    )

                    (
                        start_rms,
                        start_peak,
                        continue_rms,
                        continue_peak,
                    ) = self._thresholds(noise_floor)

                    is_start_speech = (
                        rms >= start_rms
                        and peak >= start_peak
                    )
                    is_continue_speech = (
                        rms >= continue_rms
                        and peak >= continue_peak
                    )

                    if not speech_started:
                        if is_start_speech:
                            speech_run += 1
                        else:
                            speech_run = max(0, speech_run - 1)

                            # 边监听边校准：只有明显安静的帧才进入底噪池。
                            quiet_limit = max(
                                noise_floor * 1.45,
                                noise_floor + 220.0,
                            )
                            if rms <= quiet_limit:
                                quiet_samples.append(rms)
                                noise_history.append(rms)

                        # 一旦连续人声满足条件，立即开始，不等校准完成。
                        if speech_run >= start_frames:
                            speech_started = True
                            recorded_frames.extend(pre_roll)
                            pre_roll.clear()
                            speech_frames = speech_run
                            silence_frames = 0
                            self.last_result_reason = "speech_started"

                            print(
                                "[VAD] 检测到连续人声："
                                f"rms={rms:.0f}，"
                                f"peak={peak:.0f}，"
                                f"noise={noise_floor:.0f}，"
                                f"start_threshold={start_rms:.0f}",
                                flush=True,
                            )

                        # 收到足够安静帧后更新可靠底噪，但不阻塞语音检测。
                        if (
                            not calibration_announced
                            and len(quiet_samples) >= calibration_frames
                        ):
                            calibrated_noise = max(
                                80.0,
                                min(
                                    self.max_noise_rms,
                                    self._percentile(
                                        quiet_samples,
                                        20.0,
                                    ),
                                ),
                            )
                            self._last_noise_floor = calibrated_noise
                            calibration_announced = True

                            print(
                                "[VAD] 后台底噪校准完成："
                                f"noise={calibrated_noise:.0f}",
                                flush=True,
                            )
                    else:
                        if pre_roll:
                            recorded_frames.append(pre_roll.pop())
                            pre_roll.clear()

                        if is_continue_speech:
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1

                        if silence_frames >= end_frames:
                            remove_count = max(
                                0,
                                silence_frames - post_frames,
                            )
                            if remove_count:
                                del recorded_frames[-remove_count:]

                            self.last_result_reason = "silence_end"
                            print(
                                "[VAD] 检测到连续尾部静音，自动结束",
                                flush=True,
                            )
                            break

                    if now - last_debug_time >= 1.0:
                        state = (
                            "讲话中"
                            if speech_started
                            else "等待人声"
                        )
                        current_threshold = (
                            continue_rms
                            if speech_started
                            else start_rms
                        )
                        print(
                            f"[VAD] {state}: "
                            f"rms={rms:.0f}, peak={peak:.0f}, "
                            f"noise={noise_floor:.0f}, "
                            f"threshold={current_threshold:.0f}, "
                            f"trigger={speech_run}/{start_frames}",
                            flush=True,
                        )
                        last_debug_time = now

        except Exception as exc:
            self.last_result_reason = "record_error"
            print(f"[VAD] 录音错误: {exc}", flush=True)
            return None

        if overflow_count:
            print(
                f"[VAD] 警告：录音缓冲区溢出 {overflow_count} 次",
                flush=True,
            )

        if not speech_started:
            print("[VAD] 未检测到有效连续人声，丢弃录音", flush=True)
            return None

        speech_ms = speech_frames * self.frame_ms
        if speech_frames < min_speech_frames:
            self.last_result_reason = "speech_too_short"
            print(
                f"[VAD] 有效人声过短：约 {speech_ms}ms，"
                f"至少需要 {self.min_speech_ms}ms",
                flush=True,
            )
            return None

        if not self._save_wav(wav_path, recorded_frames, sample_rate):
            self.last_result_reason = "save_error"
            return None

        # 即使本次用户一开始就讲话，结束后的尾部静音也能帮助更新底噪。
        if quiet_samples:
            measured_noise = max(
                80.0,
                min(
                    self.max_noise_rms,
                    self._percentile(quiet_samples, 20.0),
                ),
            )
            self._last_noise_floor = measured_noise

        duration_sec = (
            sum(frame.size for frame in recorded_frames)
            / float(sample_rate)
        )
        self.last_result_reason = "success"

        print(
            f"[VAD] 已保存: {wav_path}，"
            f"音频时长={duration_sec:.2f}s，"
            f"有效人声约={speech_ms / 1000.0:.2f}s，"
            f"下次底噪={self._last_noise_floor or initial_noise:.0f}",
            flush=True,
        )

        self.clean_files()
        return str(wav_path)
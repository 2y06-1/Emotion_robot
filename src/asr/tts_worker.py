import argparse
import sys

from new_txt_tranform import Text_Tranform


EVENT_READY = "TTS_READY"
EVENT_PLAYBACK_STARTED = (
    "TTS_PLAYBACK_STARTED"
)
EVENT_FAILED = "TTS_FAILED"
EVENT_DONE = "TTS_DONE"


def parse_args():
    parser = argparse.ArgumentParser(
        description="TTS worker process"
    )

    parser.add_argument(
        "--model-dir",
        required=True,
    )
    parser.add_argument(
        "--provider",
        required=True,
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--sid",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--speed",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--silence-scale",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--aplay-device",
        required=True,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        choices=[0, 1],
        required=True,
    )
    parser.add_argument(
        "--max-num-sentences",
        type=int,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(
        "[TTS Worker] 正在加载模型...",
        flush=True,
    )

    tts = Text_Tranform(
        model_dir=args.model_dir,
        provider=args.provider,
        num_threads=args.threads,
        sid=args.sid,
        speed=args.speed,
        silence_scale=args.silence_scale,
        aplay_device=args.aplay_device,
        max_chars=args.max_chars,
        warmup=bool(args.warmup),
        max_num_sentences=(
            args.max_num_sentences
        ),
    )

    print(
        "[TTS Worker] 模型就绪，等待输入...",
        flush=True,
    )

    # main.py 会一直读取启动日志，
    # 直到收到这个结构化就绪事件。
    print(
        EVENT_READY,
        flush=True,
    )

    for raw_line in sys.stdin:
        text = raw_line.strip()

        if not text:
            continue

        print(
            f"[TTS Worker] 收到: {text}",
            flush=True,
        )

        playback_started_sent = False

        def notify_playback_started():
            nonlocal playback_started_sent

            if playback_started_sent:
                return

            playback_started_sent = True

            print(
                EVENT_PLAYBACK_STARTED,
                flush=True,
            )

        try:
            success = tts.text_to_speech(
                text,
                on_playback_started=(
                    notify_playback_started
                ),
            )

            if not success:
                print(
                    EVENT_FAILED,
                    flush=True,
                )

        except Exception as exc:
            print(
                "[TTS Worker] 未处理异常: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            print(
                EVENT_FAILED,
                flush=True,
            )

        finally:
            # 无论成功还是失败都要结束本轮，
            # 防止 main.py 一直阻塞读取。
            print(
                EVENT_DONE,
                flush=True,
            )


if __name__ == "__main__":
    main()
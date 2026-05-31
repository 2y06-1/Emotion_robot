import argparse
import sys

from new_txt_tranform import Text_Tranform


def parse_args():
    parser = argparse.ArgumentParser(description="TTS worker process")

    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--sid", type=int, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--silence-scale", type=float, required=True)
    parser.add_argument("--aplay-device", required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--warmup", type=int, choices=[0, 1], required=True)
    parser.add_argument("--max-num-sentences", type=int, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    print("[TTS Worker] 正在加载模型...", flush=True)

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
        max_num_sentences=args.max_num_sentences,
    )

    print("[TTS Worker] 模型就绪，等待输入...", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        print(f"[TTS Worker] 收到: {line}", flush=True)
        tts.text_to_speech(line)
        print("TTS_DONE", flush=True)


if __name__ == "__main__":
    main()

# src/llm/emotion_prompt.py

from __future__ import annotations


SPECIAL_EMOTIONS = {"angry", "happy", "sad", "surprise"}

EMOTION_CN = {
    "happy": "开心",
    "angry": "生气烦躁",
    "sad": "难过低落",
    "surprise": "惊喜惊讶",
    "neutral": "平静",
    "no_face": "未知",
}

EMOTION_GUIDANCE = {
    "happy": "分享用户的喜悦，给予真诚肯定，语气轻快。",
    "angry": "先接住用户的烦躁和委屈，语气平稳，不反驳。",
    "sad": "先承认用户的难受，表达理解和陪伴，不说空洞大道理。",
    "surprise": "回应用户的惊喜或惊讶，语气自然轻快。",
    "neutral": "根据用户原话自然回应，表达理解和陪伴。",
    "no_face": "只根据用户原话自然回应，不猜测用户表情。",
}

EMOTION_EXAMPLES = {
    "happy": [
        "这份开心真好，愿快乐一直陪着你。",
        "看你这么开心，我也替你感到高兴。",
    ],
    "angry": [
        "这确实让人窝火，我理解你的感受。",
        "受了这样的委屈，生气也很正常。",
    ],
    "sad": [
        "今天真的很难熬，我会安静陪着你。",
        "你的难过我听见了，我会陪着你。",
    ],
    "surprise": [
        "这份惊喜太棒了，我也替你开心。",
        "这确实让人惊讶，我懂你的感受。",
    ],
    "neutral": [
        "我听见你的心情了，会一直陪着你。",
        "你的感受很重要，我会认真听着。",
    ],
    "no_face": [
        "我听见你的心情了，会一直陪着你。",
        "你的感受很重要，我会认真听着。",
    ],
}

CN_TO_KEY = {
    "开心": "happy",
    "高兴": "happy",
    "快乐": "happy",
    "生气": "angry",
    "愤怒": "angry",
    "烦躁": "angry",
    "难过": "sad",
    "伤心": "sad",
    "悲伤": "sad",
    "惊讶": "surprise",
    "惊喜": "surprise",
    "吃惊": "surprise",
    "平静": "neutral",
    "中性": "neutral",
    "未检测到人脸": "no_face",
}


def normalize_emotion_key(emotion: str | None) -> str:
    if not emotion:
        return "neutral"

    raw = str(emotion).strip()
    value = raw.lower()

    if value in {
        "angry",
        "happy",
        "sad",
        "surprise",
        "neutral",
        "no_face",
    }:
        return value

    return CN_TO_KEY.get(raw, "neutral")


def normalize_confidence(confidence) -> int:
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0

    if conf <= 1:
        conf *= 100

    return max(0, min(100, int(round(conf))))


def build_robot_system_prompt(
    emotion: str = "neutral",
    confidence=0,
    face_detected: bool = True,
    active_emotion: str | None = None,
    active_confidence=None,
    min_confidence: int = 50,
) -> tuple[str, dict]:
    current_key = normalize_emotion_key(emotion)
    current_conf = normalize_confidence(confidence)

    active_key = (
        normalize_emotion_key(active_emotion)
        if active_emotion
        else None
    )
    active_conf = (
        normalize_confidence(active_confidence)
        if active_confidence is not None
        else 0
    )

    final_key = "neutral"
    final_conf = current_conf
    source = "normal"

    # 主动触发的情绪优先。
    if active_key in SPECIAL_EMOTIONS and active_conf >= min_confidence:
        final_key = active_key
        final_conf = active_conf
        source = "active_event"

    # 没有主动情绪时，使用当前可靠视觉情绪。
    elif (
        face_detected
        and current_key in SPECIAL_EMOTIONS
        and current_conf >= min_confidence
    ):
        final_key = current_key
        final_conf = current_conf
        source = "current_state"

    elif not face_detected:
        final_key = "no_face"
        final_conf = 0
        source = "no_face"

    emotion_cn = EMOTION_CN[final_key]
    guidance = EMOTION_GUIDANCE[final_key]
    examples = EMOTION_EXAMPLES[final_key]

    system_prompt = f"""
你是智能情感陪伴机器人。

当前用户情绪：{emotion_cn}。
情绪回复方向：{guidance}

你的任务：
结合用户刚刚说的具体内容和当前情绪，生成一句自然的中文共情回复。

硬性规则：
1. 只输出一句话。
2. 回复控制在10到20个中文字符左右。
3. 必须体现理解、陪伴或情绪回应。
4. 不得只回答事实而忽略用户情绪。
5. 不反问，不使用问号。
6. 不说教，不命令用户。
7. 不输出建议清单。
8. 不使用表情符号。
9. 不提摄像头、模型、识别结果或置信度。
10. 不输出“用户”“助手”“回复”等角色标记。
11. 不复述用户整句话。
12. 只输出机器人最终要说的话。

风格示例：
{examples[0]}
{examples[1]}

示例只用于学习语气，不得机械照抄。
""".strip()

    info = {
        "mode": "emotion" if final_key in SPECIAL_EMOTIONS else "normal",
        "emotion": final_key,
        "confidence": final_conf,
        "face_detected": bool(face_detected),
        "source": source,
    }

    return system_prompt, info
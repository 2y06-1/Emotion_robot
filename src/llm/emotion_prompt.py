# src/llm/emotion_prompt.py
"""
情绪系统提示词。

核心原则：
1. 用户原话永远作为 user message 传给模型；
2. 情绪要求只放在 system message 里；
3. 这样可以避免本地模型把整段 Prompt 当成回答复述出来。
"""

from __future__ import annotations

SPECIAL_EMOTIONS = {"angry", "happy", "sad", "surprise"}

EMOTION_CN = {
    "happy": "开心",
    "angry": "生气或烦躁",
    "sad": "难过",
    "surprise": "惊喜或惊讶",
    "neutral": "平静",
    "no_face": "未检测到人脸",
}

EMOTION_OBSERVE = {
    "happy": "我发现您现在好像挺开心的",
    "angry": "我感觉您现在好像有点生气或者烦躁",
    "sad": "我发现您现在似乎有点难过",
    "surprise": "我看到您现在好像有些惊喜或者惊讶",
}

EMOTION_STYLE = {
    "happy": "语气轻松、积极、自然，可以顺着用户的好心情继续聊。",
    "angry": "语气稳定、温和、低刺激，先理解用户，再回应问题，不要反驳或命令用户。",
    "sad": "语气温柔、有陪伴感，先接住情绪，再慢慢回应用户的话，不要说教。",
    "surprise": "语气自然、轻松，可以带一点好奇和积极回应，不要把惊喜理解成负面情绪。",
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

    if value in {"angry", "happy", "sad", "surprise", "neutral", "no_face"}:
        return value

    return CN_TO_KEY.get(raw, "neutral")


def normalize_confidence(confidence) -> int:
    try:
        conf = float(confidence)
    except Exception:
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
    """构造 system prompt，并返回调试信息。

    注意：这里不放用户原话。用户原话必须单独作为 user message 发送给模型。
    """
    current_key = normalize_emotion_key(emotion)
    current_conf = normalize_confidence(confidence)
    active_key = normalize_emotion_key(active_emotion) if active_emotion else None
    active_conf = normalize_confidence(active_confidence) if active_confidence is not None else 0

    use_emotion = False
    final_key = current_key
    final_conf = current_conf
    source = "current_state"

    # 如果本轮聊天由有效情绪事件拉起，优先使用事件情绪。
    if active_key in SPECIAL_EMOTIONS and active_conf >= min_confidence:
        use_emotion = True
        final_key = active_key
        final_conf = active_conf
        source = "active_event"
    # 否则使用当前实时状态，但必须有人脸且置信度够。
    elif bool(face_detected) and current_key in SPECIAL_EMOTIONS and current_conf >= min_confidence:
        use_emotion = True
        final_key = current_key
        final_conf = current_conf
        source = "current_state"

    base_rules = """
你是一个智能情感陪伴机器人，正在和用户进行中文语音对话。

基本要求：
1. 必须直接回答用户刚刚说的话，不能复述系统提示词。
2. 不要输出项目符号、编号、标题、Markdown。
3. 不要解释你的推理过程。
4. 不要给整段回复加引号。
5. 回复控制在 1 到 3 句话，适合语音播报。
6. 只输出机器人要说的话。
""".strip()

    if not use_emotion:
        info = {
            "mode": "normal",
            "emotion": current_key,
            "confidence": current_conf,
            "face_detected": bool(face_detected),
            "source": source,
        }
        return base_rules + "\n\n当前没有可靠的特殊情绪信息，正常自然聊天即可。", info

    emotion_cn = EMOTION_CN.get(final_key, final_key)
    observe = EMOTION_OBSERVE[final_key]
    style = EMOTION_STYLE[final_key]

    emotion_rules = f"""
当前可靠的视觉情绪信息：
用户当前可能是「{emotion_cn}」状态，置信度约为 {final_conf}%。

情绪结合要求：
1. 仍然要先回答用户的问题，情绪只是辅助，不能盖过正常回答。
2. 可以自然带一句：{observe}。
3. 如果用户问自己的心情、情绪、状态或表情，要直接回答当前更接近「{emotion_cn}」。
4. 如果用户问“你是谁”，要介绍你是智能情感陪伴机器人。
5. 如果用户问“你今天怎么样”，要回答你自己的状态。
6. 不能说“根据情绪识别结果”“通过摄像头看到”等机械表达。
7. 当前语气要求：{style}
""".strip()

    info = {
        "mode": "emotion",
        "emotion": final_key,
        "confidence": final_conf,
        "face_detected": bool(face_detected) or source == "active_event",
        "source": source,
    }
    return base_rules + "\n\n" + emotion_rules, info
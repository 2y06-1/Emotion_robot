# src/llm/emotion_prompt.py
"""
情绪化 LLM 提示词构造器。 

本项目只把 angry / happy / sad / surprise 视为特殊情绪。
neutral 不触发情绪化回复，no_face / 低置信度 / 未检测到人脸也按普通聊天处理。
"""

from __future__ import annotations


SPECIAL_EMOTIONS = {"angry", "happy", "sad", "surprise"}

EMOTION_CN = {
    "happy": "开心",
    "angry": "生气",
    "sad": "难过",
    "surprise": "惊讶",
    "neutral": "平静",
    "no_face": "未检测到人脸",
}

EMOTION_REPLY_POLICY = {
    "happy": {
        "style": "语气轻松、积极、自然，可以适当回应用户的好心情。",
        "goal": "延续用户的积极状态，给予肯定和陪伴。",
        "avoid": "不要泼冷水，不要突然转向沉重话题，不要夸张奉承。",
    },
    "sad": {
        "style": "语气温柔、慢一点、陪伴感强，先接住情绪，再轻轻回应。",
        "goal": "让用户感觉被理解、被陪伴，而不是被催着立刻变好。",
        "avoid": "不要说“别难过了”，不要说教，不要强行正能量，不要急着给方案。",
    },
    "angry": {
        "style": "语气稳定、低刺激、尊重用户感受，先理解，再轻轻引导。",
        "goal": "降低对抗感，让用户先把情绪放下来。",
        "avoid": "不要反驳用户，不要质问，不要评价用户不应该生气，不要煽动攻击别人。",
    },
    "surprise": {
        "style": "语气可以带一点好奇，但要稳定，不要过度夸张。",
        "goal": "帮助用户确认发生了什么，并自然承接话题。",
        "avoid": "不要制造紧张感，不要过分惊讶，不要抢用户话题。",
    },
}


def normalize_emotion_key(emotion: str | None) -> str:
    """统一情绪 key，兼容 Angry / 生气 / None 等情况。"""
    if not emotion:
        return "neutral"

    value = str(emotion).strip().lower()

    if value in {"angry", "happy", "sad", "surprise", "neutral", "no_face"}:
        return value

    cn_to_key = {
        "开心": "happy",
        "高兴": "happy",
        "快乐": "happy",
        "生气": "angry",
        "愤怒": "angry",
        "难过": "sad",
        "伤心": "sad",
        "悲伤": "sad",
        "惊讶": "surprise",
        "吃惊": "surprise",
        "平静": "neutral",
        "中性": "neutral",
        "未检测到人脸": "no_face",
    }

    return cn_to_key.get(str(emotion).strip(), "neutral")


def normalize_confidence(confidence) -> int:
    """兼容 0.82 和 82 两种置信度格式。"""
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0

    if conf <= 1:
        conf *= 100

    return max(0, min(100, int(round(conf))))


def build_normal_llm_prompt(user_text: str) -> str:
    """普通聊天 Prompt：neutral 或无可靠情绪时使用，不提情绪。"""
    user_text = str(user_text or "").strip()

    return f"""你是一个智能陪伴机器人，正在和用户进行中文语音对话。

【回复要求】
1. 只输出机器人要说的话，不要输出分析过程。
2. 回复自然、简短，适合语音播报，通常 1 到 3 句话即可。
3. 根据用户原话正常回应，不要强行提到情绪识别、摄像头或表情。
4. 不要使用编号、标题、Markdown、括号标签。
5. 不要编造事实，不确定时就温和询问。

【用户原话】
{user_text}

【机器人回复】
"""


def build_special_emotion_llm_prompt(user_text: str, emotion_key: str, confidence: int) -> str:
    """四种特殊情绪专用 Prompt。"""
    user_text = str(user_text or "").strip()
    emotion_key = normalize_emotion_key(emotion_key)
    policy = EMOTION_REPLY_POLICY[emotion_key]
    emotion_cn = EMOTION_CN.get(emotion_key, emotion_key)

    return f"""你是一个智能情感陪伴机器人，正在和用户进行中文语音对话。

【当前情绪信息】
当前视觉模型判断用户可能处于「{emotion_cn}」情绪，置信度约为 {confidence}%。
注意：这是辅助信息，不要直接说“我通过摄像头看到你……”。

【回复策略】
- 推荐语气：{policy["style"]}
- 回复目标：{policy["goal"]}
- 需要避免：{policy["avoid"]}

【强制要求】
1. 只输出机器人要说的话，不要输出分析过程。
2. 回复要自然，像真实陪伴型机器人，不要像心理咨询报告。
3. 回复要短，适合语音播报，通常 1 到 3 句话即可。
4. 可以说“我感觉你现在好像……”，但不要频繁重复用户的表情名称。
5. 如果用户情绪偏负面，先共情，再轻轻引导；不要急着讲大道理。
6. 不要使用编号、标题、Markdown、括号标签。
7. 不要编造事实，不确定时就温和询问。

【用户原话】
{user_text}

【机器人回复】
"""


def build_robot_llm_prompt(
    user_text: str,
    emotion: str = "neutral",
    confidence=0,
    face_detected: bool = True,
    active_emotion: str | None = None,
    min_confidence: int = 45,
) -> tuple[str, dict]:
    """构造最终发给 LLM 的 prompt。

    触发特殊情绪回复的唯一条件：
    - active_emotion 是 angry/happy/sad/surprise；或
    - 当前检测到人脸，当前情绪是 angry/happy/sad/surprise，且置信度 >= min_confidence。

    neutral 永远不触发特殊情绪回复。
    """
    confidence_int = normalize_confidence(confidence)
    current_key = normalize_emotion_key(emotion)
    active_key = normalize_emotion_key(active_emotion) if active_emotion else None

    use_emotion_reply = False
    final_key = current_key

    # 被情绪事件拉起的聊天，优先使用 active_emotion。
    if active_key in SPECIAL_EMOTIONS:
        use_emotion_reply = True
        final_key = active_key
    elif bool(face_detected) and current_key in SPECIAL_EMOTIONS and confidence_int >= min_confidence:
        use_emotion_reply = True
        final_key = current_key

    info = {
        "use_emotion_reply": use_emotion_reply,
        "emotion": final_key,
        "confidence": confidence_int,
        "face_detected": bool(face_detected),
    }

    if use_emotion_reply:
        return build_special_emotion_llm_prompt(user_text, final_key, confidence_int), info

    # neutral / no_face / 低置信度 / 未检测到人脸：普通聊天。
    return build_normal_llm_prompt(user_text), info

# server/board_ws.py
import asyncio
import json
import traceback

import websockets

from server.robot_state import robot_state


# 保存当前已连接的小程序客户端
CLIENTS = set()


def _json_dumps(data):
    """统一 JSON 编码，保证中文不变成 \\uXXXX。"""
    return json.dumps(data, ensure_ascii=False)


def make_status_data():
    """把 robot_state 的首页状态转换成小程序端更容易直接使用的字段。"""
    status = robot_state.get_status()

    face_detected = bool(status.get("face_detected"))
    emotion_key = status.get("emotion", "no_face")
    emotion_cn = status.get("emotion_cn") or emotion_key

    return {
        # 小程序首页直接显示中文情绪
        "emotion": emotion_cn,
        # 保留英文 key，后面需要做逻辑判断时可以用
        "emotionKey": emotion_key,
        "emoji": status.get("emoji", "🤖"),
        "confidence": status.get("confidence", 0),
        "faceDetected": face_detected,
        "faceStatus": "已检测到" if face_detected else "未检测到",
        "lastText": status.get("last_text", ""),
        "updateTime": status.get("update_time", "--"),
    }


def make_chat_data():
    """把聊天记录转换成小程序 chat 页面使用的数据结构。"""
    result = []

    for item in robot_state.get_chat():
        role = item.get("role", "user")
        emotion_key = item.get("emotion", "")
        emotion_cn = item.get("emotion_cn", emotion_key)

        result.append({
            "role": role,
            "roleName": "机器人" if role == "robot" else "用户",
            "content": item.get("content", ""),
            "emotion": emotion_cn,
            "emotionKey": emotion_key,
            "time": item.get("time", "--"),
        })

    return result


def make_stats_data():
    """把统计结果转换成小程序 emotion 页面使用的数据结构。"""
    stats = robot_state.get_stats()
    period_stats = stats.get("periodStats") or stats.get("period_stats") or []

    return {
        "total": stats.get("total", 0),
        "mainEmotion": stats.get("mainEmotion") or stats.get("main_emotion", "暂无数据"),
        "trend": stats.get("trend", "当前还没有足够的情绪数据。"),
        "items": stats.get("items", []),
        # 第 3 个界面新增：按时间段统计图使用这个字段。
        "periodStats": period_stats,
    }


def make_alerts_data():
    """
    把提醒建议转换成小程序 alert 页面使用的数据结构。
    """
    return robot_state.get_alerts()


def make_snapshot():
    """一次性生成四个页面需要的全部数据。"""
    return {
        "status": make_status_data(),
        "chat": make_chat_data(),
        "stats": make_stats_data(),
        "alerts": make_alerts_data(),
    }


async def send_message(websocket, msg_type, data):
    """向某一个客户端发送指定类型消息。"""
    await websocket.send(_json_dumps({
        "type": msg_type,
        "data": data,
    }))


async def send_snapshot(websocket):
    """客户端刚连接时，立即推送一份完整数据，避免小程序页面空白。"""
    snapshot = make_snapshot()

    await send_message(websocket, "status", snapshot["status"])
    await send_message(websocket, "chat", snapshot["chat"])
    await send_message(websocket, "stats", snapshot["stats"])
    await send_message(websocket, "alerts", snapshot["alerts"])


async def broadcast(msg_type, data):
    """广播给所有已连接小程序客户端。"""
    if not CLIENTS:
        return

    dead_clients = []

    for client in list(CLIENTS):
        try:
            await send_message(client, msg_type, data)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        CLIENTS.discard(client)


async def broadcast_loop(push_interval=1.0):
    """
    定时推送数据给小程序。
    这里采用 WebSocket 长连接 + 定时推送。
    robot_state.update_emotion() 继续负责更新状态，
    WebSocket 服务每隔一段时间从 robot_state 读取最新状态并推送给小程序。
    """
    last_payload = ""

    while True:
        try:
            if CLIENTS:
                snapshot = make_snapshot()
                payload = _json_dumps(snapshot)

                # 数据没变化时只推送 status，减少聊天和统计反复刷新。
                if payload != last_payload:
                    await broadcast("status", snapshot["status"])
                    await broadcast("chat", snapshot["chat"])
                    await broadcast("stats", snapshot["stats"])
                    await broadcast("alerts", snapshot["alerts"])
                    last_payload = payload
                else:
                    await broadcast("status", snapshot["status"])

        except Exception as e:
            print(f"[WS] 定时推送失败: {e}", flush=True)

        await asyncio.sleep(push_interval)


async def handle_client_message(websocket, raw_message):
    """处理小程序发来的消息。"""
    try:
        msg = json.loads(raw_message)
    except Exception:
        await send_message(websocket, "error", {
            "message": "消息不是合法 JSON",
            "raw": str(raw_message),
        })
        return

    msg_type = msg.get("type")
    data = msg.get("data") or {}

    if msg_type == "hello":
        await send_snapshot(websocket)
        return

    if msg_type == "get_status":
        await send_message(websocket, "status", make_status_data())
        return

    if msg_type == "get_chat":
        await send_message(websocket, "chat", make_chat_data())
        return

    if msg_type == "get_stats":
        await send_message(websocket, "stats", make_stats_data())
        return

    if msg_type == "get_alerts":
        await send_message(websocket, "alerts", make_alerts_data())
        return

    if msg_type == "clear_chat":
        robot_state.clear_chat()
        await send_message(websocket, "chat", make_chat_data())
        await broadcast("chat", make_chat_data())
        return

    if msg_type == "reset_stats":
        robot_state.reset_stats()
        await send_message(websocket, "stats", make_stats_data())
        await broadcast("stats", make_stats_data())
        await broadcast("alerts", make_alerts_data())
        return

    # 可选：以后如果你想在小程序端发送文字给板子，就用这个消息。
    # 注意：这里只是加入聊天记录，不会触发语音/LLM 对话。
    if msg_type in ("add_chat", "send_chat"):
        content = str(data.get("content", "")).strip()
        role = data.get("role", "user")
        if content:
            robot_state.add_chat(
                role=role,
                content=content,
                emotion=robot_state.current_emotion
            )
            await broadcast("chat", make_chat_data())
        return

    await send_message(websocket, "error", {
        "message": f"未知消息类型: {msg_type}",
    })


async def ws_handler(websocket, path=None):
    """WebSocket 客户端连接处理函数。

    path=None 是为了兼容不同版本 websockets 库。
    """
    CLIENTS.add(websocket)
    client_name = getattr(websocket, "remote_address", "unknown")
    print(f"[WS] 小程序已连接: {client_name}，当前连接数: {len(CLIENTS)}", flush=True)

    try:
        await send_snapshot(websocket)

        async for message in websocket:
            await handle_client_message(websocket, message)

    except Exception as e:
        print(f"[WS] 客户端异常断开: {e}", flush=True)
    finally:
        CLIENTS.discard(websocket)
        print(f"[WS] 小程序已断开，当前连接数: {len(CLIENTS)}", flush=True)


async def _run_ws_server_async(host="0.0.0.0", port=8765, push_interval=0.5):
    async with websockets.serve(
        ws_handler,
        host,
        port,
        ping_interval=20,
        ping_timeout=20,
        max_size=2 * 1024 * 1024,
    ):
        print(f"[WS] WebSocket 服务运行中: ws://{host}:{port}", flush=True)
        asyncio.create_task(broadcast_loop(push_interval=push_interval))
        await asyncio.Future()


def run_ws_server(host="0.0.0.0", port=8765, push_interval=0.5):
    """给 main.py 的后台线程调用。"""
    try:
        asyncio.run(_run_ws_server_async(host=host, port=port, push_interval=push_interval))
    except Exception as e:
        print(f"[WS] WebSocket 服务启动失败: {e}", flush=True)
        traceback.print_exc()

# server/board_api.py
from flask import Flask, jsonify, request
from server.robot_state import robot_state


app = Flask(__name__)


@app.after_request
def after_request(response):
    """方便 Windows 浏览器调试；微信小程序 wx.request 本身也可以访问。"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        "ok": True,
        "message": "board api is running",
        "time": robot_state.now_time(),
    })


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(robot_state.get_status())


@app.route("/api/chat", methods=["GET"])
def chat():
    return jsonify({
        "list": robot_state.get_chat()
    })


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def send_chat():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    data = request.get_json(silent=True) or {}
    content = data.get("content", "")

    if content:
        robot_state.add_chat(
            role="user",
            content=content,
            emotion=robot_state.current_emotion
        )

    return jsonify({
        "ok": True
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(robot_state.get_stats())


@app.route("/api/alerts", methods=["GET"])
def alerts():
    return jsonify({
        "list": robot_state.get_alerts()
    })


@app.route("/api/reset_stats", methods=["POST", "OPTIONS"])
def reset_stats():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    robot_state.reset_stats()
    return jsonify({"ok": True})


@app.route("/api/clear_chat", methods=["POST", "OPTIONS"])
def clear_chat():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    robot_state.clear_chat()
    return jsonify({"ok": True})


def run_api_server(host="0.0.0.0", port=5000):
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )

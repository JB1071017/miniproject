from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64, cv2, numpy as np

from vision import detect_red_led
from calibration import ScreenCalibrator
from touch_engine import TouchEngine

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

calibrator = ScreenCalibrator()
touch = TouchEngine()

@app.route("/")
def board():
    return render_template("board.html")

@app.route("/mobile")
def mobile():
    return render_template("mobile.html")

@socketio.on("frame")
def handle_frame(data):
    img_bytes = base64.b64decode(data.split(",")[1])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    detected, cx, cy = detect_red_led(frame)

    if detected:
        if not calibrator.done:
            calibrator.add_point(cx, cy)
        else:
            X, Y = calibrator.map_to_screen(cx, cy)
            event = touch.update(X, Y)
            emit("touch", event, broadcast=True)
    else:
        emit("touch", touch.release(), broadcast=True)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64

app = Flask(__name__)
app.config["SECRET_KEY"] = "touch-secret"

# IMPORTANT: threading mode (NO eventlet)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mobile")
def mobile():
    return render_template("mobile.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data.split(",")[1])
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Threshold (adjustable)
        _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 2000:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2

                h_img, w_img = gray.shape

                socketio.emit("touch", {
                    "x": cx / w_img,
                    "y": cy / h_img
                })
    except Exception as e:
        print("Frame error:", e)

if __name__ == "__main__":
    # HTTPS REQUIRED for mobile camera
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        ssl_context="adhoc",
        debug=True
    )

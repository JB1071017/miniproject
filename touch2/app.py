from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

prev_y = None
SCROLL_THRESHOLD = 25

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    global prev_y

    data = request.json.get("image")
    if not data:
        return jsonify({"scroll": "none"})

    # Decode base64 image
    img_bytes = base64.b64decode(data.split(",")[1])
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    direction = "none"

    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        index_tip = result.multi_hand_landmarks[0].landmark[8]
        y = int(index_tip.y * h)

        if prev_y is not None:
            if y - prev_y > SCROLL_THRESHOLD:
                direction = "down"
            elif prev_y - y > SCROLL_THRESHOLD:
                direction = "up"

        prev_y = y

    return jsonify({"scroll": direction})

@app.route("/ping")
def ping():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, jsonify
import threading
import vision

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/touch_status")
def touch_status():
    return jsonify({
        "touch": vision.touch_detected,
        "point": vision.touch_point
    })

if __name__ == "__main__":
    t = threading.Thread(target=vision.start_vision, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)

from flask import Flask, render_template
from flask_socketio import SocketIO, join_room, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/mobile/<room>")
def mobile(room):
    return render_template("mobile.html", room=room)

@app.route("/viewer/<room>")
def viewer(room):
    return render_template("viewer.html", room=room)

@socketio.on("join")
def join(data):
    join_room(data["room"])

@socketio.on("offer")
def offer(data):
    emit("offer", data, room=data["room"], include_self=False)

@socketio.on("answer")
def answer(data):
    emit("answer", data, room=data["room"], include_self=False)

@socketio.on("ice")
def ice(data):
    emit("ice", data, room=data["room"], include_self=False)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

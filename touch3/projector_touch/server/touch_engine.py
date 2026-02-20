import time

class TouchEngine:
    def __init__(self):
        self.active = False
        self.prev = None

    def update(self, x, y):
        if not self.active:
            self.active = True
            self.prev = (x, y)
            return {"type": "down", "x": x, "y": y}

        dx = x - self.prev[0]
        dy = y - self.prev[1]
        self.prev = (x, y)
        return {"type": "move", "dx": dx, "dy": dy}

    def release(self):
        if self.active:
            self.active = False
            return {"type": "up"}
        return {"type": "none"}

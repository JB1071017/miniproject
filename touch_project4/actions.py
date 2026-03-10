import pyautogui


def move_cursor_interpolated(prev_x, prev_y, new_x, new_y, steps=8):
    for i in range(1, steps + 1):
        ix = int(prev_x + (new_x - prev_x) * i / steps)
        iy = int(prev_y + (new_y - prev_y) * i / steps)
        pyautogui.moveTo(ix, iy)
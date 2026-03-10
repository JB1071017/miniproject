import pyautogui


class AppState:
    def __init__(self):
        self.calibration_points = []
        self.mode = "idle"   # idle, calibration, interaction
        self.transform_matrix = None
        self.previous_time = 0

        self.screen_width, self.screen_height = pyautogui.size()
        self.smooth_cursor_x, self.smooth_cursor_y = pyautogui.position()
        self.prev_cursor_x, self.prev_cursor_y = self.smooth_cursor_x, self.smooth_cursor_y

        self.dwell_anchor = None
        self.dwell_start_time = None
        self.click_triggered = False
        self.active_dwell_type = None

        self.scroll_anchor_y = None
        self.last_scroll_time = 0

        self.drag_active = False
        self.drag_click_triggered = False

        self.drawing_mode = False
        self.drawing_mouse_down = False
        self.drawing_anchor = None
        self.drawing_start_time = None
        self.drawing_click_ready = True
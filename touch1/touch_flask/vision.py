import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# ===== SHARED STATE (read by Flask) =====
touch_detected = False
touch_point = None  # (sx, sy) in screen coordinates

# ---- Touch latching (IMPORTANT) ----
last_touch_time = 0
TOUCH_HOLD_TIME = 0.25  # seconds

# ======================================
def start_vision():
    global touch_detected, touch_point, last_touch_time

    CAM_ID = 0
    BASELINE_FRAMES = 30
    DIFF_THRESH = 8        # LOWER = more sensitive
    MIN_BLOB_AREA = 10     # LOWER = easier detection

    screen_w, screen_h = pyautogui.size()

    # -------- MediaPipe FaceMesh --------
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # -------- Camera --------
    cap = cv2.VideoCapture(CAM_ID)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera not accessible")

    cam_h, cam_w = frame.shape[:2]

    # Stable eye landmarks (left eye)
    LEFT_EYE = [33, 133, 159, 145]

    def eye_box(lm, pad=40):  # larger pad = easier reflection capture
        pts = [(int(lm[i].x * cam_w), int(lm[i].y * cam_h)) for i in LEFT_EYE]
        xs, ys = zip(*pts)
        return (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(cam_w, max(xs) + pad),
            min(cam_h, max(ys) + pad)
        )

    def get_centroid(roi, base):
        diff = cv2.absdiff(roi, base)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 3)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, th

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            return None, th

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None, th

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), th

    # ================= BASELINE =================
    print("Look straight. Do NOT touch the screen.")
    time.sleep(2)

    acc = None
    roi_shape = None

    for _ in range(BASELINE_FRAMES):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = eye_box(lm)
        roi = frame[y1:y2, x1:x2].astype(np.float32)

        if acc is None:
            roi_shape = roi.shape
            acc = np.zeros(roi_shape, dtype=np.float32)

        roi = cv2.resize(roi, (roi_shape[1], roi_shape[0]))
        acc += roi

    baseline = (acc / BASELINE_FRAMES).astype(np.uint8)
    print("Baseline captured.")

    # ================= CALIBRATION =================
    screen_pts = [
        (50, 50),
        (screen_w - 50, 50),
        (screen_w - 50, screen_h - 50),
        (50, screen_h - 50)
    ]

    ref_pts = []
    print("Calibration: touch each screen corner and press SPACE")

    for sp in screen_pts:
        pyautogui.moveTo(sp)
        time.sleep(0.6)

        while True:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            x1, y1, x2, y2 = eye_box(lm)
            roi = cv2.resize(frame[y1:y2, x1:x2],
                             (baseline.shape[1], baseline.shape[0]))

            c, th = get_centroid(roi, baseline)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow("Calibration", frame)
            cv2.imshow("DIFF", th)

            if cv2.waitKey(1) & 0xFF == ord(' ') and c:
                ref_pts.append((c[0] + x1, c[1] + y1))
                break

    H, _ = cv2.findHomography(
        np.array(ref_pts, dtype=np.float32),
        np.array(screen_pts, dtype=np.float32)
    )

    print("Homography calibrated.")

    # ================= MAIN LOOP =================
    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        current_time = time.time()

        if not res.multi_face_landmarks:
            touch_detected = False
            touch_point = None
            continue

        lm = res.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = eye_box(lm)
        roi = cv2.resize(frame[y1:y2, x1:x2],
                         (baseline.shape[1], baseline.shape[0]))

        c, th = get_centroid(roi, baseline)

        # -------- TOUCH LATCH LOGIC --------
        if c:
            last_touch_time = current_time

            p = np.array([[[c[0] + x1, c[1] + y1]]], dtype=np.float32)
            sx, sy = cv2.perspectiveTransform(p, H)[0][0]

            sx = int(np.clip(sx, 0, screen_w))
            sy = int(np.clip(sy, 0, screen_h))
            touch_point = (sx, sy)

        if current_time - last_touch_time < TOUCH_HOLD_TIME:
            touch_detected = True
        else:
            touch_detected = False
            touch_point = None

        # -------- DEBUG WINDOWS --------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imshow("Vision Debug", frame)
        cv2.imshow("DIFF", th)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

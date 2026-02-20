import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# ================= CONFIG =================
CAM_ID = 0
BASELINE_FRAMES = 30
DIFF_THRESH = 18
MIN_BLOB_AREA = 25
SMOOTH_ALPHA = 0.25
DWELL_TIME = 0.5
# ==========================================

screen_w, screen_h = pyautogui.size()

# -------- MediaPipe FaceMesh --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------- Camera --------
cap = cv2.VideoCapture(CAM_ID)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Camera not accessible")

cam_h, cam_w = frame.shape[:2]

# Stable eye landmarks (tight + accurate)
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

def eye_box(landmarks, idxs, pad=25):
    pts = [(int(landmarks[i].x * cam_w), int(landmarks[i].y * cam_h)) for i in idxs]
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
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BLOB_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

# ================= BASELINE =================
print("Look straight. Do NOT touch the screen.")
input("Press ENTER to capture baseline...")

accL = None
roi_shape = None

for _ in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        continue

    lm = res.multi_face_landmarks[0].landmark
    x1, y1, x2, y2 = eye_box(lm, LEFT_EYE)
    roi = frame[y1:y2, x1:x2].astype(np.float32)

    if accL is None:
        roi_shape = roi.shape
        accL = np.zeros(roi_shape, dtype=np.float32)

    roi = cv2.resize(roi, (roi_shape[1], roi_shape[0]))
    accL += roi

baseL = (accL / BASELINE_FRAMES).astype(np.uint8)
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
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = eye_box(lm, LEFT_EYE)
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (baseL.shape[1], baseL.shape[0]))

        c = get_centroid(roi, baseL)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord(' ') and c:
            ref_pts.append((c[0] + x1, c[1] + y1))
            break

H, _ = cv2.findHomography(
    np.array(ref_pts, dtype=np.float32),
    np.array(screen_pts, dtype=np.float32)
)

print("Calibration done. Running system...")

# ================= MAIN LOOP =================
smoothed = None
dwell_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        continue

    lm = res.multi_face_landmarks[0].landmark
    x1, y1, x2, y2 = eye_box(lm, LEFT_EYE)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (baseL.shape[1], baseL.shape[0]))

    c = get_centroid(roi, baseL)
    if c:
        pt = np.array([[[c[0] + x1, c[1] + y1]]], dtype=np.float32)
        sx, sy = cv2.perspectiveTransform(pt, H)[0][0]

        pos = np.array([sx, sy])
        smoothed = pos if smoothed is None else SMOOTH_ALPHA * pos + (1 - SMOOTH_ALPHA) * smoothed

        pyautogui.moveTo(int(smoothed[0]), int(smoothed[1]))

        if dwell_start is None:
            dwell_start = time.time()
        elif time.time() - dwell_start > DWELL_TIME:
            pyautogui.click()
            dwell_start = None
    else:
        dwell_start = None

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imshow("Accurate Glasses Reflection Touch", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

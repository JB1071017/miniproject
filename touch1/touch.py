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
DWELL_TIME = 0.5
BASELINE_ALPHA = 0.02   # adaptive baseline speed
# ==========================================

screen_w, screen_h = pyautogui.size()

# -------- MediaPipe --------
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

LEFT_EYE = [33, 133, 159, 145]

def eye_box(landmarks, pad=25):
    pts = [(int(landmarks[i].x * cam_w), int(landmarks[i].y * cam_h)) for i in LEFT_EYE]
    xs, ys = zip(*pts)
    return max(0,min(xs)-pad), max(0,min(ys)-pad), min(cam_w,max(xs)+pad), min(cam_h,max(ys)+pad)

# -------- Kalman Filter --------
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

def detect_blobs(roi, base):
    diff = cv2.absdiff(roi, base)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 3)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        if cv2.contourArea(c) >= MIN_BLOB_AREA:
            M = cv2.moments(c)
            if M["m00"] != 0:
                blobs.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    return blobs

# ================= AUTO CALIBRATION =================
print("Auto-calibration: Look at the moving dot and touch it")

calib_screen_pts = []
calib_ref_pts = []

for pt in [(screen_w//2,50),(screen_w-50,screen_h//2),(screen_w//2,screen_h-50),(50,screen_h//2)]:
    pyautogui.moveTo(pt)
    time.sleep(0.6)
    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        x1,y1,x2,y2 = eye_box(lm)
        roi = frame[y1:y2, x1:x2]
        if len(calib_ref_pts)==0:
            roi_shape = roi.shape
            base = roi.copy()
        roi = cv2.resize(roi,(roi_shape[1],roi_shape[0]))
        blobs = detect_blobs(roi, base)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.imshow("Calibration",frame)
        if blobs:
            calib_ref_pts.append((blobs[0][0]+x1,blobs[0][1]+y1))
            calib_screen_pts.append(pt)
            break
        if cv2.waitKey(1)==27:
            exit()

H,_ = cv2.findHomography(np.array(calib_ref_pts,np.float32),
                          np.array(calib_screen_pts,np.float32))

print("Calibration complete.")

# ================= BASELINE =================
acc = np.zeros(roi_shape,np.float32)
for _ in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        x1,y1,x2,y2 = eye_box(lm)
        roi = cv2.resize(frame[y1:y2,x1:x2],(roi_shape[1],roi_shape[0]))
        acc += roi

baseline = (acc/BASELINE_FRAMES).astype(np.uint8)

# ================= MAIN LOOP =================
dwell_start = None

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        continue

    lm = res.multi_face_landmarks[0].landmark
    x1,y1,x2,y2 = eye_box(lm)
    roi = cv2.resize(frame[y1:y2,x1:x2],(roi_shape[1],roi_shape[0]))

    blobs = detect_blobs(roi, baseline)

    # adaptive baseline update
    baseline = cv2.addWeighted(baseline,1-BASELINE_ALPHA,roi,BASELINE_ALPHA,0)

    if blobs:
        pt = np.array([[[blobs[0][0]+x1, blobs[0][1]+y1]]], np.float32)
        sx,sy = cv2.perspectiveTransform(pt,H)[0][0]

        kalman.correct(np.array([[sx],[sy]],np.float32))
        pred = kalman.predict()
        px,py = int(pred[0]), int(pred[1])

        pyautogui.moveTo(px,py)

        if dwell_start is None:
            dwell_start = time.time()
        elif time.time()-dwell_start > DWELL_TIME:
            if len(blobs)==1:
                pyautogui.click()
            else:
                pyautogui.rightClick()
            dwell_start = None
    else:
        dwell_start = None

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
    cv2.imshow("Advanced Touch System",frame)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

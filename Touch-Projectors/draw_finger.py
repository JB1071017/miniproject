#with enhanced drawing and touch detection
# Touch-Projectors - Complete Interactive Touch Screen
# Turn any projector into a fully functional touch screen!

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import math
import mediapipe as mp
import subprocess
import os

# ============================================
# INITIALIZATION
# ============================================

# Initialize video capture
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
time.sleep(1.1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize mouse
mouse = Controller()

# ============================================
# PROJECTION AREA DIMENSIONS
# ============================================

AR = (740, 1280)  # Height, Width
dst_pts = np.float32([
    [0, 0], [AR[1], 0], [AR[1], AR[0]], [0, AR[0]]
])

# ============================================
# CALIBRATION VARIABLES
# ============================================

calibration_points = []
calibration_done = False
reference_frame = None

# Feature tracking
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
reference_kp = None
reference_des = None

# ============================================
# INTERACTION MODES
# ============================================

mode = 0  # 0: Mouse Control, 1: Drawing, 2: Gesture Control
mode_names = ["MOUSE CONTROL", "DRAWING MODE", "GESTURE CONTROL"]

# Drawing variables
drawing = False
drawing_color = (0, 255, 0)  # Green
drawing_thickness = 5
canvas = None
drawings = []  # Store drawings as lists of points

# App shortcuts (you can customize these)
apps = {
    'notepad': 'notepad.exe',
    'calculator': 'calc.exe',
    'paint': 'mspaint.exe',
    'browser': 'start chrome',
    'explorer': 'explorer'
}

# Gesture recognition
last_gesture = ""
gesture_cooldown = 0

# ============================================
# TOUCH PARAMETERS
# ============================================

touch_zone_ratio = 0.8
click_cooldown = 0
click_cooldown_max = 10
touch_active = False
click_method = 0
method_names = ["Touch Zone", "Pinch", "Double Tap", "Finger Stay"]

# Tracking
prev_index_tip = None
stay_counter = 0
stay_threshold = 10
last_click_time = 0
double_tap_threshold = 0.5  # seconds

# Landmark indices
LM_INDEX = {
    'WRIST': 0, 'THUMB_TIP': 4, 'INDEX_TIP': 8,
    'MIDDLE_TIP': 12, 'RING_TIP': 16, 'PINKY_TIP': 20
}

# ============================================
# MOUSE CALLBACK FOR CALIBRATION
# ============================================

def mouse_callback(event, x, y, flags, param):
    global calibration_points, reference_frame, calibration_done
    global reference_kp, reference_des, calibration_display, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN and not calibration_done:
        cv2.circle(calibration_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(calibration_display, str(len(calibration_points)+1), 
                   (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        calibration_points.append((x, y))
        corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
        print(f"Point {len(calibration_points)}: ({x}, {y}) - {corner_names[len(calibration_points)-1]}")
        
        if len(calibration_points) == 4:
            print("\n" + "="*50)
            print("CALIBRATION COMPLETE!")
            print("="*50)
            
            src_pts = np.float32(calibration_points)
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            reference_frame = current_frame.copy()
            reference_kp, reference_des = orb.detectAndCompute(reference_frame, None)
            
            print(f"Detected {len(reference_kp)} features")
            print("\nINTERACTION MODES:")
            print("  m - Switch modes (Mouse/Draw/Gesture)")
            print("  d - Toggle drawing (in Draw mode)")
            print("  c - Clear canvas")
            print(" 1-5 - Launch apps")
            print("  + / - - Adjust touch zone")
            print("  ESC - Exit")
            
            calibration_done = True

# ============================================
# FEATURE TRACKING
# ============================================

def get_current_transform(current_frame):
    global reference_kp, reference_des
    
    if reference_kp is None or reference_des is None:
        return None
    
    current_kp, current_des = orb.detectAndCompute(current_frame, None)
    
    if current_des is None or len(current_des) < 10:
        return None
    
    matches = bf.match(reference_des, current_des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = []
    for match in matches[:100]:
        if match.distance < 50:
            good_matches.append(match)
    
    if len(good_matches) < 10:
        return None
    
    src_pts = np.float32([reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    try:
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    except:
        return None

# ============================================
# GESTURE RECOGNITION
# ============================================

def recognize_gesture(hand_landmarks):
    """Recognize hand gestures"""
    tips = [4, 8, 12, 16, 20]  # Finger tips
    pips = [3, 6, 10, 14, 18]   # Finger PIP joints
    
    fingers = []
    for i in range(5):
        tip = hand_landmarks.landmark[tips[i]]
        pip = hand_landmarks.landmark[pips[i]]
        fingers.append(tip.y < pip.y)  # True if finger is up
    
    # Recognize gestures
    if all(fingers):
        return "OPEN HAND"
    elif not any(fingers):
        return "FIST"
    elif fingers[1] and not any(fingers[2:]):  # Only index up
        return "POINTING"
    elif fingers[1] and fingers[2] and not any(fingers[3:]):  # Peace sign
        return "PEACE"
    elif fingers[0] and fingers[1] and not any(fingers[2:]):  # Gun shape
        return "GUN"
    else:
        return "UNKNOWN"

# ============================================
# LAUNCH APPLICATION
# ============================================

def launch_app(app_name):
    """Launch a Windows application"""
    try:
        if app_name in apps:
            os.system(apps[app_name])
            print(f"Launched {app_name}")
            return True
    except:
        print(f"Failed to launch {app_name}")
    return False

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    global calibration_display, current_frame, calibration_done
    global mode, drawing, canvas, drawings, drawing_color
    global click_method, touch_active, click_cooldown
    global prev_index_tip, stay_counter, last_gesture, gesture_cooldown
    global touch_zone_ratio, last_click_time
    
    print("\n" + "="*70)
    print("🚀 INTERACTIVE TOUCH PROJECTOR - Complete Touch Screen Solution")
    print("="*70)
    print("\nSTEP 1: CALIBRATION")
    print("Click the 4 corners in order:")
    print("  1. TOP-LEFT    2. TOP-RIGHT")
    print("  3. BOTTOM-RIGHT 4. BOTTOM-LEFT")
    print("\n" + "="*70)
    
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    # Initialize canvas
    canvas = np.ones((AR[0], AR[1], 3), dtype=np.uint8) * 255
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = frame.copy()
        calibration_display = frame.copy()
        
        if not calibration_done:
            # Calibration mode
            cv2.putText(calibration_display, f"CALIBRATION: Click corner {len(calibration_points)+1}/4", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for i, pt in enumerate(calibration_points):
                cv2.circle(calibration_display, pt, 8, (0, 255, 0), -1)
                cv2.putText(calibration_display, str(i+1), (pt[0]+15, pt[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Calibration', calibration_display)
            
        else:
            # Tracking mode
            current_H = get_current_transform(frame)
            
            if current_H is not None:
                # Get warped view
                warped = cv2.warpPerspective(frame, current_H, (AR[1], AR[0]))
                
                # Create display with canvas overlay for drawing mode
                if mode == 1 and drawing:
                    # In drawing mode, show canvas
                    display_warped = cv2.addWeighted(warped, 0.3, canvas, 0.7, 0)
                else:
                    display_warped = warped.copy()
                
                # Draw touch zone
                touch_line_y = int(AR[0] * touch_zone_ratio)
                cv2.line(display_warped, (0, touch_line_y), (AR[1], touch_line_y), (255, 255, 0), 2)
                
                # Process hand detection
                rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_warped)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            display_warped, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Get fingertip
                        h, w, _ = warped.shape
                        index_tip = hand_landmarks.landmark[LM_INDEX['INDEX_TIP']]
                        fingertip = (int(index_tip.x * w), int(index_tip.y * h))
                        
                        # Recognize gesture
                        gesture = recognize_gesture(hand_landmarks)
                        if gesture != last_gesture and gesture_cooldown <= 0:
                            print(f"Gesture: {gesture}")
                            last_gesture = gesture
                            gesture_cooldown = 10
                        
                        # Convert to screen coordinates
                        screen_x, screen_y = warped_to_screen(fingertip)
                        
                        # ============================================
                        # MODE-SPECIFIC ACTIONS
                        # ============================================
                        
                        if mode == 0:  # MOUSE CONTROL MODE
                            mouse.position = (screen_x, screen_y)
                            
                            # Click detection
                            click_triggered = False
                            
                            if click_method == 0:  # Touch Zone
                                if fingertip[1] > touch_line_y:
                                    if not touch_active and click_cooldown <= 0:
                                        mouse.click(Button.left)
                                        click_triggered = True
                                        touch_active = True
                                        click_cooldown = click_cooldown_max
                                else:
                                    touch_active = False
                            
                            elif click_method == 1:  # Pinch
                                is_pinching, _, _ = detect_pinch(hand_landmarks, warped.shape)
                                if is_pinching and not touch_active and click_cooldown <= 0:
                                    mouse.click(Button.left)
                                    click_triggered = True
                                    touch_active = True
                                    click_cooldown = click_cooldown_max
                                elif not is_pinching:
                                    touch_active = False
                            
                            elif click_method == 2:  # Double Tap
                                current_time = time.time()
                                if fingertip[1] > touch_line_y:
                                    if current_time - last_click_time < double_tap_threshold:
                                        mouse.double_click(Button.left)
                                        click_triggered = True
                                        click_cooldown = click_cooldown_max
                                    last_click_time = current_time
                            
                            elif click_method == 3:  # Finger Stay
                                if prev_index_tip is not None:
                                    distance = math.sqrt(
                                        (fingertip[0] - prev_index_tip[0])**2 + 
                                        (fingertip[1] - prev_index_tip[1])**2
                                    )
                                    if distance < 5:
                                        stay_counter += 1
                                        if stay_counter >= stay_threshold and click_cooldown <= 0:
                                            mouse.click(Button.left)
                                            click_triggered = True
                                            click_cooldown = click_cooldown_max
                                            stay_counter = 0
                                    else:
                                        stay_counter = max(0, stay_counter - 1)
                                
                                prev_index_tip = fingertip
                            
                            if click_triggered:
                                cv2.putText(display_warped, "CLICK!", 
                                          (fingertip[0]-30, fingertip[1]-60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        elif mode == 1:  # DRAWING MODE
                            # In drawing mode, mouse follows but we also draw
                            mouse.position = (screen_x, screen_y)
                            
                            # Toggle drawing with thumb-index pinch
                            is_pinching, pinch_center, _ = detect_pinch(hand_landmarks, warped.shape, 0.08)
                            
                            if is_pinching:
                                if not drawing:
                                    drawing = True
                                    drawings.append([])  # New stroke
                                # Draw on canvas
                                if drawings:
                                    drawings[-1].append(fingertip)
                                    # Draw on canvas
                                    if len(drawings[-1]) > 1:
                                        cv2.line(canvas, drawings[-1][-2], drawings[-1][-1], 
                                                drawing_color, drawing_thickness)
                            else:
                                drawing = False
                            
                            # Change color with different gestures
                            if gesture == "FIST" and gesture_cooldown <= 0:
                                drawing_color = (np.random.randint(0, 255), 
                                               np.random.randint(0, 255), 
                                               np.random.randint(0, 255))
                                print(f"Color changed to {drawing_color}")
                                gesture_cooldown = 20
                            
                            # Draw on display
                            cv2.circle(display_warped, fingertip, 10, drawing_color, -1)
                            cv2.putText(display_warped, "DRAWING" if drawing else "PINCH TO DRAW", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                      drawing_color if drawing else (255,255,255), 2)
                        
                        elif mode == 2:  # GESTURE CONTROL MODE
                            mouse.position = (screen_x, screen_y)
                            
                            # Gesture actions
                            if gesture_cooldown <= 0:
                                if gesture == "FIST":
                                    # Right click
                                    mouse.click(Button.right)
                                    print("Right click")
                                    gesture_cooldown = 20
                                
                                elif gesture == "PEACE":
                                    # Double click
                                    mouse.double_click(Button.left)
                                    print("Double click")
                                    gesture_cooldown = 20
                                
                                elif gesture == "GUN":
                                    # Launch notepad
                                    launch_app('notepad')
                                    gesture_cooldown = 30
                                
                                elif gesture == "OPEN HAND":
                                    # Scroll up
                                    pyautogui.scroll(3)
                                    gesture_cooldown = 10
                            
                            # Show gesture on screen
                            cv2.putText(display_warped, f"Gesture: {gesture}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Draw fingertip
                        cv2.circle(display_warped, fingertip, 15, (0, 0, 255), 2)
                        cv2.circle(display_warped, fingertip, 5, (255, 255, 255), -1)
                
                else:
                    cv2.putText(display_warped, "No hand detected", 
                              (AR[1]//2-80, AR[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    prev_index_tip = None
                    stay_counter = 0
                
                # Add mode and status info
                cv2.putText(display_warped, f"MODE: {mode_names[mode]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_warped, f"Click: {method_names[click_method]}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Warped View', display_warped)
                
                # Show canvas in separate window for drawing mode
                if mode == 1:
                    cv2.imshow('Canvas', canvas)
                
                # Update cooldowns
                if click_cooldown > 0:
                    click_cooldown -= 1
                if gesture_cooldown > 0:
                    gesture_cooldown -= 1
                
                # Show tracking status
                cv2.putText(calibration_display, f"✓ {mode_names[mode]}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                cv2.putText(calibration_display, "✗ TRACKING LOST", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', calibration_display)
        
        # ============================================
        # KEYBOARD CONTROLS
        # ============================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('m') and calibration_done:
            mode = (mode + 1) % 3
            print(f"\nSwitched to {mode_names[mode]} mode")
        elif key == ord('d') and calibration_done and mode == 1:
            drawing = not drawing
            print(f"Drawing: {'ON' if drawing else 'OFF'}")
        elif key == ord('c') and calibration_done:
            canvas = np.ones((AR[0], AR[1], 3), dtype=np.uint8) * 255
            drawings = []
            print("Canvas cleared")
        elif key == ord('1') and calibration_done:
            launch_app('notepad')
        elif key == ord('2') and calibration_done:
            launch_app('calculator')
        elif key == ord('3') and calibration_done:
            launch_app('paint')
        elif key == ord('4') and calibration_done:
            launch_app('browser')
        elif key == ord('5') and calibration_done:
            launch_app('explorer')
        elif key == ord('t') and calibration_done:
            click_method = (click_method + 1) % 4
            print(f"Click method: {method_names[click_method]}")
        elif key == ord('+') and calibration_done:
            touch_zone_ratio = min(0.95, touch_zone_ratio + 0.05)
            print(f"Touch zone: {int(touch_zone_ratio*100)}%")
        elif key == ord('-') and calibration_done:
            touch_zone_ratio = max(0.5, touch_zone_ratio - 0.05)
            print(f"Touch zone: {int(touch_zone_ratio*100)}%")
        elif key == ord('r'):  # Recalibrate
            calibration_points = []
            calibration_done = False
            reference_frame = None
            reference_kp = None
            reference_des = None
            print("\nCalibration reset")
    
    # Cleanup
    hands.close()
    cv2.destroyAllWindows()
    cap.release()
    print("\nProgram terminated")

# ============================================
# HELPER FUNCTIONS
# ============================================

def warped_to_screen(warped_point):
    screen_width, screen_height = pyautogui.size()
    screen_x = int((warped_point[0] / AR[1]) * screen_width)
    screen_y = int((warped_point[1] / AR[0]) * screen_height)
    return (screen_x, screen_y)

def detect_pinch(hand_landmarks, image_shape, threshold=0.05):
    h, w = image_shape[:2]
    thumb_tip = hand_landmarks.landmark[LM_INDEX['THUMB_TIP']]
    index_tip = hand_landmarks.landmark[LM_INDEX['INDEX_TIP']]
    
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    distance = math.sqrt(dx**2 + dy**2)
    
    center_x = int((thumb_tip.x + index_tip.x) * w / 2)
    center_y = int((thumb_tip.y + index_tip.y) * h / 2)
    
    return distance < threshold, (center_x, center_y), distance

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.release()   
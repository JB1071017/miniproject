# Touch-Projectors - Real Touchscreen Experience
# Works exactly like a tablet: tap, hold, double-tap, and two-finger right-click!

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import math
import mediapipe as mp

# ============================================
# INITIALIZATION
# ============================================

# Initialize video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
time.sleep(1.1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow 2 hands for right-click gestures
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
# TOUCHSCREEN PARAMETERS
# ============================================

# Touch sensitivity
touch_sensitivity = 30  # pixels from bottom

# Touch states
is_touching = False
touch_start_time = 0
touch_position = None
touch_cooldown = 0
touch_cooldown_max = 5

# Hold detection
hold_threshold = 0.8  # seconds to trigger right-click on hold
hold_triggered = False

# Double tap detection
last_tap_time = 0
double_tap_threshold = 0.3  # seconds between taps
last_tap_position = None
double_tap_distance = 50  # max pixels between taps

# Right-click detection (two fingers)
right_click_triggered = False
right_click_cooldown = 0

# For tracking
fingertip_history = []
index_tip_id = 8  # MediaPipe landmark for index fingertip
middle_tip_id = 12  # MediaPipe landmark for middle fingertip

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
            print("\n" + "="*60)
            print("✅ CALIBRATION COMPLETE!")
            print("="*60)
            print("\n📱 TOUCHSCREEN READY!")
            print("\n🎯 How to use:")
            print("  • 👆 One finger: Move cursor")
            print("  • 👆 Quick tap: Left click")
            print("  • 👆 Hold finger: Right click (after 0.8 seconds)")
            print("  • 👆 Double tap: Double click")
            print("  • ✌️ Two fingers: Right click (instant)")
            print("\n⚙️ Controls:")
            print("  + / - : Adjust sensitivity")
            print("  c : Recalibrate")
            print("  ESC : Exit")
            print("="*60)
            
            src_pts = np.float32(calibration_points)
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            reference_frame = current_frame.copy()
            reference_kp, reference_des = orb.detectAndCompute(reference_frame, None)
            
            print(f"Detected {len(reference_kp)} tracking points")
            
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
# SMOOTH POSITION
# ============================================

def smooth_position(new_pos, history, max_history=3):
    if new_pos is None:
        return None
    
    history.append(new_pos)
    if len(history) > max_history:
        history.pop(0)
    
    if len(history) > 1:
        avg_x = int(sum(p[0] for p in history) / len(history))
        avg_y = int(sum(p[1] for p in history) / len(history))
        return (avg_x, avg_y)
    else:
        return new_pos

# ============================================
# CONVERT TO SCREEN COORDINATES
# ============================================

def warped_to_screen(warped_point):
    screen_width, screen_height = pyautogui.size()
    screen_x = int((warped_point[0] / AR[1]) * screen_width)
    screen_y = int((warped_point[1] / AR[0]) * screen_height)
    
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    
    return (screen_x, screen_y)

# ============================================
# DETECT TWO FINGERS (for right click)
# ============================================

def detect_two_fingers(hand_landmarks):
    """Check if both index and middle fingers are extended"""
    if len(hand_landmarks) < 21:
        return False
    
    # Index finger
    index_tip = hand_landmarks.landmark[index_tip_id]
    index_pip = hand_landmarks.landmark[6]  # INDEX_PIP
    index_extended = index_tip.y < index_pip.y
    
    # Middle finger
    middle_tip = hand_landmarks.landmark[middle_tip_id]
    middle_pip = hand_landmarks.landmark[10]  # MIDDLE_PIP
    middle_extended = middle_tip.y < middle_pip.y
    
    # Ring and pinky should be down (for peace sign gesture)
    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]
    ring_extended = ring_tip.y < ring_pip.y
    
    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    # Two fingers up = index and middle up, others down
    return index_extended and middle_extended and not ring_extended and not pinky_extended

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    global calibration_display, current_frame, calibration_done
    global is_touching, touch_start_time, touch_position, touch_cooldown
    global hold_triggered, last_tap_time, last_tap_position
    global right_click_triggered, right_click_cooldown
    global touch_sensitivity, fingertip_history
    
    print("\n" + "="*70)
    print("📱 REAL TOUCHSCREEN PROJECTOR")
    print("="*70)
    print("\nSTEP 1: CALIBRATION")
    print("Click the 4 corners of your projection area:")
    print("  1. TOP-LEFT corner")
    print("  2. TOP-RIGHT corner")
    print("  3. BOTTOM-RIGHT corner")
    print("  4. BOTTOM-LEFT corner")
    print("\n(Just click once on each corner)")
    print("="*70)
    
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = frame.copy()
        calibration_display = frame.copy()
        
        if not calibration_done:
            # ============================================
            # CALIBRATION MODE
            # ============================================
            
            cv2.putText(calibration_display, f"CALIBRATION: Click corner {len(calibration_points)+1}/4", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for i, pt in enumerate(calibration_points):
                cv2.circle(calibration_display, pt, 8, (0, 255, 0), -1)
                cv2.putText(calibration_display, str(i+1), (pt[0]+15, pt[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Calibration', calibration_display)
            
        else:
            # ============================================
            # TOUCHSCREEN MODE
            # ============================================
            
            current_H = get_current_transform(frame)
            
            if current_H is not None:
                warped = cv2.warpPerspective(frame, current_H, (AR[1], AR[0]))
                display = warped.copy()
                
                # Draw touch zone
                touch_line_y = AR[0] - touch_sensitivity
                cv2.line(display, (0, touch_line_y), (AR[1], touch_line_y), (255, 255, 0), 2)
                
                # Process hand detection
                rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_warped)
                
                # Check for two-finger gesture first (priority for right click)
                two_fingers_detected = False
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if detect_two_fingers(hand_landmarks):
                            two_fingers_detected = True
                            
                            # Get position between the two fingers
                            index_tip = hand_landmarks.landmark[index_tip_id]
                            middle_tip = hand_landmarks.landmark[middle_tip_id]
                            h, w, _ = warped.shape
                            
                            x = int((index_tip.x + middle_tip.x) * w / 2)
                            y = int((index_tip.y + middle_tip.y) * h / 2)
                            center = (x, y)
                            
                            # Draw two-finger indicator
                            cv2.circle(display, center, 20, (255, 0, 255), 3)
                            cv2.putText(display, "RIGHT CLICK", (center[0]-50, center[1]-30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            
                            # Trigger right click if not on cooldown
                            if right_click_cooldown <= 0 and not right_click_triggered:
                                screen_x, screen_y = warped_to_screen(center)
                                mouse.click(Button.right)
                                print(f"👉 Right click at ({screen_x}, {screen_y})")
                                right_click_triggered = True
                                right_click_cooldown = 20
                
                # Regular one-finger tracking
                if results.multi_hand_landmarks and not two_fingers_detected:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            display,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Get index fingertip
                        h, w, _ = warped.shape
                        index_tip = hand_landmarks.landmark[index_tip_id]
                        raw_fingertip = (int(index_tip.x * w), int(index_tip.y * h))
                        
                        # Smooth position
                        fingertip = smooth_position(raw_fingertip, fingertip_history)
                        
                        if fingertip:
                            # Draw fingertip cursor
                            cv2.circle(display, fingertip, 15, (0, 0, 255), 2)
                            cv2.circle(display, fingertip, 5, (255, 255, 255), -1)
                            
                            # Convert to screen coordinates
                            screen_x, screen_y = warped_to_screen(fingertip)
                            
                            # Move mouse to fingertip position
                            mouse.position = (screen_x, screen_y)
                            
                            # ============================================
                            # TOUCH DETECTION
                            # ============================================
                            
                            # Check if finger is in touch zone
                            if fingertip[1] > touch_line_y:
                                # Visual feedback for touching
                                cv2.circle(display, fingertip, 25, (0, 255, 255), 3)
                                
                                if not is_touching:
                                    # Just started touching
                                    is_touching = True
                                    touch_start_time = time.time()
                                    touch_position = (screen_x, screen_y)
                                    hold_triggered = False
                                    
                                else:
                                    # Currently touching - check for hold
                                    touch_duration = time.time() - touch_start_time
                                    
                                    # Draw progress bar for hold
                                    bar_width = 60
                                    bar_height = 8
                                    progress = min(touch_duration / hold_threshold, 1.0)
                                    filled_width = int(bar_width * progress)
                                    
                                    cv2.rectangle(display, 
                                                (fingertip[0] - bar_width//2, fingertip[1] - 40),
                                                (fingertip[0] + bar_width//2, fingertip[1] - 32),
                                                (100, 100, 100), -1)
                                    cv2.rectangle(display, 
                                                (fingertip[0] - bar_width//2, fingertip[1] - 40),
                                                (fingertip[0] - bar_width//2 + filled_width, fingertip[1] - 32),
                                                (0, 255, 255), -1)
                                    
                                    # Trigger right click on hold
                                    if touch_duration > hold_threshold and not hold_triggered:
                                        mouse.click(Button.right)
                                        print(f"👉 Hold right click at ({screen_x}, {screen_y})")
                                        hold_triggered = True
                                    
                                    cv2.putText(display, f"HOLD: {touch_duration:.1f}s", 
                                              (fingertip[0]-40, fingertip[1]-55),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            else:
                                # Finger not touching - check for tap
                                if is_touching and touch_cooldown <= 0 and not hold_triggered:
                                    # This was a tap (quick touch and release)
                                    current_time = time.time()
                                    
                                    # Check for double tap
                                    if (current_time - last_tap_time < double_tap_threshold and
                                        last_tap_position and
                                        abs(screen_x - last_tap_position[0]) < double_tap_distance and
                                        abs(screen_y - last_tap_position[1]) < double_tap_distance):
                                        # Double tap!
                                        mouse.double_click(Button.left)
                                        print(f"👆👆 Double click at ({screen_x}, {screen_y})")
                                        cv2.putText(display, "DOUBLE TAP!", (fingertip[0]-50, fingertip[1]-70),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                        last_tap_time = 0  # Reset
                                    else:
                                        # Single tap
                                        mouse.click(Button.left)
                                        print(f"👆 Tap at ({screen_x}, {screen_y})")
                                        cv2.putText(display, "TAP!", (fingertip[0]-30, fingertip[1]-70),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        last_tap_time = current_time
                                        last_tap_position = (screen_x, screen_y)
                                    
                                    touch_cooldown = touch_cooldown_max
                                
                                is_touching = False
                                hold_triggered = False
                            
                            # Show coordinates
                            cv2.putText(display, f"({screen_x}, {screen_y})", 
                                      (fingertip[0]-30, fingertip[1]-45),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                else:
                    # No hand detected or two-finger mode active
                    if not two_fingers_detected:
                        cv2.putText(display, "Show your finger", 
                                  (AR[1]//2 - 100, AR[0]//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        is_touching = False
                        fingertip_history = []
                
                # Update cooldowns
                if touch_cooldown > 0:
                    touch_cooldown -= 1
                if right_click_cooldown > 0:
                    right_click_cooldown -= 1
                    right_click_triggered = False
                
                # Show instructions
                instructions = [
                    "👆 Tap: Left click",
                    "👆 Hold: Right click",
                    "👆👆 Double tap: Double click",
                    "✌️ Two fingers: Right click"
                ]
                
                for i, instruction in enumerate(instructions):
                    cv2.putText(display, instruction, (10, 90 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.putText(display, f"Sensitivity: {touch_sensitivity}px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Touchscreen', display)
                cv2.putText(calibration_display, "✅ TOUCHSCREEN ACTIVE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                cv2.putText(calibration_display, "⚠️ TRACKING LOST", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', calibration_display)
        
        # ============================================
        # KEYBOARD CONTROLS
        # ============================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('+') and calibration_done:
            touch_sensitivity = min(100, touch_sensitivity + 5)
            print(f"Sensitivity: {touch_sensitivity}px (easier to touch)")
        elif key == ord('-') and calibration_done:
            touch_sensitivity = max(10, touch_sensitivity - 5)
            print(f"Sensitivity: {touch_sensitivity}px (harder to touch)")
        elif key == ord('c'):
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
    print("\nTouchscreen terminated")

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
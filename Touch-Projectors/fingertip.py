# Touch-Projectors Skin color - Fingertip Detection with Perspective Transform

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import math

# Initialize video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
time.sleep(1.1)

# Initialize variables
mouse = Controller()
check = False
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)  # Height, Width of projection area
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

# Fingertip detection parameters
min_hand_area = 5000  # Minimum area to consider as hand
max_hand_area = 30000  # Maximum area for hand
touch_threshold = 30  # Distance from bottom to consider as "touch" (pixels)
click_cooldown = 0  # Frames between clicks
click_cooldown_max = 10  # Max cooldown frames
touch_active = False  # Track if currently touching

# Skin color range in HSV (adjust for your skin tone)
# These are default values - you can calibrate with 'c' key
lower_skin = (0, 20, 70)    # Lower bound for skin (Hue, Saturation, Value)
upper_skin = (20, 255, 255)  # Upper bound for skin

# For tracking fingertip position
prev_fingertip = None
fingertip_history = []  # For smoothing

def adjust_gamma(image, gamma):
    """Apply gamma correction to image"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def draw_circle(event, x, y, flags, param):
    """Mouse callback function for selecting perspective points"""
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        print(f"Point {pointIndex + 1} selected: ({x}, {y})")
        pointIndex = pointIndex + 1

def show_window():
    """Display window for perspective point selection"""
    while True:
        cv2.imshow('img', img)

        if pointIndex == 4:
            print("All 4 points selected!")
            break

        if cv2.waitKey(20) & 0xFF == 27:
            break

def get_persp(image, pts):
    """Apply perspective transform to get top-down view"""
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts)
    warped = cv2.warpPerspective(image, Map, (AR[1], AR[0]))
    return warped

def get_hand_contour(mask):
    """Find the largest contour that could be a hand"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    
    # Check if area is within hand range
    if min_hand_area < area < max_hand_area:
        return max_contour
    return None

def detect_fingertips(contour, hull_indices, defects):
    """
    Detect fingertips using convexity defects
    Returns list of fingertip points
    """
    fingertips = []
    
    if defects is None:
        return fingertips
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        
        # Get points
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # Calculate distances
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # Calculate angle using cosine law
        if b * c > 0:  # Avoid division by zero
            # Calculate angle in degrees
            angle_cos = (b**2 + c**2 - a**2) / (2 * b * c)
            # Clamp to valid range
            angle_cos = max(-1.0, min(1.0, angle_cos))
            angle = math.acos(angle_cos) * 180 / math.pi
            
            # Finger tips have angle < 90 degrees and depth > threshold
            depth_threshold = 5000  # Minimum depth to consider
            if angle < 90 and d > depth_threshold:
                fingertips.append(end)
                fingertips.append(start)
    
    # Remove duplicates by converting to set of tuples and back
    unique_fingertips = list(set([(p[0], p[1]) for p in fingertips]))
    return [tuple(p) for p in unique_fingertips]

def get_fingertip(contour):
    """
    Main function to detect the highest fingertip (index finger)
    """
    if contour is None or len(contour) < 10:  # Need enough points
        return None
    
    # Calculate convex hull and defects
    hull = cv2.convexHull(contour, returnPoints=False)
    
    if hull is not None and len(hull) > 3:  # Need at least 3 points for convexity defects
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is not None:
            # Detect all fingertips
            fingertips = detect_fingertips(contour, hull, defects)
            
            if fingertips:
                # Find the highest fingertip (smallest y-coordinate) - likely index finger
                # Also sort by x to get leftmost if multiple at same height
                highest_fingertip = min(fingertips, key=lambda p: (p[1], p[0]))
                return highest_fingertip
    
    # Fallback: use the highest point of contour (simpler but less accurate)
    highest_point = tuple(contour[contour[:, :, 1].argmin()][0])
    return highest_point

def smooth_fingertip(new_fingertip, history, max_history=5):
    """Apply moving average smoothing to fingertip position"""
    if new_fingertip is None:
        return None
    
    history.append(new_fingertip)
    if len(history) > max_history:
        history.pop(0)
    
    if len(history) > 1:
        # Calculate average
        avg_x = int(sum(p[0] for p in history) / len(history))
        avg_y = int(sum(p[1] for p in history) / len(history))
        return (avg_x, avg_y)
    else:
        return new_fingertip

def is_touching_screen(fingertip, touch_zone_y, threshold=touch_threshold):
    """
    Detect if fingertip is touching the screen based on vertical position
    """
    if fingertip is None:
        return False
    
    # Check if fingertip is near the bottom of the screen (touch zone)
    # For projection, touching happens when finger is close to the surface
    # which appears near the bottom of the warped image
    return fingertip[1] > (touch_zone_y - threshold)

def detect_touch_gesture(fingertip, prev_pos, threshold=5):
    """
    Alternative: Detect touch based on finger stopping movement
    (like holding finger still to click)
    """
    if prev_pos is None or fingertip is None:
        return False
    
    distance = math.sqrt((fingertip[0] - prev_pos[0])**2 + 
                        (fingertip[1] - prev_pos[1])**2)
    
    return distance < threshold

def calibrate_skin_color(roi_hsv):
    """Calibrate skin color range from selected ROI"""
    if roi_hsv.size == 0:
        return lower_skin, upper_skin
    
    # Calculate average HSV in ROI
    avg_hue = np.mean(roi_hsv[:, :, 0])
    avg_sat = np.mean(roi_hsv[:, :, 1])
    avg_val = np.mean(roi_hsv[:, :, 2])
    
    # Set new skin color range with some margin
    new_lower = (max(0, avg_hue - 10), 
                 max(20, avg_sat - 30), 
                 max(50, avg_val - 30))
    new_upper = (min(180, avg_hue + 10), 
                 min(255, avg_sat + 50), 
                 min(255, avg_val + 30))
    
    return new_lower, new_upper

# Setup window for perspective point selection
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Select points in order: Top Left, Top Right, Bottom Right, Bottom Left')
_, img = cap.read()
show_window()

# Main detection loop
print("\n" + "="*60)
print("FINGERTIP DETECTION MODE - Touch Projector")
print("="*60)
print("Instructions:")
print("  - Show your hand to the camera")
print("  - Point with your index finger")
print("  - Touch the projection surface (bottom of warped view) to click")
print("\nControls:")
print("  ESC - Exit")
print("  c - Calibrate skin color (draw rectangle first)")
print("  r - Reset to default skin color range")
print("  + / - - Adjust touch threshold")
print("  t - Toggle touch detection mode (position-based/stillness-based)")
print("="*60)

detection_method = 0  # 0: Position-based touch, 1: Stillness-based click
method_names = ["Position-based Touch", "Stillness-based Click"]
touch_zone_y = int(AR[0] * 0.8)  # Bottom 20% of screen is touch zone
calibration_mode = False
calibration_roi = None
stillness_counter = 0
click_triggered = False

while True:
    # Read frame
    _, frame = cap.read()
    if frame is None:
        print("Failed to grab frame")
        break

    # Apply perspective transform
    warped = get_persp(frame, pts)
    
    # Create a copy for display
    display = warped.copy()
    
    # Calibration mode overlay
    if calibration_mode and calibration_roi is not None:
        x, y, w, h = calibration_roi
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(display, "Place hand in BLUE rectangle and press 'c'", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Preprocessing for skin detection
    blurred = cv2.GaussianBlur(display, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.medianBlur(skin_mask, 5)
    
    # Get hand contour
    hand_contour = get_hand_contour(skin_mask)
    
    # Detect fingertip
    fingertip = None
    if hand_contour is not None:
        # Draw hand contour
        cv2.drawContours(display, [hand_contour], -1, (0, 255, 0), 2)
        
        # Get raw fingertip position
        raw_fingertip = get_fingertip(hand_contour)
        
        # Apply smoothing
        fingertip = smooth_fingertip(raw_fingertip, fingertip_history)
        
        if fingertip:
            # Draw fingertip
            cv2.circle(display, fingertip, 10, (0, 0, 255), -1)
            cv2.putText(display, "Fingertip", 
                       (fingertip[0] - 30, fingertip[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Convert to screen coordinates
            screen_width, screen_height = pyautogui.size()
            x_percent = (fingertip[0] / AR[1]) * 100
            y_percent = (fingertip[1] / AR[0]) * 100
            screen_x = int((screen_width * x_percent) / 100)
            screen_y = int((screen_height * y_percent) / 100)
            
            # Move mouse to fingertip position (always)
            mouse.position = (screen_x, screen_y)
            
            # Touch/Click detection based on selected method
            touch_detected = False
            
            if detection_method == 0:
                # Position-based touch (fingertip near bottom of screen)
                touch_detected = is_touching_screen(fingertip, touch_zone_y)
                
                if touch_detected and not touch_active and click_cooldown <= 0:
                    # Just touched down - perform click
                    mouse.click(Button.left)
                    print(f"\rTOUCH CLICK at ({screen_x}, {screen_y})", end="")
                    touch_active = True
                    click_cooldown = click_cooldown_max
                elif not touch_detected:
                    touch_active = False
                    
            else:
                # Stillness-based click (finger held still)
                if detect_touch_gesture(fingertip, prev_fingertip, threshold=5):
                    stillness_counter += 1
                    if stillness_counter > 5 and not click_triggered and click_cooldown <= 0:
                        mouse.click(Button.left)
                        print(f"\rSTILLNESS CLICK at ({screen_x}, {screen_y})", end="")
                        click_triggered = True
                        click_cooldown = click_cooldown_max
                else:
                    stillness_counter = 0
                    click_triggered = False
            
            # Update previous fingertip
            prev_fingertip = fingertip
    
    # Update cooldown
    if click_cooldown > 0:
        click_cooldown -= 1
    
    # Draw touch zone
    cv2.line(display, (0, touch_zone_y), (AR[1], touch_zone_y), (255, 255, 0), 2)
    cv2.putText(display, "Touch Zone", (10, touch_zone_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Create skin mask display (colorized for better visualization)
    skin_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    skin_display[skin_mask == 255] = [0, 255, 0]  # Green for detected skin
    
    # Add status text
    hand_status = "Hand Detected" if hand_contour is not None else "No Hand"
    hand_color = (0, 255, 0) if hand_contour is not None else (0, 0, 255)
    cv2.putText(display, hand_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
    
    touch_status = "TOUCHING" if touch_active else ""
    if touch_active:
        cv2.putText(display, touch_status, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add info text
    info_lines = [
        f"Method: {method_names[detection_method]}",
        f"Touch Zone Y: {touch_zone_y}",
        f"Cooldown: {click_cooldown}",
        f"Skin Range: H:{lower_skin[0]}-{upper_skin[0]}"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(display, line, (AR[1] - 300, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw the selected points on original frame
    for i, point in enumerate(pts):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
    
    # Display all windows
    cv2.imshow('Original with Points', frame)
    cv2.imshow('Fingertip Detection', display)
    cv2.imshow('Skin Mask', skin_display)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('t'):  # Toggle detection method
        detection_method = 1 - detection_method
        print(f"\nSwitched to {method_names[detection_method]}")
    elif key == ord('c'):  # Calibration
        if not calibration_mode:
            calibration_mode = True
            calibration_roi = (200, 100, 200, 200)  # x, y, width, height
            print("\nCalibration mode: Place hand in blue rectangle and press 'c' again")
        else:
            # Calibrate skin color from ROI
            if calibration_roi is not None:
                x, y, w, h = calibration_roi
                roi = hsv[y:y+h, x:x+w]
                lower_skin, upper_skin = calibrate_skin_color(roi)
                print(f"\nCalibrated skin range: Lower{lower_skin}, Upper{upper_skin}")
                calibration_mode = False
                calibration_roi = None
    elif key == ord('r'):  # Reset skin range
        lower_skin = (0, 20, 70)
        upper_skin = (20, 255, 255)
        print("\nReset to default skin color range")
    elif key == ord('+'):  # Increase touch threshold
        touch_threshold = min(100, touch_threshold + 5)
        touch_zone_y = AR[0] - touch_threshold
        print(f"\nTouch threshold: {touch_threshold}")
    elif key == ord('-'):  # Decrease touch threshold
        touch_threshold = max(10, touch_threshold - 5)
        touch_zone_y = AR[0] - touch_threshold
        print(f"\nTouch threshold: {touch_threshold}")

# Cleanup
cv2.destroyAllWindows()
cap.release()
print("\nProgram terminated")
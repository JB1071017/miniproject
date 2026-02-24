# Touch-Projectors - Brightness Detection with Perspective Transform
# Simplified version of test2.py with improved brightness detection and user controls
import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time

# Initialize video capture
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
# cap = cv2.VideoCapture(0)  # Use this for webcam
time.sleep(1.1)

# Initialize variables
mouse = Controller()
check = False
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)  # Aspect ratio of your area
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

# Brightness detection parameters
brightness_threshold = 200  # Pixels brighter than this are detected (0-255)
min_area = 50  # Minimum size of bright area to detect
max_area = 1000  # Maximum size to avoid detecting large bright areas
use_adaptive = False  # Use adaptive thresholding (better for varying light)

def draw_circle(event, x, y, flags, param):
    """Mouse callback for selecting perspective points"""
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
    """Apply perspective transform"""
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts)
    warped = cv2.warpPerspective(image, Map, (AR[1], AR[0]))
    return warped

def detect_bright_spots(image):
    """
    Detect bright spots in the image
    Returns: mask of bright regions, list of contours
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_adaptive:
        # Adaptive thresholding - good for uneven lighting
        mask = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    else:
        # Simple threshold - good for controlled lighting
        _, mask = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            valid_contours.append(c)
    
    return mask, valid_contours

def get_brightness_stats(image, contour):
    """Get brightness statistics for a detected region"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Calculate mean brightness in the region
    mean_val = cv2.mean(gray, mask=mask)[0]
    max_val = np.max(gray[mask == 255])
    
    return mean_val, max_val

# Setup perspective point selection
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Select points in order: Top Left, Top Right, Bottom Right, Bottom Left')
_, img = cap.read()
show_window()

print("\n" + "="*50)
print("BRIGHTNESS DETECTION MODE")
print("="*50)
print("Controls:")
print("  ESC - Exit")
print("  + / - : Increase/Decrease brightness threshold")
print("  a / s : Increase/Decrease minimum area")
print("  t : Toggle adaptive thresholding")
print("  r : Reset to defaults")
print("="*50)

while True:
    # Read frame
    _, frame = cap.read()
    if frame is None:
        print("Failed to grab frame")
        break

    # Apply perspective transform
    warped = get_persp(frame, pts)

    # Detect bright spots
    mask, contours = detect_bright_spots(warped)

    # Create visualization images
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Color the mask for better visualization
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[mask == 255] = [0, 255, 0]  # Green for detected areas

    # Process detections
    check = False
    if len(contours) > 0:
        # Find the largest valid bright spot
        c = max(contours, key=cv2.contourArea)
        
        # Get contour properties
        area = cv2.contourArea(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Calculate centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (int(x), int(y))
        
        # Get brightness statistics
        mean_brightness, max_brightness = get_brightness_stats(warped, c)
        
        check = True
        print(f"\rDetected: Pos({center[0]}, {center[1]}) | Area:{area:.0f} | Brightness:{max_brightness:.0f}", end="")

        # Draw detection
        cv2.circle(warped, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(warped, center, 5, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(warped, f"{max_brightness:.0f}", 
                   (center[0] - 20, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert to screen coordinates
        screen_width, screen_height = pyautogui.size()
        x_percent = (center[0] / AR[1]) * 100
        y_percent = (center[1] / AR[0]) * 100
        screen_x = int((screen_width * x_percent) / 100)
        screen_y = int((screen_height * y_percent) / 100)

        # Perform mouse click
        mouse.press(Button.left)
        time.sleep(0.05)
        mouse.release(Button.left)

    # Release mouse if no detection
    if not check:
        mouse.release(Button.left)
        print("\rNo bright spot detected" + " " * 30, end="")

    # Draw perspective points on original frame
    for i, point in enumerate(pts):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    # Add parameter info to display
    info_text = [
        f"Threshold: {brightness_threshold}",
        f"Min Area: {min_area}",
        f"Adaptive: {'ON' if use_adaptive else 'OFF'}",
        f"Detected: {'YES' if check else 'NO'}"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(warped, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show all windows
    cv2.imshow('Original with Points', frame)
    cv2.imshow('Warped Detection', warped)
    cv2.imshow('Bright Areas', mask_colored)
    cv2.imshow('Grayscale', gray)

    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+'):  # Increase threshold
        brightness_threshold = min(255, brightness_threshold + 5)
        print(f"\nThreshold: {brightness_threshold}")
    elif key == ord('-'):  # Decrease threshold
        brightness_threshold = max(0, brightness_threshold - 5)
        print(f"\nThreshold: {brightness_threshold}")
    elif key == ord('a'):  # Increase min area
        min_area += 10
        print(f"\nMin Area: {min_area}")
    elif key == ord('s'):  # Decrease min area
        min_area = max(10, min_area - 10)
        print(f"\nMin Area: {min_area}")
    elif key == ord('t'):  # Toggle adaptive
        use_adaptive = not use_adaptive
        print(f"\nAdaptive: {'ON' if use_adaptive else 'OFF'}")
    elif key == ord('r'):  # Reset
        brightness_threshold = 200
        min_area = 50
        use_adaptive = False
        print("\nParameters reset")

# Cleanup
cv2.destroyAllWindows()
cap.release()
print("\nProgram terminated")
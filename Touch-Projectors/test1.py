# Touch-Projectors - Brightness Detection with Perspective Transform

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time

# Initialize video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
time.sleep(1.1)

# Initialize variables
mouse = Controller()
check = False
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])
a = 0
b = 0

# Brightness detection parameters
brightness_threshold = 200  # Threshold for bright pixels (0-255)
min_bright_area = 50  # Minimum area of bright region to detect
blur_size = (5, 5)  # Gaussian blur kernel size

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

def detect_bright_spots(image, threshold=200, min_area=50):
    """
    Detect bright spots in image using multiple methods
    
    Args:
        image: Input BGR image
        threshold: Brightness threshold (0-255)
        min_area: Minimum contour area to consider
    
    Returns:
        mask: Binary mask of bright regions
        contours: List of contours found
    """
    # Method 1: Using Value channel from HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    
    # Method 2: Using grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 3: Using LAB color space (L channel for lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    
    # Combine methods (you can choose one or combine them)
    # For pure brightness, value_channel or gray works best
    brightness = value_channel  # Using Value channel from HSV
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(brightness, blur_size, 0)
    
    # Apply threshold to get bright regions
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return mask, valid_contours

def detect_bright_spots_adaptive(image, min_area=50):
    """
    Detect bright spots using adaptive thresholding
    Better for varying lighting conditions
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_size, 0)
    
    # Adaptive thresholding
    mask = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return mask, valid_contours

def get_intensity_profile(image):
    """Create intensity profile for debugging"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a colorized version of intensity for visualization
    intensity_colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Add intensity scale
    height, width = gray.shape
    scale = np.zeros((50, width, 3), dtype=np.uint8)
    for i in range(width):
        val = int(255 * i / width)
        scale[:, i] = [val, val, val]
    
    return intensity_colored, scale

# Setup window for perspective point selection
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Select points in order: Top Left, Top Right, Bottom Right, Bottom Left')
_, img = cap.read()
show_window()

# Main detection loop
print("\nBrightness Detection Mode")
print("Controls:")
print("  ESC - Exit")
print("  t - Toggle detection method")
print("  + / - - Increase/Decrease brightness threshold")
print("  a - Increase min area")
print("  s - Decrease min area")
print("  r - Reset parameters")

detection_method = 0  # 0: Simple threshold, 1: Adaptive
method_names = ["Simple Threshold", "Adaptive Threshold"]

while True:
    # Read frame
    _, frame = cap.read()
    if frame is None:
        print("Failed to grab frame")
        break

    # Apply perspective transform
    warped = get_persp(frame, pts)

    # Detect bright spots based on selected method
    if detection_method == 0:
        mask, contours = detect_bright_spots(warped, brightness_threshold, min_bright_area)
    else:
        mask, contours = detect_bright_spots_adaptive(warped, min_bright_area)

    # Create visualization of brightness/intensity
    intensity_view, scale = get_intensity_profile(warped)

    # Process detections
    check = False
    if len(contours) > 0:
        # Find the largest bright spot
        c = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Calculate centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            a, b = center[0], center[1]
        else:
            center = (int(x), int(y))
            a, b = center
        
        # Calculate brightness at center (for debugging)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        center_brightness = gray[b, a] if 0 <= b < gray.shape[0] and 0 <= a < gray.shape[1] else 0
        
        check = True
        print(f"Bright spot detected at ({a:.1f}, {b:.1f}) | Brightness: {center_brightness} | Area: {cv2.contourArea(c):.1f}")

        # Draw the detection
        cv2.circle(warped, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(warped, center, 5, (0, 0, 255), -1)
        
        # Add brightness info
        cv2.putText(warped, f"B:{center_brightness}", 
                   (center[0] - 30, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert coordinates to screen position
        screen_width, screen_height = pyautogui.size()
        x_percent = (a / AR[1]) * 100
        y_percent = (b / AR[0]) * 100
        screen_x = int((screen_width * x_percent) / 100)
        screen_y = int((screen_height * y_percent) / 100)

        # Optional: Move mouse
        # mouse.position = (screen_x, screen_y)

        # Perform mouse click
        mouse.press(Button.left)
        time.sleep(0.05)
        mouse.release(Button.left)

    # Release mouse if no detection
    if not check:
        mouse.release(Button.left)

    # Draw the selected points
    for i, point in enumerate(pts):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    # Add info text to displays
    cv2.putText(warped, f"Method: {method_names[detection_method]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(warped, f"Threshold: {brightness_threshold}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(warped, f"Min Area: {min_bright_area}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display all windows
    cv2.imshow('Original with Points', frame)
    cv2.imshow('Warped with Detection', warped)
    cv2.imshow('Bright Spots Mask', mask)
    cv2.imshow('Intensity Map', intensity_view)

    # Keyboard controls
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('t'):  # Toggle method
        detection_method = 1 - detection_method
        print(f"Switched to {method_names[detection_method]}")
    elif key == ord('+') or key == ord('='):  # Increase threshold
        brightness_threshold = min(255, brightness_threshold + 10)
        print(f"Brightness threshold: {brightness_threshold}")
    elif key == ord('-') or key == ord('_'):  # Decrease threshold
        brightness_threshold = max(0, brightness_threshold - 10)
        print(f"Brightness threshold: {brightness_threshold}")
    elif key == ord('a'):  # Increase min area
        min_bright_area += 25
        print(f"Min area: {min_bright_area}")
    elif key == ord('s'):  # Decrease min area
        min_bright_area = max(10, min_bright_area - 25)
        print(f"Min area: {min_bright_area}")
    elif key == ord('r'):  # Reset parameters
        brightness_threshold = 200
        min_bright_area = 50
        detection_method = 0
        print("Parameters reset")

# Cleanup
cv2.destroyAllWindows()
cap.release()
print("Program terminated")
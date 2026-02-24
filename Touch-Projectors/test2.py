#Red light detection with perspective transform and mouse control
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
alpha = 2
mouse = Controller()
gamma = 0.5
check = False
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])
a = 0
b = 0

# Red color HSV ranges (red wraps around hue spectrum)
lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)
lower_red2 = (160, 100, 100)
upper_red2 = (180, 255, 255)

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

def get_hsv_value_at_point(event, x, y, flags, param):
    """Debug function to get HSV values at mouse position"""
    if event == cv2.EVENT_LBUTTONDOWN and 'hsv_frame' in param:
        hsv_value = param['hsv_frame'][y, x]
        print(f"HSV at ({x}, {y}): {hsv_value}")

# Setup window for perspective point selection
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Select points in order: Top Left, Top Right, Bottom Right, Bottom Left')
_, img = cap.read()
show_window()

# Main detection loop
while True:
    # Read frame
    _, frame = cap.read()
    if frame is None:
        print("Failed to grab frame")
        break

    # Apply perspective transform
    warped = get_persp(frame, pts)

    # Preprocessing
    blurred = cv2.GaussianBlur(warped, (5, 5), 0)
    adjusted = adjust_gamma(blurred, gamma)

    # Convert to HSV color space
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    # Create masks for red color (both ranges)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask using morphological operations
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)

    # Find contours in the mask
    cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # Process contours if any are found
    if len(cnts) > 0:
        # Filter contours by area to remove noise
        min_contour_area = 100  # Adjust this value based on your needs
        valid_contours = [c for c in cnts if cv2.contourArea(c) > min_contour_area]

        if valid_contours:
            # Find the largest valid contour
            c = max(valid_contours, key=cv2.contourArea)

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

            # Check minimum radius to filter out small detections
            min_radius = 5  # Adjust this value based on your needs
            if radius > min_radius:
                check = True
                print(f"Red light detected at ({a:.1f}, {b:.1f}) with radius {radius:.1f}")

                # Draw the detection on the warped image
                cv2.circle(warped, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(warped, center, 5, (0, 0, 255), -1)

                # Convert coordinates to screen position
                screen_width, screen_height = pyautogui.size()

                # Calculate percentage of screen
                x_percent = (a / AR[1]) * 100
                y_percent = (b / AR[0]) * 100

                # Convert to actual screen coordinates
                screen_x = int((screen_width * x_percent) / 100)
                screen_y = int((screen_height * y_percent) / 100)

                # Optional: Move mouse to detected position
                # mouse.position = (screen_x, screen_y)

                # Perform mouse click
                mouse.press(Button.left)
                time.sleep(0.05)  # Small delay for the click
                mouse.release(Button.left)
            else:
                check = False
        else:
            check = False
    else:
        check = False

    # Release mouse if no detection
    if not check:
        mouse.release(Button.left)

    # Draw the selected points on the original frame (for reference)
    for i, point in enumerate(pts):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    # Display the frames
    cv2.imshow('Original Frame with Points', frame)
    cv2.imshow('Warped View with Detection', warped)
    cv2.imshow('Red Mask', red_mask)

    # Optional: Show HSV values on mouse click for debugging
    # Uncomment the following lines to enable HSV debugging
    # cv2.imshow('HSV Debug', hsv)
    # cv2.setMouseCallback('HSV Debug', get_hsv_value_at_point, {'hsv_frame': hsv})

    # Check for ESC key press
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('r'):  # Press 'r' to reset gamma
        gamma = 0.5
        print("Gamma reset to 0.5")
    elif key == ord('+') or key == ord('='):  # Press '+' to increase gamma
        gamma = min(3.0, gamma + 0.1)
        print(f"Gamma increased to {gamma:.1f}")
    elif key == ord('-') or key == ord('_'):  # Press '-' to decrease gamma
        gamma = max(0.1, gamma - 0.1)
        print(f"Gamma decreased to {gamma:.1f}")

# Cleanup
cv2.destroyAllWindows()
cap.release()
print("Program terminated")
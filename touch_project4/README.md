# Touchless Wall Interaction System (Computer Vision)

A computer vision based system that allows users to control a computer using **hand gestures** without touching a mouse or keyboard.  
The system detects hand landmarks using **MediaPipe** and converts gestures into **mouse actions** such as cursor movement, clicks, drag, scroll, and drawing.

This project is designed for **touchless wall interaction using a camera and projector setup**.

---

# Features

## Cursor Control
- Cursor movement using **index fingertip tracking**
- Smooth cursor movement using **position smoothing**
- Interaction limited to a **calibrated region**

---

# Mouse Actions

The system supports the following gesture-based interactions:

## Left Click
**Gesture** Index finger only
**Behavior**
- Cursor must remain stable for **1.2 seconds**
- A dwell timer is shown
- Left click is triggered

---

## Double Click
**Gesture** Thumb + Index finger pinch
**Behavior**
- Hold for **1.2 seconds**
- Double click is executed

---

## Right Click
**Gesture** Index + Middle finger
**Behavior**
- Hold gesture stable for **1.4 seconds**
- Right click is triggered

---

## Drag / Text Selection

**Gesture** Index + Middle + Ring fingers up
            Thumb and pinky closed
**Behavior**
- Hold for **1 second**
- Mouse button is pressed down
- Cursor movement performs drag/select

---

## Scroll
**Gesture** Open palm (All four fingers extended)
**Behavior**
- Palm vertical movement controls scrolling
- Up movement → Scroll Up
- Down movement → Scroll Down

---

# Drawing Mode

Drawing mode allows the system to behave like a **virtual pen**.

Toggle drawing mode using the **D key**.

## Start Drawing

**Gesture** Index finger only
**Behavior**
- Hold finger steady for **0.8 seconds**
- Drawing begins (mouse down)

---

## Stop Drawing

Drawing stops when:
- Gesture changes
- Finger leaves region

---

## Instant Click in Drawing Mode
**Gesture** Index + Middle finger
**Behavior**
- Immediate left click
- No dwell delay

---

# System Modes

## Idle Mode
Initial state of the system.

Press **K** to start calibration.

---

## Calibration Mode

User selects **4 points** to define the interaction region.

Point order:

1. Top Left  
2. Top Right  
3. Bottom Right  
4. Bottom Left  

These points are used to create a **perspective transform**.

---

## Interaction Mode

Activated using **S key** after calibration.

Finger positions inside the region are mapped to **screen coordinates**.

---

# Keyboard Controls

| Key | Function |
|----|----|
| **K** | Start calibration |
| **S** | Start interaction |
| **D** | Toggle drawing mode |
| **R** | Reset calibration |
| **Q** | Quit program |

---

# Project Structure
touch_project/
│
├── main.py
├── config.py
├── state.py
├── utils.py
├── calibration.py
├── hand_tracking.py
├── gestures.py
├── drawing_mode.py
├── actions.py
└── README.md

---

# File Description

### main.py
- Main application loop
- Camera processing
- Gesture action control

### config.py
- System parameters
- Camera settings
- Dwell times
- Scroll parameters

### state.py
- Stores runtime states
- Cursor position
- Interaction states

### utils.py
- Helper functions
- Distance calculations
- Smoothing functions
- Homography mapping

### calibration.py
- Calibration point selection
- Region overlay drawing

### hand_tracking.py
- MediaPipe hand detection
- Landmark extraction

### gestures.py
- Gesture recognition logic

### drawing_mode.py
- Drawing behavior
- Dwell progress display
- Mode information display

### actions.py
- Cursor movement interpolation

---

# Requirements

Install the following Python packages:

pip install opencv-python
pip install mediapipe==0.10.21
pip install numpy
pip install pyautogui


---

# Running the Project

### 1. Clone or download the project

### 2. Create a virtual environment
python -m venv venv


### 3. Activate environment
Windows
venv\Scripts\activate


### 4. Install dependencies
pip install opencv-python mediapipe==0.10.21 numpy pyautogui


### 5. Run the program
python main.py


---

# System Workflow

1. Start program
2. Press **K** for calibration
3. Select 4 calibration points
4. Press **S** to start interaction
5. Use hand gestures to control the system
6. Press **D** to enable drawing mode

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

---

# Applications

This system can be used for:

- Interactive wall displays
- Smart classrooms
- Touchless presentation control
- Virtual whiteboards
- Assistive technology
- Public interactive systems

---

# Future Improvements

Possible enhancements include:

- Multi-hand tracking
- Gesture customization
- Gesture training with AI models
- Better cursor stabilization
- Gesture-based zoom
- GUI control panel
- Dynamic calibration

---

# Author

Computer Vision Mini Project  
Touchless Interaction using Hand Gestures
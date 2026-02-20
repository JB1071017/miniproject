import cv2
import mediapipe as mp
import fitz  # PyMuPDF
import tkinter as tk
from PIL import Image, ImageTk

# ========== PDF SETUP ==========
PDF_PATH = "sample.pdf"
doc = fitz.open(PDF_PATH)
page_index = 0

def render_page(index):
    page = doc.load_page(index)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.resize((600, 800))
    return ImageTk.PhotoImage(img)

# ========== UI SETUP ==========
root = tk.Tk()
root.title("Finger Controlled PDF Viewer")

canvas = tk.Label(root)
canvas.pack()

current_image = render_page(page_index)
canvas.config(image=current_image)

# ========== HAND TRACKING ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_y = None
SCROLL_THRESHOLD = 25  # finger movement sensitivity

def update():
    global page_index, prev_y, current_image

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        index_tip = hand.landmark[8]  # index finger tip

        h, w, _ = frame.shape
        y = int(index_tip.y * h)

        if prev_y is not None:
            diff = y - prev_y

            if diff > SCROLL_THRESHOLD:
                if page_index < len(doc) - 1:
                    page_index += 1
                    current_image = render_page(page_index)
                    canvas.config(image=current_image)

            elif diff < -SCROLL_THRESHOLD:
                if page_index > 0:
                    page_index -= 1
                    current_image = render_page(page_index)
                    canvas.config(image=current_image)

        prev_y = y

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()
        return

    root.after(10, update)

update()
root.mainloop()

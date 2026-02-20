import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print(ret)

    if not ret:
        print("Camera frame not received")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

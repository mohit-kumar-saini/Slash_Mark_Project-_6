import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

cascade_path = "haarcascade_russian_plate_number.xml"
plate_detector = None

if os.path.exists(cascade_path):
    plate_detector = cv2.CascadeClassifier(cascade_path)
    print("Plate detector loaded")
else:
    print("Plate detector not found, running without detection")

model_path = "character_recognition_model.h5"
char_model = None

if os.path.exists(model_path):
    char_model = load_model(model_path)
    print("Character recognition model loaded")
else:
    print("OCR model not found, skipping character recognition")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(
            frame,
            "ALPR System Running",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("License Plate Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import pygame
from imutils import face_utils

pygame.mixer.init()

sound = pygame.mixer.Sound("ping.mp3")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0) #this is index 0 ie the default camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

blink_threshold = 100
blink_counter = 0
blinking_detected = False

mouth_x = np.inf
mouth_y = np.inf
mouth_w = np.inf
mouth_h = np.inf

min_mouth_x = np.inf
min_mouth_y = np.inf
min_mouth_w = np.inf
min_mouth_h = np.inf

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # extract region of interest around the mouth
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0 , 0), 5)
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 5, minSize=(50, 50))

        mouth_roi = gray[y+h//2:y+h, x:x+w]
#       #mouth_roi = face_utils.shape_to_np(mouth_roi)

        mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
        _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY)
        # counting the number of white pixels
        white_pixels = np.sum(mouth_thresh == 255)

        if white_pixels > 75000:
            print("Yawning")
        else:
            continue
        for (x, y, w, h) in mouth:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if abs(x - mouth_x) > 10 or abs(y - mouth_y) > 10 or abs(w - mouth_w) > 10 or abs(h - mouth_h) > 10:
                print("Yawn")
            min_mouth_x = min(x, mouth_x)
            min_mouth_y = min(y, mouth_y)
            min_mouth_w = min(w, mouth_w)
            min_mouth_h = min(h, mouth_h)

        print(mouth_roi)

        for (e_x, e_y, e_w, e_h) in eyes:
            eye_roi = face_roi[e_y:e_y + e_h, e_x:e_x + e_w]

            # apply the threshold around the region
            _, eye_thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
            # counting the number of white pixels ie. open eyes
            white_pixels = np.sum(eye_thresh == 255)

            if blink_counter > 5:
                sound.play()
                print("Asleep")

            if len(eyes) >= 2:
                cv2.putText(frame, "Eyes open", (70,70),
                    cv2.FONT_ITALIC, 3,
                    (0,255,0), 2)
                blink_counter = 0
            else:
                cv2.putText(frame, "Eyes shut", (70,70),
                    cv2.FONT_ITALIC, 3,
                    (0,255,0),2)
                blink_counter += 1

            cv2.rectangle(frame, (x + e_x, y + e_y), (x + e_x + e_w, y + e_y + e_h), (0, 255, 0), 2)


    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
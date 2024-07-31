import cv2
import numpy as np
import pygame

pygame.mixer.init()

sound = pygame.mixer.Sound("ping.mp3")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1) #this is index 0 ie the default camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

is_drinking = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # extract region of interest around the mouth
        mouth_roi = gray[y+h//2:y+h, x:x+w]
        # apply the threshold around the region
        _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY)
        # counting the number of white pixels
        white_pixels = np.sum(mouth_thresh == 255)

        if white_pixels > 75000:
            if not is_drinking:
                print("Drinking coffee detected!")
                is_drinking = True
                sound.play()
            else:
                is_drinking = False

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y + h // 2), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
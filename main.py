import cv2
import numpy as np
import pygame

pygame.mixer.init()

sound = pygame.mixer.Sound("ping.mp3")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0) #this is index 0 ie the default camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

blink_threshold = 100000000
blink_counter = 0
blinking_detected = False

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
        eyes = eye_cascade.detectMultiScale(face_roi)

        for (e_x, e_y, e_w, e_h) in eyes:
            eye_roi = face_roi[e_y:e_y + e_h, e_x:e_x + e_w]

            # apply the threshold around the region
            _, eye_thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
            # counting the number of white pixels ie. open eyes
            white_pixels = np.sum(eye_thresh == 255)

            if white_pixels < blink_threshold:
                blink_counter += 1
            else:
                if blink_counter > 5:
                    blinking_detected = True
                    sound.play()
                    print("Blinking!!!")
                blink_counter = 0

            cv2.rectangle(frame, (x + e_x, y + e_y), (x + e_x + e_w, y + e_y + e_h), (0, 255, 0), 2)


    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
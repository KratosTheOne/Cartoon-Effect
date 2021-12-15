# The following script captures the images through a webcam when a human is detected
# Download 'haarcascade_frontalface_default'
# from 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
# install cv2 and time to run the project

import cv2
import time

# Load the Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0
while True:
    # Reading the frame
    _, img = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using library to detect faces through the webcam
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the captured the image
        cv2.imshow('img', img)
        t = time.strftime("%Y-%m-%d_%H-%M-%S")

        print("Image " +t+ " saved")

        file = 'C:\\Users\\Kaustubh Sinha\\PycharmProjects\\Cartoon1\\images'+t+'.jpeg'

        cv2.imwrite(file, img)

        count += 1

    # Use the ESC button on your keyboard to stop the script from taking multiple images
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the videoCapture object
cap.release()

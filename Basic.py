import cv2
import numpy as np


# Image processing
img = cv2.imread('Resources/Lenna.png')
print(img.shape)
# Gray
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(imgGray.shape)
# Blur
imgBlur = cv2.GaussianBlur(img, (5, 5), 10)
# Canny
imgCanny = cv2.Canny(img, 100, 100)

# Resize
imgResize = cv2.resize(img, (300, 300))
# Crop
imgCrop = img[150:400, 100:300]

img = np.zeros((400, 400, 3), np.uint8)
# Rectangle
cv2.rectangle(img, (10, 200), (370, 380), (120, 120, 150), cv2.FILLED)
# Line
cv2.line(img, (80, 80), (300, 10), (200, 40, 20), 5)
# Circle
cv2.circle(img, (200,200), 40, (100, 0, 140))
# Text
cv2.putText(img, 'OpenCV', (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 2, (120, 150, 40), 2)
cv2.imshow('Image', img)
cv2.waitKey(1000)


# Video processing
cap = cv2.VideoCapture('Resources/test.mp4')
while True:
    success, img = cap.read()
    if not success or cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.imshow('Video', img)
    cv2.waitKey(5)


# Webcam processing
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success or cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.imshow('Video', img)
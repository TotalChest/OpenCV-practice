import cv2
import numpy as np


def empty(param):
    pass


cv2.namedWindow('Settings')

cap = cv2.VideoCapture(0)
cv2.createTrackbar('h1', 'Settings', 0, 255, empty)
cv2.createTrackbar('s1', 'Settings', 0, 255, empty)
cv2.createTrackbar('v1', 'Settings', 0, 255, empty)
cv2.createTrackbar('h2', 'Settings', 255, 255, empty)
cv2.createTrackbar('s2', 'Settings', 255, 255, empty)
cv2.createTrackbar('v2', 'Settings', 255, 255, empty)
h_min, h_max = [], []

while True:
    success, img = cap.read()
    if not success or cv2.waitKey(10) & 0xFF == ord('q'):
        print(f'min_range = {h_min}')
        print(f'max_range = {h_max}')
        break

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h1 = cv2.getTrackbarPos('h1', 'Settings')
    s1 = cv2.getTrackbarPos('s1', 'Settings')
    v1 = cv2.getTrackbarPos('v1', 'Settings')
    h2 = cv2.getTrackbarPos('h2', 'Settings')
    s2 = cv2.getTrackbarPos('s2', 'Settings')
    v2 = cv2.getTrackbarPos('v2', 'Settings')

    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    mask = cv2.inRange(imgHSV, h_min, h_max)
    img_with_mask = cv2.bitwise_and(img, img, mask=mask)

    mask = np.stack((mask, mask, mask), axis=2)
    result = np.hstack((mask, img_with_mask))
    cv2.imshow('Settings', result)
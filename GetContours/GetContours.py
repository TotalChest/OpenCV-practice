import cv2
import numpy as np


kernel = np.ones((3, 3))

def getContours(prepare_img):
    contours, _ = cv2.findContours(prepare_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            Perimeter = cv2.arcLength(contour, True)
            cv2.drawContours(img, contour, -1, (20, 100, 200), 2)
            approx = cv2.approxPolyDP(contour, 0.02 * Perimeter, True)
            cv2.drawContours(img, approx, -1, (200, 100, 0), 10)
            cv2.imshow('Contours', img)
            cv2.waitKey(1000)


img = cv2.imread('../Resources/figures.jpg')

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray, 200, 200)
imageDilate = cv2.dilate(imgCanny, kernel, iterations=1)
getContours(imageDilate)

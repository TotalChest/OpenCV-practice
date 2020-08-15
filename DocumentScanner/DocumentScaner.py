import cv2
import numpy as np
from glob import glob


def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 120, 150)
    kernel = np.ones((3,3))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
    imgErode = cv2.erode(imgDilate, kernel, iterations=1)
    return imgErode


def get_contour(prepare_img):
    max_area = 0
    best_approx = None
    contours, _ = cv2.findContours(prepare_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            Perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * Perimeter, True)
            if area > max_area and approx.shape[0] == 4:
                best_approx = approx
                max_area = area
    return best_approx


def reorder(contour):
    contour = contour.reshape((4,2))
    new_contour = np.zeros((4,2), np.float32)
    add = np.sum(contour, axis=1)
    new_contour[0] = contour[np.argmin(add)]
    new_contour[3] = contour[np.argmax(add)]
    diff = np.diff(contour, axis=1)
    new_contour[1] = contour[np.argmin(diff)]
    new_contour[2] = contour[np.argmax(diff)]
    return new_contour

def get_warp(img, contour):
    contour = reorder(contour)
    target = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    transform = cv2.getPerspectiveTransform(contour, target)
    warpped_img = cv2.warpPerspective(img, transform, (500, 500))
    warpped_img = cv2.resize(warpped_img[10: -10, 10: -10], (500, 500))
    return warpped_img


for image in glob('Documents/*.jpg'):
    img = cv2.imread(image)
    img = cv2.resize(img, (500, 500))
    preprocess_img = preprocess(img)

    contour = get_contour(preprocess_img)
    if contour is None:
        continue
    warp_image = get_warp(img, contour)

    preprocess_img = np.stack((preprocess_img,
                               preprocess_img,
                               preprocess_img), axis=2)
    result_image = np.hstack((img, preprocess_img, warp_image))
    cv2.imshow('Document', result_image)
    cv2.waitKey(5000)
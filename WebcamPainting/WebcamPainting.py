import cv2
import numpy as np


all_points = []  # (x, y, color)
colors = {'orange': {'range': [0, 140, 170, 255, 255, 255],
                     'color': [80, 127, 255]},
          'green': {'range': [65, 135, 95, 95, 255, 255],
                    'color': [50, 205, 50]}
          }


def get_contours(prepare_img):
    contours, _ = cv2.findContours(prepare_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 300:
            return cont


def find_color(img, color):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min_range = np.array(colors[color]['range'][0:3])
    max_range = np.array(colors[color]['range'][3:6])
    mask = cv2.inRange(imgHSV, min_range, max_range)
    return get_contours(mask)


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success or cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for key in colors.keys():
        contour = find_color(img, key)
        if contour is not None:
            points = np.array([point[0] for point in contour])
            x, y = contour[np.argmin(points[:, 1])][0]
            all_points.append((x, y, key))

    for x, y, color in all_points:
        cv2.circle(img, (x, y), 5, colors[color]['color'], cv2.FILLED)

    img=img[:,::-1,:]
    cv2.imshow('Painting', img)

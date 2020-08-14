import cv2


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_frontalface_default.xml')

# WebCam processing
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success or cv2.waitKey(10) & 0xFF == ord('q'):
        break

    faces = faceCascade.detectMultiScale(img, 1.1, 4)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

    cv2.imshow('WebCam', img)

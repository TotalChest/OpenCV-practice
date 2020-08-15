import cv2
from glob import glob


numberCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_russian_plate_number.xml')

for image_path in glob('Numbers/*.jpg'):
    img = cv2.imread(image_path)

    numbers = numberCascade.detectMultiScale(img, 1.1, 6)
    for x, y, w, h in numbers:
        number_img = img[y:y+h, x:x+w].copy()
        cv2.rectangle(img, (x, y), (x+w, y+h), (20, 100, 0), 5)
        cv2.putText(img, 'Number', (x+5, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (20, 100, 0), 3)

    cv2.imshow('Number Detection', img)
    if cv2.waitKey(3000) & 0xFF == ord('s') and len(numbers):
        cv2.imwrite(f'Save/{image_path.split("/")[1]}', number_img)
        cv2.putText(img, 'SAVE', (10, 100), cv2.FONT_HERSHEY_PLAIN, 10, (50, 200, 250), 8)
        cv2.imshow('Number Detection', img)
        cv2.waitKey(500)

import cv2
#variables de classes
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

def detect_face(img):


    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for (x ,y ,w ,h) in face_rects:
        cv2.rectangle(face_img, (x ,y), ( x +w , y +h), (255 ,255 ,255), 3)

    return face_img


def adj_detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 3)

    return face_img
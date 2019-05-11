import numpy as np
import cv2
import imutils
from keras.models import load_model
#variables de classes
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
emotion_model_path = 'models/_mini_XCEPTION.56-0.64.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
def detect_face(img):


    face_img = img.copy()
    target = ["fache", "degouter", "effrayer", "heureux", "triste", "surpris","neutre"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
    for (x ,y ,w ,h) in faces:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = face_img[y:y + h, x:x + w]
        face_crop = cv2.resize((48, 48),face_crop)
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
        result = target[np.argmax(emotion_classifier.predict(face_crop))]
        cv2.putText(face_img, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

    return face_img


def adj_detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5,minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return face_img
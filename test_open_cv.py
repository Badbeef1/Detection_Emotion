import numpy as np
import cv2
import imutils
from keras.models import load_model
from fonction_utilitaires import detect_face
from fonction_utilitaires import adj_detect_face
import matplotlib.pyplot as plt
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
#model entrainé
emotion_model_path = 'models/_mini_XCEPTION.56-0.64.hdf5'
#varibles de classes
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["fache" ,"degouter","effrayer", "heureux", "triste", "surpris",
 "neutre"]

# Tout dépendament de l'ordi la valeur peut changer entre 0 et 1 (webcam #1 ou 2/ avant ou arrière etc.)
cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #trouver le visage dans l'image.
        frame = adj_detect_face(frame)

        #afficher
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

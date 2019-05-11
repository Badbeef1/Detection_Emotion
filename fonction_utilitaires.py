import numpy as np
import cv2
import imutils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
#variables de classes
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
emotion_model_path = 'models/_mini_XCEPTION.56-0.64.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["fache" ,"degouter","effrayer", "heureux", "triste", "surpris",
 "neutre"]
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

#cette fonction recoit en paramètre un image et permet d'afficer le visage
def trouver_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        #boucle qui permet d'afficher les pourcentages de toutes mes catégories (toutes mes émotions)
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                             0.45,
                        (255, 255, 255), 2)

        cv2.putText(frameClone, label, (fX, fY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
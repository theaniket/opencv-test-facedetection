import os
import numpy as np
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("cascade.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current = 0
names_id_mapping = {}
train = []
names = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # print(f"{label} => ${path}")
            if label in names_id_mapping:
                pass
            else:
                names_id_mapping[label] = current
                current += 1

            id_ = names_id_mapping[label]
            pillow_image = Image.open(path).convert("L")
            image_array = np.array(pillow_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                region_of_image = image_array[y: y+h, x: x+w]
                train.append(region_of_image)
                names.append(id_)

with open("labels.pickle", "wb") as file:
    pickle.dump(names_id_mapping, file)

recognizer.train(train, np.array(names))
recognizer.save("trainer.yml")
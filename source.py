import cv2
import numpy as np
import  pickle
import faces_train as ft

face_cascade = cv2.CascadeClassifier('cascade.xml')
eye_cascade = cv2.CascadeClassifier('eye_cascade.xml')
smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
hand_cascade = cv2.CascadeClassifier('hand_cascade.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")


def main():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.resize(frame, (1500, 1500), interpolation=cv2.INTER_AREA)
        flip = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(flip, scaleFactor=1.2, minNeighbors=5)
        eyes = eye_cascade.detectMultiScale(flip, scaleFactor=1.2, minNeighbors=5)
        #hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
        # smiles = smile_cascade.detectMultiScale(flip, scaleFactor=4.5, minNeighbors=1)
        #
        # for(x, y, w, h) in smiles:
        #     color = (0, 230, 0)
        #     stroke = 2
        #     end_cord_x = x + w
        #     end_cord_y = y + h
        #
        #     cv2.rectangle(flip, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # for (x, y, w, h) in hands:
        #     color = (123, 30, 40)
        #     stroke = 2
        #     end_cord_x = x + w
        #     end_cord_y = y + h
        #     cv2.rectangle(flip, (x, y), (end_cord_x, end_cord_y), color, stroke)
        #print(len(faces))
        # if not saved and len(faces) > 0:
        #     cv2.imwrite("new_image.png", flip)
        #     saved = True

        faces_list = list()
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if 45 <= conf <= 85:
                for name, name_id in ft.names_id_mapping.items():
                    if name_id == id_:
                        faces_list.append([[x, y, w, h], str(name)])

        for (cord, name) in faces_list:
            x = cord[0]
            y = cord[1]
            w = cord[2]
            h = cord[3]
            # print(x, y, w, h)
            roi_color = flip[y:y+h, x:x+h]
            img_item = "my_image.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.putText(flip, name, (x, y), 1, 1.0, (255, 255, 255), 2)
            cv2.rectangle(flip, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # for (x, y, w, h) in eyes:
        #     # roi_color = flip[y:y+h, x:x+w]
        #     color = (0, 150, 0)
        #     stroke = 2
        #     end_cord_x = x + w
        #     end_cord_y = y + h
        #
        #     cv2.rectangle(flip, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow('frame', flip)
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(f"{ft.names_id_mapping}")
    main()



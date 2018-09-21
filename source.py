import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascade.xml')
eye_cascade = cv2.CascadeClassifier('eye_cascade.xml')
smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')


def main():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        flip = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(flip, scaleFactor=1.2, minNeighbors=5)
        eyes = eye_cascade.detectMultiScale(flip, scaleFactor=1.2, minNeighbors=5)
        # smiles = smile_cascade.detectMultiScale(flip, scaleFactor=4.5, minNeighbors=1)
        #
        # for(x, y, w, h) in smiles:
        #     color = (0, 230, 0)
        #     stroke = 2
        #     end_cord_x = x + w
        #     end_cord_y = y + h
        #
        #     cv2.rectangle(flip, (x, y), (end_cord_x, end_cord_y), color, stroke)

        for (x, y, w, h) in faces:
            # print(x, y, w, h)
            roi_color = flip[y:y+h, x:x+h]
            img_item = "my_image.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h

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

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



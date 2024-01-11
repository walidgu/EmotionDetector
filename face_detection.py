import threading
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

with open('labels.pickle', 'rb') as handle:
    dictionary_labels = pickle.load(handle)

model = load_model('model.h5', compile=False)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
coord_x, coord_y = 0, 0
x_2, y_2 = 0, 0
counter = 0


text = "Waiting"
# font 
font = cv2.FONT_HERSHEY_SIMPLEX                
# fontScale 
fontScale = 2        
# Red color in BGR 
color = (0, 255, 0)         
# Line thickness of 2 px 
thickness = 4        

while True:
    ret, img = cap.read()

    if ret:

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = gray_image.copy()

        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in face:
            coord_x, coord_y = x, y
            x, y = x-50, y-50
            w, h = w + 50, h + 50
            x_2, y_2 = int((x + w)*1.2), int((y + h)*1.2)
            cv2.rectangle(img, (x, y), (x_2, y_2), (0, 255, 0), 4)
            break
                

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        cropped_image = img_rgb2[coord_y:y_2,coord_x:x_2]



        if counter % 30 == 0:
            try:
                image_x = cv2.resize(cropped_image, (48,48), interpolation = cv2.INTER_AREA)
                img_gray_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
                img_flatten_x = img_gray_x.flatten()
                img_shaped_x = img_flatten_x.reshape(48, 48, 1).astype('float32')
                img_normalized_x = img_shaped_x /255.
                #print(np.array([img_normalized_x]).shape)
                y_perso = model.predict(np.array([img_normalized_x]))
                #print(f'Emotion Detected : {dictionary_labels[np.argmax(y_perso, axis=1).tolist()[0]]}')

                # text 
                text = dictionary_labels[np.argmax(y_perso, axis=1).tolist()[0]].capitalize() 
            except ValueError:
                pass

        print(text)       
        # Using cv2.putText() method 
        # org 
        org = (coord_x - 30, coord_y - 60)  
        cv2.putText(img, text, org, font, fontScale,  
                        color, thickness, cv2.LINE_AA, False) 
        
        counter += 1
        cv2.imshow('video', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
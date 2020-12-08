import cv2
from keras.models import load_model

import numpy as np

model = load_model('model- 12.model')
import matplotlib.pyplot as plt
img_size = 100
print('model loaded')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
label_dict = {0:'mask',1: 'no mask'}
color_dict = {0:(0,255,0),1:(255,0,0)}


def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray)

    for x,y,w,h in face_rects:

        roi = gray[y:y+h,x:x+h]
        resized = cv2.resize(roi, (100,100))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,100,100,1))
        result = model.predict(reshaped)
        label = np.argmin(result,axis=1)[0]
        print(result)
        print(label)
        cv2.rectangle(img, (x, y), (x + w, y + h),color_dict[label], 2)
        cv2.putText(img,label_dict[label],(x-20,y-5),cv2.FONT_ITALIC,0.8,color_dict[label],1)

    return img

#img = cv2.imread(r'C:\Users\Mayank Jha\Desktop\images.jpg')
   

while True:
    success,img = cap.read()
    result = detect_face(img)
    cv2.imshow('final img', result)
    if cv2.waitKey(1) & 0xff == 27:
        break
cv2.destroyAllWindows()
cap.release()
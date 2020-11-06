

import cv2
import numpy as np
import os 
from google_speech import Speech

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Juan', 'Roberto', 'Ilza', 'Z', 'W'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
saludado = [False]* len(names)

genderProto = "deploy_gender.prototxt"
genderModel = "gender_net.caffemodel"
while True:

    ret, img =cam.read()


    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    genderList = ['Hombre', 'Mujer']
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence > 20):
            try:
                if (saludado[id] == False): 

                    os.system("google_speech -l es 'Hola " + names[id] + " '")
                    saludado[id] = True
            except IndexError:
                    print("error")
            id = names[id]
            confidence = str(round(confidence, 2))
        else:
            id = "Desconocido"
            confidence = str(round(confidence, 2))

        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n Cerrando---")
cam.release()
cv2.destroyAllWindows()

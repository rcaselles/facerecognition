
import cv2
import os
from pathlib import Path

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


face_id = input('\n Introduce una id y dale a  <return> ==>  ')

print("\n Mira a la camara y espera ...")
count = 0
aux = 1
auxcond = False
while auxcond == False:
    my_file = Path("dataset/User." + str(face_id) + '.' + str(aux) + ".jpg")
    if my_file.exists() == True:
                aux += 1
    else:
                auxcond = True
                aux -=1
                break
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    

    

    count += 1
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count + aux) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:

        break
    elif count >= (300 + aux ):
         print(str(300+aux))
         print("salgo por el count")
         break

cam.release()
cv2.destroyAllWindows()



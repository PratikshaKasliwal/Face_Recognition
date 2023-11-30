import numpy as np
import cv2 as cv

haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
# features=np.load('features.npy',allow_pickel=True)
# labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img=cv.imread(r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\Faces\val\ben_afflek\2.jpg')
# img=cv.imread(r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\Faces\val\ben_afflek\5.jpg')
# img=cv.imread(r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\Faces\val\Madonna\3.jpg')
img=cv.imread(r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\Faces\val\Elton_John\1.jpg')
#Multiple images are giving wrong result hence we can rely on built in face detection of open cv
#this can be because of less training data
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

faces_react=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_react:
    faces_roi=gray[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f'Label={people[label]}with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

cv.imshow('Detected faces',img)

cv.waitKey(0)


#we read that saved yaml file and made predictions on image
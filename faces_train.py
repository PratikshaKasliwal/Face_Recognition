import os

import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

# p=[]
# for i in os.listdir(r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\face detection images'):
#     p.append(i)
# print(p)


DIR = r'C:\Users\HP\OneDrive\Documents\OpenCV\photos\Faces\train'

haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_react=haar_cascade.detectMultiScale(gray,1.1,4)

            for(x,y,w,h) in faces_react:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
#we are assigning number i.e label to string to make computer asy task
#convert them to numpy array and use it
#whole process need to be done again and again instead
#save this trained model so that we can use this trained model in another directory and file just by using yml file

create_train()
print('Training Done........')
features=np.array(features,dtype='object')
labels=np.array(labels)
# print(f'Length of the features = {len(features)}')
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)

#We have discussde face recognition in open cv where we have built features list and a label list n we train a recognizer on those and we save the model as yml source file
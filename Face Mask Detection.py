#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import dlib
import numpy as np
import pickle




# Loading the trained model 
with open('mask_detector_model.pkl','rb') as file:                 
    model = pickle.load(file)
    
    

# Loading PCA object
with open('pca_obj.pkl','rb') as file:                        
    pca = pickle.load(file)
    
    
    
    

detector = dlib.get_frontal_face_detector()


# Load the shape_predictor.dat file from your current directory
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Read the image from webcam
cap = cv2.VideoCapture(0)     #0 for webcam and video path to read video

names = {0:'Mask',1:'No Mask'}

while(True):
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)
    
    
    faces=detector(frame)
    
    for face1 in faces:
        x = face1.left() # left point
        y = face1.top() # top point
        w = face1.right() # right point
        h = face1.bottom() # bottom point
        
        landmarks = predictor(image=frame,box=face1)

        # cuts the face frame out
        face = frame[y:h, x:w]                                 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        try:
            face = cv2.resize(face,(80,80))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = model.predict(face)
            text = names[int(pred)]
            font = cv2.FONT_HERSHEY_DUPLEX
            red = (0,0,255)
            green=(0,255,0)
            if int(pred)>0:
                cv2.rectangle(frame,(x, y),(w, h), red, 4)
                cv2.rectangle(frame, (x,h),(w,h+22), red, cv2.FILLED)                 # coloring the box red 
                cv2.putText(frame, text, (x+10,h+15), font, 0.8, (255, 255, 255), 1)  # labelling "no mask"
            else:
                cv2.rectangle(frame,(x, y),(w, h), green, 4)
                cv2.rectangle(frame, (x,h),(w,h+22), green, cv2.FILLED)               # coloring the box green
                cv2.putText(frame, text, (x+10,h+15), font, 0.8, (255, 255, 255), 1)  # labelling "mask"    
        except Exception as e:
            pass
        
        

            
    cv2.imshow('face',frame)
    
    # Upon hitting "Esc", program will be terminated.
    if cv2.waitKey(1) == 27:
        break
        
        

        
        
        
# Releasing camera and Closing the opened window
cap.release()
cv2.destroyAllWindows()


# In[ ]:





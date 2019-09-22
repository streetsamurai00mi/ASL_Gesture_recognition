# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:38:28 2019

@author: Anupam Anand
"""

import cv2
from keras.models import load_model, Sequential, Model
import numpy as np

def letter(t):
    
    return(chr(65+t))
    
video = cv2.VideoCapture(0)

model = load_model('static_ges.h5')
f=0
while video.isOpened():
    
    ret,  frame = video.read()
    
    if (ret==True):
        
        cv2.rectangle(frame, (440,250),(590,100),(0,100,0),2 )
        
        image = frame[100:250, 440:590]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(28,28))
        #image = image.flatten()
        image =cv2.flip(image, 1)
        image = image.reshape(1,28,28,1)
        
        text = model.predict_classes(image)
        
        if(cv2.waitKey(10) == ord('c')):
            
            cv2.imwrite('output/image.jpg', image)
        
        
        cv2.putText(frame, letter(text[0]), (300,340), cv2.FONT_HERSHEY_COMPLEX, 1, (0,100,0),3)
        
        
        cv2.imshow('Video Feed', frame);
                
        if (cv2.waitKey(25) == ord('q')):
            
            break
        
    else:
        break
    f=f+1
    
    
    
video.release()
cv2.destroyAllWindows()
        
    
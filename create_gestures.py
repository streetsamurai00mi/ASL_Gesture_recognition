# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:49:31 2020

@author: anupam
"""
import cv2
from collections import deque
import numpy as np
import sqlite3
from sqlite3 import Error
from image_gen import image_aug
    
def camera_fuction(alt_ges):
    
    video = cv2.VideoCapture(0)
    tracker = cv2.TrackerKCF_create()
    hand = False
    initBB = None
    direction_list=''
    print('Started')
    i=0

    pts = deque(maxlen=15)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""
    
    
    while video.isOpened():
            
        ret,  frame = video.read()
        frame = cv2.flip(frame,flipCode=1)
        
        if(ret==True):
            
            if(not hand):
                cv2.rectangle(frame, (412,278),(608,82),(0,100,0),2 )
                
            cv2.putText(frame,str(i),org= (10,10),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,0,255),thickness=1)
            
            if((i>10 and counter<60) or alt_ges):
                cv2.putText(frame,"Press s to record movement",org= (400,300),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,100,255),thickness=1)
                
            elif(i>10 and counter==60):
                cv2.putText(frame,"Press q to quit",org= (400,300),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,100,255),thickness=1)
                
            else:
                cv2.putText(frame,"Take atleast 10 captures!",org= (400,300),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,100,255),thickness=1)
                
            if ((initBB is not None) and counter<60):
            
                success, box = tracker.update(frame)
                
                if success:
                     
                    (x,y,w,h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    centroid = (x+0.5*w, y+0.5*h)
                    
                    pts.appendleft(centroid)
                    counter +=1
                    for i in range(1,len(pts)):
                        
                        if(pts[i-1] is None or pts[i] is None):
                            continue
                        
                        if counter >= 15 and i==1 and pts[-10] is not None:
                            dX = pts[-10][0] - pts[i][0]
                            dY = pts[-10][1] - pts[i][1]
                            direction = ""
                            
                            if(np.abs(dX)>20):
                                direction = "left" if np.sign(dX)==1 else "right"
                                if(len(direction_list)==0):
                                    direction_list += direction[0]
                                elif(direction_list[len(direction_list)-1]!=direction[0]):
                                    direction_list += direction[0] 
                                print('dir')
                                
                                
                            elif(np.abs(dY)>20):
                                direction = "down" if np.sign(dX)==1 else "up"
                                if(len(direction_list)==0):
                                    direction_list += direction[0]
                                elif(direction_list[len(direction_list)-1]!=direction[0]):
                                    direction_list += direction[0] 
                                print('dir2')
                                
            cv2.putText(frame, direction, (10,30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
            
            cv2.imshow('Video Feed', frame);
            
            
            
        if(cv2.waitKey(5)==ord('q')):
            
            if(i<10):
                cv2.putText(frame,"Take atleast 10 captures!",org= (10,10),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,0,255),thickness=1)
                
            else:
                break
        
        elif(cv2.waitKey(5)== ord('c') and (not altges) ):
            
            i+=1
            image = frame[82:278,412:608]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite('C:\\Users\\anupa\\Documents\\college\\Final Project\\ngestures\\train\\hand\\image' + str(i) +'.jpg', image)
            if(i%2==0):
                cv2.imwrite('C:\\Users\\anupa\\Documents\\college\\Final Project\\ngestures\\test\\hand\\image' + str(i) +'.jpg', image)
            
        elif(cv2.waitKey(5)==ord('s') and (i>10 or alt_ges)):
            
            initBB = (412,82,196,196)
            tracker.init(frame, initBB)
            hand = True
            
    print(direction_list)    
    video.release()
    cv2.destroyAllWindows()
    return(direction_list)
    
def sqlite_fun(seq,word,mov):
    
    conn = None
    
    try:
        conn = sqlite3.connect('C:\\Users\\anupa\\gesturedb')
        c = conn.cursor()
        c.execute("INSERT INTO ges_dy VALUES(?,?,?)", (seq,word,mov))
        conn.commit()
    
    except Error as e:
        print(e)
    
    conn.close()
    print("Database Fed!")
    
def create_ges(seq,word,alt_ges):
    
    print("INITIATING GESTURE CREATION")
    
    move=camera_fuction(alt_ges)
    
    if(alt_ges== False):
        image_aug(seq)
    
    sqlite_fun(seq,word,move)
    
    print("Created gesture for Sequence ", seq)
    
    
        
    
    
    
    
    
    
        
        

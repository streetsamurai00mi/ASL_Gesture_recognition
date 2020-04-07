# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:48:09 2020

@author: anupa
"""

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os


def image_aug(seq):
    
    train_datagen = ImageDataGenerator(brightness_range=(0,1),
                                       width_shift_range= (-20,20),
                                       horizontal_flip=True,
                                       zoom_range= 0.2
                                       )
    
    test_datagen = ImageDataGenerator(horizontal_flip=True,
                                      zoom_range=0.2)
    
    train_dg = train_datagen.flow_from_directory(
            'ngestures\\train',
            target_size=(196,196),
            batch_size = 1,
            color_mode = "grayscale",
            class_mode= None,
            save_to_dir="iaug\\train")
    
    test_dg = test_datagen.flow_from_directory(
            'ngestures\\test',
            target_size=(196,196),
            batch_size = 1,
            color_mode = "grayscale",
            class_mode= None,
            save_to_dir="iaug\\test")
    
    i=0
    print("--------------training set---------------")
    
    for batch in train_dg:
        
        if(i<=1000):
            
            i+=1

        else:
            break
        
    feed_to_dir("train",seq)
        
    i=0
    print("--------------test set---------------")
        
    for batch in test_dg:
        
        if(i<=125):
            
            i+=1
            continue
        else:
            break
        
    feed_to_dir('test',seq)
    
    return(25)
        
    
        
def feed_to_dir(path,seq):
    
    images = os.listdir('iaug\\'+path)
    data_dic = {'label':[]}
    
    for i in range(1,785):
        data_dic["pixel"+str(i)]=[]
        
    df = pd.DataFrame(data_dic)
    
    if(path+"_cus.csv" not in os.listdir("C:\\Users\\anupa\\Documents\\college\\Final Project\\sign-language-mnist")):
        
        df.to_csv('C:\\Users\\anupa\\Documents\\college\\Final Project\\sign-language-mnist\\'+path+'_cus.csv', header=True, index=False)
        
    
 
    for i in images:
        
        print("feeding "+i)
        image = cv2.imread('iaug\\'+path+"\\"+i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(28,28))
        data_dic['label'].append(seq)
        k=1
        for row in image:
            for j in row:
                data_dic["pixel"+str(k)].append(j)
                k+=1
    for delimg in images:        
        os.remove('iaug\\'+path+'\\'+delimg)
                
    df2 = pd.DataFrame(data_dic)
                
    df2.to_csv('C:\\Users\\anupa\\Documents\\college\\Final Project\\sign-language-mnist\\'+path+'_cus.csv', header=False, index=False, mode='a')



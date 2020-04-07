import tkinter as tk
import os
from create_gestures import create_ges
import sqlite3
from sqlite3 import Error
#import cv2
#import numpy as np
#import pandas as pd
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import sqlite3
#from collections import deque



def run_gesture_recognition():
    
    window.withdraw()
    os.system('python gesture_with_hand_tracking.py')
    window.deiconify()
    
def run_create_gesture():
    
    conn = None
    row = None 
    seq=0
    
    if(txt_ges.get()!=''):
        try:
            conn = sqlite3.connect('C:\\Users\\anupa\\gesturedb')
            c = conn.cursor()
            row = c.execute("SELECT max(gescode) from ges_dy")
            conn.commit()
        
        except Error as e:
            print(e)
            
        for i in row:
            seq=i[0]
        
        conn.close()
        
        window.withdraw()
        i+=1
        create_ges(seq+1,txt_ges.get(),False)
        window.deiconify()
    
def run_create_altgesture():
    
    conn = None
    row = None 
    seq=0
    
    if(txt_gesalt.get()!=''):
        try:
            conn = sqlite3.connect('C:\\Users\\anupa\\gesturedb')
            c = conn.cursor()
            row = c.execute("SELECT max(gescode) from ges_dy")
            conn.commit()
        
        except Error as e:
            print(e)
            
        for i in row:
            seq=i[0]
        
        conn.close()
    
        window.withdraw()
        create_ges(seq,txt_gesalt.get(),True)
        window.deiconify()
        
def run_train():
    
    window.withdraw()
    os.system('python custom_model_trainer.py')
    window.deiconify()
    

window = tk.Tk()
window.title('Gesture To Voice Translator')
window.geometry('640x420+20+20')
window.resizable(False,False)

frameHead = tk.Frame(master= window, relief = tk.RAISED, bg="red", borderwidth=2)
frameHead.pack()
labelTop=tk.Label(master = frameHead,text='Welcome to Gesture to Voice Translator',font=30)
labelTop.pack()

frameInputLayer = tk.Frame(master= window)
frameInputLayer.pack(fill=tk.X,ipady=5, pady=10)

lbl_txtges= tk.Label(master = frameInputLayer,text='Input new gesture word',font =('Ariel',10), fg = "red")
lbl_txtges.grid(row=0,column=0,sticky='we', padx=16)

txt_ges = tk.Entry(master = frameInputLayer)
txt_ges.grid(row=1,column=0,sticky='we', padx=20)

lbl_txtgesalt= tk.Label(master = frameInputLayer,text='Input alternate gesture word',font =('Ariel',10), fg = "red")
lbl_txtgesalt.grid(row=0,column=1,sticky='we',padx=60)

txt_gesalt = tk.Entry(master = frameInputLayer)
txt_gesalt.grid(row=1,column=1,sticky='we',padx=60)

lbl_ges= tk.Label(master = frameInputLayer,text='Start Gesture Recognition',font =('Ariel',10), fg = "red")
lbl_ges.grid(row=1,column=2,sticky='se',padx=20)

frameMid = tk.Frame(master= window, relief = tk.SUNKEN, borderwidth=2)
frameMid.pack( fill= tk.X,ipady=10)
btn_gs = tk.Button(master = frameMid,text="Add new Gestures", fg="red",
                   height= 5, width=20, command=run_create_gesture)
btn_gs.pack(side = tk.LEFT, padx = 15)

btn_gsrep = tk.Button(master = frameMid,text="Add variant of previous gesture", fg="red",
                   height= 5, width=25, command= run_create_altgesture)
btn_gsrep.pack(side = tk.LEFT,padx = 45)

btn_det = tk.Button(master = frameMid, text="Gesture detection",  fg="blue",
                    height= 5, width=20, command= run_gesture_recognition)
btn_det.pack(side = tk.RIGHT, padx = 8)

btn_train = tk.Button(text="Train", height= 5, width= 25,fg='red', command= run_train)
btn_train.pack(side = tk.LEFT, padx= 16)


labelthree= tk.Label(text='INSTRUCTIONS\n'+'1.Input new gesture word and\n'+'press button to register new gesture.\n'+
                     '2. Use alternate gesture for same\n'+'hand sign but different movement',font=('Courier',10),justify='left', fg = "black")
labelthree.pack(side=tk.RIGHT)
labelfour= tk.Label(text='2. Press q to quit',font=10, fg = "black")
window.mainloop()
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:35:33 2019

@author: Anupam Anand
"""
import pandas as pd
import numpy as np

file_train = pd.read_csv('sign_mnist_train.csv')
X_train =  file_train.iloc[:, 1:]
Y_train = file_train.iloc[:,0]

file_test = pd.read_csv('sign_mnist_test.csv')
X_test = file_test.iloc[:, 1:]
Y_test = file_test.iloc[:,0]

Y_train = pd.get_dummies(Y_train)
Y_train.insert(9,'9', np.zeros(Y_train.shape[0]))
Y_test = pd.get_dummies(Y_test)
Y_test.insert(9,'9', np.zeros(Y_test.shape[0]))

X_train = X_train.values
X_test = X_test.values

X_train_final = X_train.reshape(X_train.shape[0],28,28,1)
X_test_final = X_test.reshape(X_test.shape[0],28,28,1)



#imag = cv.imread('amer_sign2.png')
#cv.imshow('image',imag)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
classifier = Sequential()

classifier.add(Conv2D(filters = 64,kernel_size=(3,3), padding = "same", activation = "relu", input_shape= (28,28,1)))
classifier.add(Conv2D(filters = 64,kernel_size=(3,3), padding = "same", activation = "relu"))

classifier.add(MaxPool2D((2,2)))
classifier.add(Dropout(0.5))

classifier.add(Conv2D(filters = 64,kernel_size=(3,3), padding = "same", activation = "relu"))
classifier.add(Conv2D(filters = 64,kernel_size=(3,3), padding = "same", activation = "relu"))

classifier.add(MaxPool2D((2,2)))
classifier.add(Dropout(0.75))
classifier.add(Flatten())
classifier.add(Dense(units=768, activation = "relu"))
classifier.add(Dropout(0.75))
classifier.add(Dense(units=25, activation="softmax"))

classifier.summary()


classifier.compile(optimizer= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss= 'categorical_crossentropy', metrics=['categorical_accuracy','accuracy'])

hist = classifier.fit(X_train_final, Y_train, batch_size= 100, epochs = 50, validation_data=(X_test_final, Y_test))

classifier.save('static_ges.h5')

from keras.models import load_model

model = load_model('static_ges.h5')
y_pred = model.predict_classes(X_test_final)

from sklearn.metrics import classification_report

model.evaluate(X_test_final, model.predict(X_test_final))





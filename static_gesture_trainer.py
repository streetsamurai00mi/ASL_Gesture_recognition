# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:35:33 2019

@author: Anupam Anand
"""
import pandas as pd
import numpy as np

file_train = pd.read_csv('sign-language-mnist/sign_mnist_train.csv')
X_train =  file_train.iloc[:, 1:]
Y_train = file_train.iloc[:,0]

file_test = pd.read_csv('sign-language-mnist/sign_mnist_test.csv')
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
print("Data set Prepared\n Preparing the CNN...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

classifier = Sequential()

classifier.add(Conv2D(filters = 32,kernel_size=(3,3), padding = "same", activation = "relu", input_shape= (28,28,1)))

classifier.add(Conv2D(filters = 32,kernel_size=(3,3), padding = "same", activation = "relu"))

classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(filters = 64,kernel_size=(5,5), padding = "same", activation = "relu"))

classifier.add(Conv2D(filters = 128,kernel_size=(5,5), padding = "same", activation = "relu"))

classifier.add(MaxPool2D((2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=256, activation = "relu"))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=32,activation= 'relu'))

classifier.add(Dropout(0.3))

classifier.add(Dense(units=25, activation="softmax"))

classifier.summary()


classifier.compile(optimizer= Adam(learning_rate=0.0001), loss= 'categorical_crossentropy', metrics=['accuracy'])

print("CNN prepared.... Training")

hist = classifier.fit(X_train_final, Y_train.values, batch_size= 32, epochs = 30, validation_data=(X_test_final, Y_test.values))

classifier.save('static_ges.h5')

from tensorflow.keras.models import load_model

model = load_model('static_ges.h5')
y_pred = model.predict_classes(X_test_final.astype(float))

from sklearn.metrics import classification_report, accuracy_score

model.evaluate(X_test_final, model.predict(X_test_final.astype(float)))

accuracy_score(file_test.iloc[:,0].values, y_pred)





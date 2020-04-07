from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

# save the original weights
file_train = pd.read_csv('sign-language-mnist/sign_mnist_train.csv')
X_train =  file_train.iloc[:, 1:]
Y_train = file_train.iloc[:,0]
file_train = pd.read_csv('sign-language-mnist/train_cus.csv')
X_train = X_train.append(file_train.iloc[:, 1:])
Y_train = Y_train.append(file_train.iloc[:,0])

file_test = pd.read_csv('sign-language-mnist/sign_mnist_test.csv')
X_test = file_test.iloc[:, 1:]
Y_test = file_test.iloc[:,0]
file_test = pd.read_csv('sign-language-mnist/test_cus.csv')
X_test = X_test.append(file_test.iloc[:, 1:])
Y_test = Y_test.append(file_test.iloc[:,0])

Y_train = pd.get_dummies(Y_train)
Y_train.insert(9,'9', np.zeros(Y_train.shape[0]))
Y_test = pd.get_dummies(Y_test)
Y_test.insert(9,'9', np.zeros(Y_test.shape[0]))

X_train = X_train.values
X_test = X_test.values

X_train_final = X_train.reshape(X_train.shape[0],28,28,1)
X_test_final = X_test.reshape(X_test.shape[0],28,28,1)

model = load_model('static_ges.h5')
weights_bak = model.layers[-1].get_weights()
nb_classes = model.layers[-1].output_shape[-1]
model.summary()
model.pop()

model.add(Dense(units=nb_classes+1, activation='softmax', name='dense_3'))
weights_new = model.layers[-1].get_weights()

weights_new[0][:,:-1] = weights_bak[0]
weights_new[1][:-1] = weights_bak[1]

weights_new[0][:,-1] = np.mean(weights_bak[0], axis=1)
weights_new[1][-1] = np.mean(weights_bak[1])

model.layers[-1].set_weights(weights_new)
model.summary()
model.compile(optimizer= Adam(learning_rate=0.0001), loss= 'categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train_final, Y_train.values, batch_size= 32, epochs = 10, validation_data=(X_test_final, Y_test.values))
model.save('new_ges.h5')
model = load_model('new_ges.h5')

from sklearn.metrics import classification_report, accuracy_score
model.evaluate(X_test_final, model.predict(X_test_final.astype(float)))
accuracy_score(Y_test.values, model.predict_classes(X_test_final.astype(float)))

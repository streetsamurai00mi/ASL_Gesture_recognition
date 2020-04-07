# ASL_Gesture_recognition
A CNN based for Gesture Recognition System.

Dependencies:

1. Numpy 2. Pandas 3. Tensorflow 4. tkinter 5. OpenCV 6. SQLite3

How to use:

Using static_gesture_trainer.py file train the model to recognize static static ASL gestures. 
Proceed after the above process is completed.
The GUI can be launched by running the gesture_gui.py file.
You will see various options with instructions to use the system.

Options available are:
1. Create new Gestures: Using this option we can create new gestures i.e. with a new hand sign.
2. Create alternate Gesture: This option can be used to create gestures with unique movement but the hand sign of the gesture create with the first option.
3. Gesture Recognition: This button with launch the Gesture Recognition window.
4. Train: This button will start training with the new dataset created.

Information on Indivisual files:
1. static_gesture_trainer: This file will train a model on the MNIST Dataset for ASL letters.
2. create_gestures: This file contains functions to capture gestures and feed the movement data into the database.
3. image_gen: This file contains funtions for image augmentation and feed the images into a csv file.
4. custom_model_trainer: This file helps to train the custom dataset created over the already trained model.
5. gesture_with_hand_tracking: It contains code for gesture detection using the trained model and the movement database.

Database Details:
  Database Name: gesturedb
  Table Name: ges_dy
  Table Schema:
              CREATE TABLE ges_dy(
                gescode INTEGER,
                word TEXT,
                movement TEXT);
                
  Database is built n sqlite3.

MNIST Dataset for ASL letters:
https://www.kaggle.com/datamunge/sign-language-mnist

""" 
Author: Aishni Parab
File: model_personid.py
Description: calls functions in data_processing to init data. runs trianing and testing on data.
"""
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from keras.utils import np_utils

import data_processing as data 
import numpy as np
from time import time
import os

np.random.seed(123) # for reproducibility
    
# load data
dataset = data.getData() #instance 
dataset.get()
(X_train, Y_train), (X_test, Y_test) = (dataset.X_train, dataset.Y_train), (dataset.X_test, dataset.Y_test)

# preprocess data
X_train = X_train.reshape(X_train.shape[0], 1, 430, 1)
X_test = X_test.reshape(X_test.shape[0], 1, 430, 1)
print X_train.shape
print X_test.shape

# normalize data values to range [0, 1]
X_train /= 255
X_test /= 255

# convert flat array to [Person1 .. Person90] one-hot coded array
Y_train = Y_train - 1
Y_test = Y_test - 1
Y_train = np_utils.to_categorical(Y_train, 90)
Y_test = np_utils.to_categorical(Y_test, 90)

#model architecture
model = Sequential()
model.add(Convolution2D(32, 1, 5, activation='tanh', input_shape=(1,430,1), kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Convolution2D(64, 1, 5, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(90, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# fit model on training data
tensorboard = TensorBoard(log_dir="logs_personid/{}".format(time()))
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, Y_train, 
          batch_size=10, validation_data=(X_test, Y_test), nb_epoch=100, verbose=1,
          callbacks = [earlystopping, tensorboard])

# evaluate model on test data
print "Evaluating model"
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model to dir
model.save_weights(os.path.join('saved_models', 'rsampled_.h5'))

# plot performance graph
plot_fn = data.plotHelper()
plot_fn.plot_keys(history)

# confusion matrix
y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

inst = data.Setup()
feats, personid, info = inst.get_data()
names, age, gender = inst.dissect_labels(info)
target_names = np.asarray(names)
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# load model from dir
# model = model.load_weights(os.path.join('saved_pid_w_m', 'rsampled_.h5'))

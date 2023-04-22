import os
import random
import numpy as np
import csv

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def train_model(X, Y, model_file, dimension=(224, 224), epochs=10, batch_size=32):
    '''
    Trains a model using the VGG16 architecture. Saves the model to the specified file.

    Parameters:
        X: The training data
        Y: The training labels
        model_file: The file to save the model to
        dimension: The dimension of the images
        epochs: The number of epochs to train for
        batch_size: The batch size to use
    '''
    
    trained_model = VGG16(input_shape = (dimension[0], dimension[1], 3), include_top = False, weights = 'imagenet')

    # do not train all layers
    for layer in trained_model.layers:
        layer.trainable = False

    # Add final layers that will need to be trained
    model = Sequential()
    model.add(trained_model)
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'sigmoid'))

    # compile the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # split train data into train and validation sets
    Y = to_categorical(Y, num_classes = 3)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    # train the model
    vgghist =  model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val), shuffle=True, workers=1, use_multiprocessing=True)

    # evaluate the model by calculating accuracy
    scores = model.evaluate(x_val, y_val, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # print accuracy
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # save the model
    model.save(model_file)


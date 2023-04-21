import preprocessing
from vgg16 import train_model

X, Y = preprocessing.sampleImages(10, 10, 10)

train_model(X, Y, 'vgg-16_fully_trained.h5')
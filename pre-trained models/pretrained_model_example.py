import preprocessing
from vgg16 import train_model

X, Y = preprocessing.sampleImages(1000, 1000, 1000)

train_model(X, Y, 'vgg-16_fully_trained.h5')
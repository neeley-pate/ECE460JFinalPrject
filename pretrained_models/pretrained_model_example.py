import preprocessing
from vgg16 import train_model
from vgg16 import predict

# 64 x 64 images
X, Y = preprocessing.sampleImages(4000, 4000, 4000, verbose=True, dimension=(64, 64))
train_model(X, Y, 'vgg-16_4000_64x64.h5', dimension=(64, 64))
X_test, Y_test = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(64, 64))
predict(X_test, Y_test, 'vgg-16_fully_trained.h5')

# 128 X 128 images
X, Y = preprocessing.sampleImages(2500, 2500, 2500, verbose=True, dimension=(128, 128))
train_model(X, Y, 'vgg-16_2500_128x128.h5', dimension=(128, 128))
X_test, Y_test = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128))
predict(X_test, Y_test, 'vgg-16_fully_trained.h5')
import preprocessing
import vgg16
import inceptionv3
import resnet50
import efficientnet

from keras.utils import to_categorical
from keras.models import load_model

def accuracy(X, Y, model_file):
    '''
    prints accuracy of model on test data

    Parameters:
        X: The test data
        Y: The test labels
        model_file: The file to load the model from
    '''

    model = load_model(model_file)
    scores = model.evaluate(X, to_categorical(Y, num_classes=3), verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# vgg16: 64 x 64 images 
# X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(64, 64), testing=False)
# vgg16.train_model(X, Y, 'vgg-16_4000_64x64.h5', dimension=(64, 64))


# # vgg16:128 X 128 images
# X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=False)
# vgg16.train_model(X, Y, 'vgg-16_2500_128x128.h5', dimension=(128, 128))

# # inception: 128 X 128 images
X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=False)
inceptionv3.train_model(X, Y, 'inceptionv3_1000_128x128.h5', dimension=(128, 128))

# # resnet50
# X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=False)
# resnet50.train_model(X, Y, 'resnet50_100_128x128.h5', dimension=(128, 128))

# # efficientnet
# X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=False)
# efficientnet.train_model(X, Y, 'efficientnet_100_128x128.h5', dimension=(128, 128))

# # mobileNet
# X, Y = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=False)
# efficientnet.train_model(X, Y, 'efficientnet_100_128x128.h5', dimension=(128, 128))

# accuracy
X_test, Y_test = preprocessing.sampleImages(1000, 1000, 1000, verbose=True, dimension=(128, 128), testing=True)
accuracy(X_test, Y_test, 'inceptionv3_1000_128x128.h5')

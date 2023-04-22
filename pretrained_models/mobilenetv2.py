from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def train_model(X, Y, model_file, dimension=(224, 224), epochs=10, batch_size=32):
    mobilenet_model = MobileNetV2(input_shape=(dimension[0], dimension[1], 3), include_top=False, weights='imagenet')

    # do not train all layers
    for layer in mobilenet_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(mobilenet_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    Y = to_categorical(Y, num_classes=3)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val), shuffle=True, workers=1, use_multiprocessing=True)

    scores = model.evaluate(x_val, y_val, verbose=1)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save(model_file)
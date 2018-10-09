from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


def train_and_save():
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)

    X = X/255.0

    dense_layers = [0]
    layer_sizes = [64]
    conv_layers = [3]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, y,
                          batch_size=32,
                          epochs=10,
                          validation_split=0.3,
                          callbacks=[tensorboard])

    model.save('64x3-CNN.model')

# train_and_save()

def prepare(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, 1)

def load_model():
    model = tf.keras.models.load_model("64x3-CNN.model")
    return model

def predict(model, img_path):
    return model.predict([prepare('dog.jpg')])
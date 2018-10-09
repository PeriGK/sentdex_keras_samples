import os
from random import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

datadir = '/home/periklis/Documents/my_code/sentdex_keras/kagglecatsanddogs_3367a/PetImages'
categories = ['Dog', 'Cat']

IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in categories:  # do dogs and cats

        path = os.path.join(datadir,category)  # create path to dogs and cats
        class_num = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

print(len(training_data))

shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()



# IMG_SIZE = 50
# new_array = cv2.resize(training_data, (IMG_SIZE, IMG_SIZE))

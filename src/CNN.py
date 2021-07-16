import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob

import os

from tensorflow import keras
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from keras.optimizers import Adam
from keras.initializers import glorot_uniform

import librosa.display
import librosa as librosa

from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.activations import relu

import warnings
warnings.filterwarnings('ignore')


genres = ['hip-hop', 'classical', 'country', 'electronic', 'metal']

audio_files_path = '/home/aqeelali7/Documents/Galvanized/Capstone-2-Music-Genre-Classifier/data/'
img_save_path = '/home/aqeelali7/Documents/Galvanized/Capstone-2-Music-Genre-Classifier/data/images/'

# count = 1

X = []
target = []
hip_hop_dummies = [1, 0, 0, 0, 0]
classical_dummies = [0, 1, 0, 0, 0]
country_dummies = [0, 0, 1, 0, 0]
electronic_dummies = [0, 0, 0, 1, 0]
metal_dummies = [0, 0, 0, 0, 1]

for i in range(len(genres)):
    
    new_path = os.path.join(audio_files_path,genres[i])
    os.chdir(new_path)
    
    for track_num in range(len(os.listdir())):
    
        
        window_size = 1024
        window = np.hanning(window_size)    
        y, sr = librosa.load((os.listdir()[track_num]), duration = 30.0)
        stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        
        X.append(np.resize(librosa.amplitude_to_db(out, ref=np.max),(513,1292)))
           
        
        if genres[i] == "hip-hop":
            target.append(hip_hop_dummies)
        if genres[i] == "classical":
            target.append(classical_dummies)
        if genres[i] == "country":
            target.append(country_dummies)
        if genres[i] == "electronic":
            target.append(electronic_dummies)
        if genres[i] == "metal":
            target.append(metal_dummies)
            
    print('done with ', genres[i],"!")
        

X = np.array(X)
target = np.array(target)



X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.reshape(-1,513, 1292,1)
X_test = X_test.reshape(-1, 513, 1292,1)

print(X_train.shape)

print(X_test.shape)

# X_training_set, X_val, y_training_set, y_val = train_test_split(X_train, y_train, test_size = 0.25)

batch_size = 128
num_classes = 5
epochs = 5

training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))

testing_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100

training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
testing_data = testing_data.batch(BATCH_SIZE)

del X_train, X_test, y_train, y_test,target,X

print("I'm alive up till here")

model = Sequential()


model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(513, 1292,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print("I'm still alive up till here")

model.fit(training_data,
          batch_size=8,
          epochs=epochs,
          verbose=1,
          validation_data=testing_data)

score = model.evaluate(testing_data, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

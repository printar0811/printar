from __future__ import print_function    
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Enable GPU usage in TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
else:
    print("No GPU detected, using CPU.")

# Load and preprocess data
df = pd.read_csv('op7tloud.csv', header=None, names=['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','sharpness','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','maxfreq','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','class'])

df['labels'] = df['class'].astype('category').cat.codes

df = df.drop(['sharpness', 'q75'], axis='columns')

X = df[['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','maxfreq']]
Y = df['labels']
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle=True)

num_classes = 2
input_shape = (40, 1)

y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], 40, 1)
x_test = x_test.reshape(x_test.shape[0], 40, 1)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Define the model
model = keras.Sequential([
    keras.layers.Conv1D(256, 8, padding='same', input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv1D(128, 8, padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling1D(pool_size=2, strides=1),
    keras.layers.Conv1D(64, 8, padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling1D(pool_size=4, strides=1),
    keras.layers.Conv1D(64, 8, padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(2),
    keras.layers.Activation('softmax'),
    keras.layers.Dropout(0.25)
])

# Compile the model
opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 170
history = model.fit(x_train, y_train_binary,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=False,
                    verbose=1,
                    validation_split=0.1)

# Evaluate the model
predictions = model.predict(x_test)
preds = model.evaluate(x_test, y_test_binary)

print(f"Loss = {preds[0]}")
print(f"Test Accuracy = {preds[1]}")

conf_matrix = confusion_matrix(np.argmax(y_test_binary, axis=1), np.argmax(predictions, axis=1))
print(conf_matrix)

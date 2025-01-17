from __future__ import print_function    
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from glob import glob

# Enable GPU usage in TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
else:
    print("No GPU detected, using CPU.")

# Path to CSV files
folder_path = 'input_files/csv_device_1_6000/'

# Load and preprocess data
all_files = glob(os.path.join(folder_path, "*.csv"))

# Initialize empty dataframe
df_list = []

# Read each file and append to the list
for filename in all_files:
    df = pd.read_csv(filename, header=None)
    df_list.append(df)

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)

print(df.head())

# Check for NaN values
if df.isnull().values.any():
    print("There are NaN values in the DataFrame.")
    df = df.fillna(0)

if df.isnull().values.any():
    print("There are NaN values in the DataFrame.")
else:
    print("All NaN values are taken care of")

# Convert DataFrame to numeric, coercing when told to
df = df.apply(pd.to_numeric, errors='coerce')

# Now check for Inf values
if np.isinf(df.values).any():
    print("There are Inf values in the DataFrame.")

# Replace all inf values with 10000 in df
df = df.replace([np.inf, -np.inf], 10000)

# Now check for Inf values again
if np.isinf(df.values).any():
    print("There are again Inf values in the DataFrame.")
else:
    print("Inf values are taken care of!")

# Check for columns where all values are the same
same_value_columns = df.columns[df.nunique() <= 1].tolist()
print(f"Columns with all same values: {same_value_columns}")

# Calculate the percentage of NaN values in each column
for i in range(df.shape[1]):
    nan_count = df.iloc[:, i].isnull().sum()
    nan_percentage = (nan_count / len(df)) * 100
    # If more than 50% of the values are NaN, print the column index
    if nan_percentage > 50:
        print(f"Column {i+1} has more than 50% NaN values.")

# Last column is the class
X = df.iloc[:, :-1]
Y = df.iloc[:, -1].astype('category').cat.codes

# Determine the number of classes
num_classes = len(np.unique(Y))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle=True)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

# Reshape input data to fit the model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Ensure all data is numeric
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Normalize the data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Define the model
model = keras.Sequential([
    keras.layers.Conv1D(256, 8, padding='same', input_shape=(x_train.shape[1], 1)),
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
    keras.layers.Dense(num_classes),
    keras.layers.Activation('softmax'),
    keras.layers.Dropout(0.25)
])

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 12
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

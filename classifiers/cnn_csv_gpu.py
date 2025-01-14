from __future__ import print_function    
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from glob import glob
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

class TerminateOnNaN(Callback):
    def __init__(self, save_path, lr_reduction_factor=0.1):
        super(TerminateOnNaN, self).__init__()
        self.save_path = save_path
        self.lr_reduction_factor = lr_reduction_factor

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"Batch {batch}: Invalid loss, terminating training")
            self.model.stop_training = True
            self.model.load_weights(self.save_path)  # Load the last saved weights
            old_lr = keras.backend.get_value(self.model.optimizer.lr)
            new_lr = old_lr * self.lr_reduction_factor
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"Reducing learning rate from {old_lr} to {new_lr}")

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Enable GPU usage in TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
else:
    print("No GPU detected, using CPU.")

# Path to CSV files
folder_path = 'input_files/output_device_10000_device_user_corrected/'

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

# Check for NaN values
df = df.fillna(0)

# Convert DataFrame to numeric, coercing when told to
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Replace all inf values with 10000 in df
df = df.replace([np.inf, -np.inf], 10000)

# Check for columns where all values are the same
same_value_columns = df.columns[df.nunique() <= 1].tolist()
print(f"Columns with all same values: {same_value_columns}")

# Calculate the percentage of NaN values in each column
for i in range(df.shape[1]):
    nan_count = df.iloc[:, i].isnull().sum()
    nan_percentage = (nan_count / len(df)) * 100
    # If more than 50% of the values are NaN, print the column index
    if nan_percentage > 0.05:
        print(f"Column {i+1} has more than 50% NaN values.")

# Last column is the class
X = df.iloc[:, :-1]
Y = df.iloc[:, -1].astype('category').cat.codes

# Determine the number of classes
num_classes = len(np.unique(Y))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle=True)

# Ensure all values in x_train and x_test are numeric
x_train = pd.DataFrame(x_train).apply(pd.to_numeric, errors='coerce').fillna(0).values
x_test = pd.DataFrame(x_test).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Convert class vectors to binary class matrices (one-hot encoding)
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

# Reshape input data to fit the model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

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
    keras.layers.Dense(num_classes, kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Activation('softmax'),
    keras.layers.Dropout(0.25)
])

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.000001)  # Lower learning rate
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define callbacks
checkpoint_filepath = 'model_checkpoint.h5'
callbacks = [
    ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=False, verbose=1),
    TerminateOnNaN(save_path=checkpoint_filepath),
    LearningRateScheduler(scheduler)
]

# Train the model
batch_size = 64
epochs = 40
history = model.fit(x_train, y_train_binary,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=False,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=callbacks)

# Evaluate the model
predictions = model.predict(x_test)
preds = model.evaluate(x_test, y_test_binary)

print(f"Loss = {preds[0]}")
print(f"Test Accuracy = {preds[1]}")

conf_matrix = confusion_matrix(np.argmax(y_test_binary, axis=1), np.argmax(predictions, axis=1))
#print(conf_matrix)

# Generate classification report
report = classification_report(np.argmax(y_test_binary, axis=1), np.argmax(predictions, axis=1), output_dict=True)
#print(report)

# Extract macro and weighted averages for Precision, Recall, F1-score
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1 = report['weighted avg']['f1-score']

# Manually calculate TNR
tnr_per_class = []
for i in range(num_classes):
    tn = np.sum(conf_matrix) - (np.sum(conf_matrix[:, i]) + np.sum(conf_matrix[i, :]) - conf_matrix[i, i])
    fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
    tnr = tn / (tn + fp)
    tnr_per_class.append(tnr)

macro_tnr = np.mean(tnr_per_class)
weighted_tnr = np.sum([tnr_per_class[i] * np.sum(conf_matrix[i, :]) for i in range(num_classes)]) / np.sum(conf_matrix)

# Print macro and weighted averages
print("\nMacro Average Precision: {:.4f}".format(macro_precision))
print("Macro Average Recall (TPR): {:.4f}".format(macro_recall))
print("Macro Average F1-Score: {:.4f}".format(macro_f1))
print("Macro Average TNR: {:.4f}".format(macro_tnr))

print("\nWeighted Average Precision: {:.4f}".format(weighted_precision))
print("Weighted Average Recall (TPR): {:.4f}".format(weighted_recall))
print("Weighted Average F1-Score: {:.4f}".format(weighted_f1))
print("Weighted Average TNR: {:.4f}".format(weighted_tnr))

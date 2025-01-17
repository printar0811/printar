from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
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
folder_path = 'input_files/device_70_files_users/'

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
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Now check for Inf values
if np.isinf(df_numeric.values).any():
    print("There are Inf values in the DataFrame.")

# Replace all inf values with 10000 in df
df = df.replace([np.inf, -np.inf], 10000)

# Convert DataFrame to numeric, coercing when told to
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Now check for Inf values
if np.isinf(df_numeric.values).any():
    print("There are again Inf values in the DataFrame.")
else:
    print("Inf values are taken care of!")

# Check for columns where all values are the same
same_value_columns = df.columns[df.nunique() <= 1].tolist()
if same_value_columns:
    print(f"Columns with all same values: {same_value_columns}")
    # Drop columns with all same values
    df.drop(columns=same_value_columns, inplace=True)

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

# Save the original labels for later use
original_labels = df.iloc[:, -1].astype('category').cat.categories

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
opt = keras.optimizers.Adam(lr=0.00001)
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

# Confusion matrix and classification report
y_pred = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Find the classes present in the test data
unique_classes = np.unique(y_test)

# Create a mapping from class indices to class names using the unique classes in y_test
present_labels = original_labels[unique_classes]

# Calculate metrics
report = classification_report(y_test, y_pred, target_names=present_labels, output_dict=True)

# Extract and print macro and weighted metrics
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']  # Equivalent to macro TPR
macro_f1 = report['macro avg']['f1-score']

weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']  # Equivalent to weighted TPR
weighted_f1 = report['weighted avg']['f1-score']

print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall (TPR): {macro_recall}")
print(f"Macro F1 Score: {macro_f1}")
print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall (TPR): {weighted_recall}")
print(f"Weighted F1 Score: {weighted_f1}")

# Calculate True Negative Rate (TNR) for each class and average
tnr_list = []
for i, label in enumerate(present_labels):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    tnr_list.append(tnr)
    #print(f"Class '{label}' TNR: {tnr}")

macro_tnr = np.mean(tnr_list)
weighted_tnr = np.average(tnr_list, weights=conf_matrix.sum(axis=1) / conf_matrix.sum())

print(f"Macro TNR: {macro_tnr}")
print(f"Weighted TNR: {weighted_tnr}")

# Count total correctly and incorrectly classified
total_correctly_classified = np.trace(conf_matrix)
total_incorrectly_classified = conf_matrix.sum() - total_correctly_classified

print(f"Total Correctly Classified: {total_correctly_classified}")
print(f"Total Incorrectly Classified: {total_incorrectly_classified}")

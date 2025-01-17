import os
import time
import numpy as np
import cudf
import pandas as pd
from cuml.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

# Define the path to the folder containing CSV files
folder_path = 'input_files/csv_device_2/'

# List all CSV files in the specified folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Check if there are CSV files in the folder
if len(csv_files) == 0:
    print('No CSV files found in the specified folder.')
else:
    # Initialize lists to hold the data
    all_X = []
    all_y = []

    # Load and concatenate all CSV files
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)

        # Ensure the last column is the target variable
        X = data.iloc[:, :-1].values  # All columns except the last one
        y = data.iloc[:, -1].values  # Last column

        all_X.append(X)
        all_y.append(y)

    # Concatenate all data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Print the total number of classes
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"Total number of classes: {num_classes}")

    # Check for NaN and infinity values and replace them
    X = np.nan_to_num(X, nan=0.0, posinf=1e308, neginf=-1e308)
    y = np.nan_to_num(y, nan=0.0, posinf=1e308, neginf=-1e308)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure the training set has more than one unique class label
    if len(np.unique(y_train)) > 1:
        # Create a label encoder
        label_encoder_pre = LabelEncoder()
        y_train_encoded = label_encoder_pre.fit_transform(y_train)
        y_test_encoded = label_encoder_pre.fit_transform(y_test)

        # Convert to cuDF DataFrame
        X_train = cudf.DataFrame.from_records(X_train)
        X_test = cudf.DataFrame.from_records(X_test)
        y_train = cudf.Series(y_train_encoded)
        y_test = cudf.Series(y_test_encoded)

        # Create a LightGBM classifier
        classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=10)

        # Start timer
        start_time = time.time()

        # Train the classifier
        classifier.fit(X_train.to_pandas(), y_train.to_pandas())

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Make predictions on the test set
        predictions = classifier.predict(X_test.to_pandas())

        # Create a label encoder
        label_encoder = LabelEncoder()

        # Fit the label encoder on the target variable
        label_encoder.fit(y_test.to_pandas())

        # Encode the predicted and actual class labels
        predictions = label_encoder.transform(predictions)
        y_test = label_encoder.transform(y_test.to_pandas())

        # Generate confusion matrix
        matrix = confusion_matrix(y_test, predictions)

        # Calculate TPR and TNR
        TPR = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])  # True Positive Rate
        TNR = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])  # True Negative Rate

        # Print the results
        print("\nConfusion Matrix:")
        print(matrix)
        print("TP Rate: {:.4f}".format(TPR))
        print("TN Rate: {:.4f}".format(TNR))

    else:
        raise ValueError("The target variable y_train contains only one unique class label. Ensure your dataset has more than one class.")

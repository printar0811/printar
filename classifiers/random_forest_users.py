import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define the path to the folder containing CSV files
folder_path = 'input_files/output_device_10000_device3_user/'

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
        data = pd.read_csv(file_path, header=None)

        # Ensure the last column is the target variable
        X = data.iloc[:, :-1].values.astype(np.float32)  # All columns except the last one
        y = data.iloc[:, -1].values  # Last column (class names)

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

    # Clip values to be within the range of float32
    X = np.clip(X, np.finfo(np.float32).min, np.finfo(np.float32).max)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure the training set has more than one unique class label
    if len(np.unique(y_train)) > 1:
        # Create a label encoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Create a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt')

        # Start timer
        start_time = time.time()

        # Train the classifier
        classifier.fit(X_train, y_train_encoded)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Decode the predicted and actual class labels
        predictions = label_encoder.inverse_transform(predictions)

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Generate classification report with zero_division parameter
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions)

        # Check if confusion matrix shape matches the number of classes
        if matrix.shape[0] != num_classes:
            print("Warning: The number of classes does not match the confusion matrix size.")
            num_classes = min(matrix.shape[0], num_classes)  # Adjust the number of classes to fit the matrix

        # Compute True Negative Rate (TNR) for each class
        tnrs = []
        for i in range(num_classes):
            tn = matrix.sum() - (matrix[i, :].sum() + matrix[:, i].sum() - matrix[i, i])
            fp = matrix[:, i].sum() - matrix[i, i]
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            tnrs.append(tnr)

        # Calculate macro and weighted averages
        macro_tnr = np.mean(tnrs)
        weighted_tnr = np.average(tnrs, weights=np.sum(matrix, axis=1) / np.sum(matrix))

        print("Time taken to build model: {:.2f} seconds\n".format(elapsed_time))
        print("=== Stratified cross-validation ===")
        print("=== Summary ===\n")
        print("Correctly Classified Instances    {:>10}     {:.4f} %".format(np.sum(predictions == y_test), accuracy * 100))
        print("Incorrectly Classified Instances  {:>10}     {:.4f} %".format(np.sum(predictions != y_test), (1 - accuracy) * 100))
        print("Kappa statistic                    {:.4f}".format(accuracy))

        # Printing macro and weighted averages for the metrics
        print("\nMacro and Weighted Averages:")
        print(f"Macro TPR (Recall): {report['macro avg']['recall']:.4f}")
        print(f"Weighted TPR (Recall): {report['weighted avg']['recall']:.4f}")
        print(f"Macro TNR: {macro_tnr:.4f}")
        print(f"Weighted TNR: {weighted_tnr:.4f}")
        print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
        print(f"Weighted Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Macro F1 Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1 Score: {report['weighted avg']['f1-score']:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    else:
        raise ValueError("The target variable y_train contains only one unique class label. Ensure your dataset has more than one class.")

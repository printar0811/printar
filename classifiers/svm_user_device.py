import os
import time
import numpy as np
import cudf
from cuml.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

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
    print(f"Unique classes: {unique_classes}")

    # Check for NaN and infinity values and replace them
    X = np.nan_to_num(X, nan=0.0, posinf=1e308, neginf=-1e308)
    y = np.nan_to_num(y, nan=0.0, posinf=1e308, neginf=-1e308)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a label encoder and fit it on the entire dataset
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert to cuDF DataFrame
    X_train = cudf.DataFrame.from_records(X_train)
    X_test = cudf.DataFrame.from_records(X_test)
    y_train = cudf.Series(y_train_encoded)
    y_test = cudf.Series(y_test_encoded)

    # Create a Linear SVM classifier
    classifier = LinearSVC()

    # Start timer
    start_time = time.time()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    predictions = classifier.predict(X_test).to_numpy()

    # Ensure predictions are integers
    predictions = predictions.astype(int)

    # Encode the predicted and actual class labels
    y_test = y_test.to_numpy()

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    mean_abs_error = np.mean(np.abs(predictions - y_test))
    root_mean_sq_error = np.sqrt(mean_squared_error(y_test, predictions))
    relative_abs_error = (mean_abs_error / np.mean(y_test)) * 100
    root_relative_sq_error = (root_mean_sq_error / np.sqrt(np.mean(y_test))) * 100

    # Generate classification report and confusion matrix
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    # Use the shape of the confusion matrix to determine the number of classes
    num_classes_in_matrix = matrix.shape[0]

    # Calculate TNR (Specificity) for each class and its macro and weighted average
    tnrs = []
    tprs = []
    precisions = []
    f1_scores = []

    for i in range(num_classes_in_matrix):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = matrix.sum() - (tp + fn + fp)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        tprs.append(tpr)
        tnrs.append(tnr)
        precisions.append(precision)
        f1_scores.append(f1_score)

    macro_tpr = np.mean(tprs)
    macro_tnr = np.mean(tnrs)
    macro_precision = np.mean(precisions)
    macro_f1 = np.mean(f1_scores)

    weighted_tpr = np.average(tprs, weights=np.sum(matrix, axis=1) / np.sum(matrix))
    weighted_tnr = np.average(tnrs, weights=np.sum(matrix, axis=1) / np.sum(matrix))
    weighted_precision = np.average(precisions, weights=np.sum(matrix, axis=1) / np.sum(matrix))
    weighted_f1 = np.average(f1_scores, weights=np.sum(matrix, axis=1) / np.sum(matrix))

    # Print the classification report
    print("Classification Report:")
    for label, metrics in report.items():
        print("\nClass:", label)
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        else:
            print(f"{label}: {metrics}")

    # Print the confusion matrix
    print("\nConfusion Matrix:")
    print(matrix)

    # Print the results
    print("Time taken to build model: {:.2f} seconds\n".format(elapsed_time))
    print("=== Stratified cross-validation ===")
    print("=== Summary ===\n")
    print("Correctly Classified Instances    {:>10}     {:.4f} %".format(np.sum(predictions == y_test), accuracy * 100))
    print("Incorrectly Classified Instances  {:>10}     {:.4f} %".format(np.sum(predictions != y_test), (1 - accuracy) * 100))
    print("Kappa statistic                    {:.4f}".format(accuracy))
    print("Mean absolute error                {:.4f}".format(mean_abs_error))
    print("Root mean squared error            {:.4f}".format(root_mean_sq_error))
    print("Relative absolute error            {:.4f} %".format(relative_abs_error))
    print("Root relative squared error        {:.4f} %".format(root_relative_sq_error))
    print("Total Number of Instances          {:>10}\n".format(y_test.shape[0]))

    print("\nMacro and Weighted Averages:")
    print(f"Macro TPR (Recall): {macro_tpr:.4f}")
    print(f"Weighted TPR (Recall): {weighted_tpr:.4f}")
    print(f"Macro TNR (Specificity): {macro_tnr:.4f}")
    print(f"Weighted TNR (Specificity): {weighted_tnr:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

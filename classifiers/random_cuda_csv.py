import os
import time
import numpy as np
import cudf
import pandas as pd
import logging
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Set up logging
logging.basicConfig(filename='result_log.txt', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Define the path to the folder containing CSV files
folder_path = 'input_files/output_device_10000_device_user_corrected/'

# List all CSV files in the specified folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Check if there are CSV files in the folder
if len(csv_files) == 0:
    logger.info('No CSV files found in the specified folder.')
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

    # Log the total number of classes
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    logger.info(f"Total number of classes: {num_classes}")
    # logger.info(f"Unique classes: {unique_classes}")

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

        # Create a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, max_depth=10)

        # Start timer
        start_time = time.time()

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Make predictions on the test set
        predictions = classifier.predict(X_test).to_numpy()

        # Create a label encoder
        label_encoder = LabelEncoder()

        # Fit the label encoder on the target variable
        label_encoder.fit(y_test.to_numpy())

        # Encode the predicted and actual class labels
        predictions = label_encoder.transform(predictions)
        y_test = label_encoder.transform(y_test.to_numpy())

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, predictions)
        mean_abs_error = np.mean(np.abs(predictions - y_test))
        root_mean_sq_error = np.sqrt(mean_squared_error(y_test, predictions))
        relative_abs_error = (mean_abs_error / np.mean(y_test)) * 100
        root_relative_sq_error = (root_mean_sq_error / np.sqrt(np.mean(y_test))) * 100

        # Generate classification report and confusion matrix
        report_str = classification_report(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)

        # Log confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info(matrix)

        # Log the classification report
        logger.info("\nClassification Report:")
        logger.info(report_str)

        # Calculate metrics for detailed accuracy by class
        report = classification_report(y_test, predictions, output_dict=True)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']
        
        logger.info("\nMacro Average Precision: {:.4f}".format(macro_precision))
        logger.info("Macro Average Recall: {:.4f}".format(macro_recall))
        logger.info("Macro Average F1-Score: {:.4f}".format(macro_f1))
        
        logger.info("\nWeighted Average Precision: {:.4f}".format(weighted_precision))
        logger.info("Weighted Average Recall: {:.4f}".format(weighted_recall))
        logger.info("Weighted Average F1-Score: {:.4f}".format(weighted_f1))
        
        logger.info("\nAccuracy: {:.4f}".format(accuracy))
        
        logger.info("Mean Absolute Error: {:.4f}".format(mean_abs_error))
        logger.info("Root Mean Squared Error: {:.4f}".format(root_mean_sq_error))
        logger.info("Relative Absolute Error: {:.4f} %".format(relative_abs_error))
        logger.info("Root Relative Squared Error: {:.4f} %".format(root_relative_sq_error))
        logger.info("Total Number of Instances: {:>10}\n".format(y_test.shape[0]))

        # Calculate and log macro and weighted TNR
        tnr_list = []
        weighted_tnr_list = []
        class_support = []

        for i in range(len(matrix)):
            tn = matrix.sum() - (matrix[i, :].sum() + matrix[:, i].sum() - matrix[i, i])
            fp = matrix[:, i].sum() - matrix[i, i]
            fn = matrix[i, :].sum() - matrix[i, i]
            tp = matrix[i, i]

            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
            tnr_list.append(tnr)
            class_support.append(matrix[i, :].sum())

        macro_tnr = np.mean(tnr_list)
        weighted_tnr = np.average(tnr_list, weights=class_support)

        logger.info("\nMacro Average TNR: {:.4f}".format(macro_tnr))
        logger.info("Weighted Average TNR: {:.4f}".format(weighted_tnr))

    else:
        raise ValueError("The target variable y_train contains only one unique class label. Ensure your dataset has more than one class.")

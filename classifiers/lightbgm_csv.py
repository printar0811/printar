import os
import time
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

import lightgbm as lgb

# Set up logging
logging.basicConfig(filename='result_log.txt', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Define the path to the folder containing CSV files
folder_path = 'input_files/device_70_files/'

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

    # Check for NaN and infinity values and replace them
    X = np.nan_to_num(X, nan=0.0, posinf=1e308, neginf=-1e308)
    y = np.nan_to_num(y, nan=0.0, posinf=1e308, neginf=-1e308)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Create a label encoder and fit on all labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Create a LightGBM classifier
    classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=10)

    # Start timer
    start_time = time.time()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    predictions = classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    mean_abs_error = np.mean(np.abs(predictions - y_test))
    root_mean_sq_error = np.sqrt(mean_squared_error(y_test, predictions))
    relative_abs_error = (mean_abs_error / np.mean(y_test)) * 100
    root_relative_sq_error = (root_mean_sq_error / np.sqrt(np.mean(y_test))) * 100

    # Generate classification report and confusion matrix
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    # Log confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(matrix)

    # Log the classification report
    logger.info("\nClassification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(f"\nClass {label}:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
        else:
            logger.info(f"{label}: {metrics}")

    # Calculate TNR for each class
    class_counts = np.sum(matrix, axis=1)
    tn_counts = np.sum(matrix, axis=0) - np.diag(matrix)
    tnr_per_class = tn_counts / (class_counts - np.diag(matrix))

    # Calculate macro and weighted TNR
    macro_tnr = np.mean(tnr_per_class)
    weighted_tnr = np.sum(tnr_per_class * class_counts) / np.sum(class_counts)

    # Calculate macro and weighted metrics
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    macro_tpr = macro_recall

    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    weighted_tpr = weighted_recall

    # Log macro and weighted metrics
    logger.info("\nMacro Average Precision: {:.4f}".format(macro_precision))
    logger.info("Macro Average Recall (TPR): {:.4f}".format(macro_tpr))
    logger.info("Macro Average F1-Score: {:.4f}".format(macro_f1))
    logger.info("Macro Average TNR: {:.4f}".format(macro_tnr))

    logger.info("\nWeighted Average Precision: {:.4f}".format(weighted_precision))
    logger.info("Weighted Average Recall (TPR): {:.4f}".format(weighted_tpr))
    logger.info("Weighted Average F1-Score: {:.4f}".format(weighted_f1))
    logger.info("Weighted Average TNR: {:.4f}".format(weighted_tnr))

    # Log other metrics
    logger.info("\nAccuracy: {:.4f}".format(accuracy))
    logger.info("Mean Absolute Error: {:.4f}".format(mean_abs_error))
    logger.info("Root Mean Squared Error: {:.4f}".format(root_mean_sq_error))
    logger.info("Relative Absolute Error: {:.4f} %".format(relative_abs_error))
    logger.info("Root Relative Squared Error: {:.4f} %".format(root_relative_sq_error))
    logger.info("Total Number of Instances: {:>10}\n".format(y_test.shape[0]))

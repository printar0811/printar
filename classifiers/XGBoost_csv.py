import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='result_log.txt', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Define the path to the folder containing CSV files
folder_path = 'input_files/device_70_files_users/'

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
    logger.info(f"Unique classes: {unique_classes}")

    # Check for NaN and infinity values and replace them
    X = np.nan_to_num(X, nan=0.0, posinf=1e308, neginf=-1e308)
    y = np.nan_to_num(y, nan=0.0, posinf=1e308, neginf=-1e308)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a label encoder and fit it on both y_train and y_test combined
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])  # Combine train and test labels
    label_encoder.fit(all_labels)  # Fit the label encoder on all labels

    # Encode the training and testing labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dtest = xgb.DMatrix(X_test, label=y_test_encoded)
    # Define XGBoost parameters for multi-class classification
    params = {
        'objective': 'multi:softmax',  # For multi-class classification
        'num_class': num_classes,       # Number of classes
        'max_depth': 10,
        'eta': 0.1,
        'eval_metric': 'mlogloss',      # Multi-class logloss
        'tree_method': 'hist',          # Use histogram-based algorithm
        'device': 'cuda'                # Specify to use GPU
    }

    # Start timer
    start_time = time.time()

    # Train the classifier
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    predictions = bst.predict(dtest).astype(int)  # Convert probabilities to class labels

    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, predictions)
    mean_abs_error = np.mean(np.abs(predictions - y_test_encoded))
    root_mean_sq_error = np.sqrt(mean_squared_error(y_test_encoded, predictions))
    relative_abs_error = (mean_abs_error / np.mean(y_test_encoded)) * 100
    root_relative_sq_error = (root_mean_sq_error / np.sqrt(np.mean(y_test_encoded))) * 100

    # Generate classification report and confusion matrix
    report = classification_report(y_test_encoded, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test_encoded, predictions)

    # Log confusion matrix
    logger.info("\nXGBoost Confusion Matrix:")
    logger.info(matrix)

    # Log the classification report
    logger.info("\nXGBoost Classification Report:")
    logger.info(classification_report(y_test_encoded, predictions))

    # Calculate metrics for detailed accuracy by class
    tnr_list = []
    tpr_list = []
    class_support = []

    for i in range(len(matrix)):
        tn = matrix.sum() - (matrix[i, :].sum() + matrix[:, i].sum() - matrix[i, i])
        fp = matrix[:, i].sum() - matrix[i, i]
        fn = matrix[i, :].sum() - matrix[i, i]
        tp = matrix[i, i]

        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr_list.append(tnr)
        tpr_list.append(tpr)
        class_support.append(matrix[i, :].sum())

    macro_tnr = np.mean(tnr_list)
    weighted_tnr = np.average(tnr_list, weights=class_support)
    macro_tpr = np.mean(tpr_list)
    weighted_tpr = np.average(tpr_list, weights=class_support)

    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']

    logger.info("\nXGBoost Macro Average TNR: {:.4f}".format(macro_tnr))
    logger.info("XGBoost Weighted Average TNR: {:.4f}".format(weighted_tnr))
    logger.info("XGBoost Macro Average TPR: {:.4f}".format(macro_tpr))
    logger.info("XGBoost Weighted Average TPR: {:.4f}".format(weighted_tpr))
    logger.info("XGBoost Macro Average Precision: {:.4f}".format(macro_precision))
    logger.info("XGBoost Weighted Average Precision: {:.4f}".format(weighted_precision))
    logger.info("XGBoost Macro Average Recall: {:.4f}".format(macro_recall))
    logger.info("XGBoost Weighted Average Recall: {:.4f}".format(weighted_recall))
    logger.info("XGBoost Macro Average F1-Score: {:.4f}".format(macro_f1))
    logger.info("XGBoost Weighted Average F1-Score: {:.4f}".format(weighted_f1))

    logger.info("\nXGBoost Accuracy: {:.4f}".format(accuracy))
    logger.info("XGBoost Mean Absolute Error: {:.4f}".format(mean_abs_error))
    logger.info("XGBoost Root Mean Squared Error: {:.4f}".format(root_mean_sq_error))
    logger.info("XGBoost Relative Absolute Error: {:.4f} %".format(relative_abs_error))
    logger.info("XGBoost Root Relative Squared Error: {:.4f} %".format(root_relative_sq_error))
    logger.info("XGBoost Total Number of Instances: {:>10}\n".format(y_test_encoded.shape[0]))

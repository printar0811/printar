import os
import time
import numpy as np
import cudf
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define the path to the ARFF file
file_path = 'input_files/arff/robocall_op9 -loud.arff'

# Check if the file exists
if os.path.exists(file_path):
    # Load the ARFF file
    data, meta = arff.loadarff(file_path)

    # Extract attribute names
    attributes = meta.names()[:-1]  # Exclude the last attribute (target)

    # Convert data to numpy array
    data = np.array(data.tolist())

    # Get the features and target variables
    X = data[:, :-1]  # All columns except the last one
    y = data[:, -1]  # Last column

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)

    # Calculate metrics for detailed accuracy by class
    TP_rate = report['1']['recall']
    FP_rate = 1 - report['0']['recall']
    precision = report['1']['precision']
    f_measure = report['1']['f1-score']
    MCC = (report['1']['recall'] * precision - FP_rate * (1 - TP_rate)) / \
          np.sqrt((precision + FP_rate) * (1 - precision) * (TP_rate + FP_rate) * (1 - TP_rate))
    ROC_area = report['1']['roc_auc'] if 'roc_auc' in report['1'] else 0.0
    PRC_area = report['1']['average_precision'] if 'average_precision' in report['1'] else 0.0

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

    print("=== Detailed Accuracy By Class ===\n")
    print("                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class")
    print("                 {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}     robo".format(
        TP_rate, FP_rate, precision, TP_rate, f_measure, MCC, ROC_area, PRC_area))
    print("                 {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}     human".format(
        1 - TP_rate, 1 - FP_rate, 1 - precision, 1 - TP_rate, 1 - f_measure, 1 - MCC, 1 - ROC_area, 1 - PRC_area))
    print("Weighted Avg.    {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}\n".format(
        TP_rate, FP_rate, precision, TP_rate, f_measure, MCC, ROC_area, PRC_area))

    print("=== Confusion Matrix ===\n")
    print("   a   b   <-- classified as")
    print("{:>4} {:>4} |   a = robo".format(matrix[0][0], matrix[0][1]))
    print("{:>4} {:>4} |   b = human".format(matrix[1][0], matrix[1][1]))

else:
    print('ARFF file not found at the specified location.')

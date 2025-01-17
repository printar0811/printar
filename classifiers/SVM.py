import os
import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Define the path to the ARFF file
file_path = '/Users/tanvirmahdad/robocall/data/robocall_op9 -loud.arff'

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

    # Create an SVM classifier
    classifier = svm.SVC(kernel='linear')

    # Start timer
    start_time = time.time()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    predictions = classifier.predict(X_test)

    # Create a label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the target variable
    label_encoder.fit(y_test)

    # Encode the predicted and actual class labels
    predictions = label_encoder.transform(predictions)
    y_test = label_encoder.transform(y_test)

    # Now you can perform the subtraction operation
    error = predictions - y_test

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Generate classification report and confusion matrix
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    # Calculate the TP, FP, TN, FN values from the confusion matrix
    TP = matrix[1, 1]
    FP = matrix[0, 1]
    TN = matrix[0, 0]
    FN = matrix[1, 0]

    # Calculate the TP Rate (Recall)
    TP_rate = TP / (TP + FN)

    # Calculate the FP Rate
    FP_rate = FP / (FP + TN)

    # Calculate the Precision
    precision = TP / (TP + FP)

    # Calculate the F-Measure
    f_measure = 2 * (precision * TP_rate) / (precision + TP_rate)

    # Calculate the MCC (Matthews correlation coefficient)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # Calculate the ROC Area
    ROC_area = 0.5 * (TP_rate + (1 - FP_rate))

    # Calculate the PRC Area (Precision-Recall Curve Area)
    PRC_area = precision * TP_rate



    # Print the results
    print("Time taken to build model: {:.2f} seconds\n".format(elapsed_time))
    print("=== Stratified cross-validation ===")
    print("=== Summary ===\n")
    print("Correctly Classified Instances    {:>10}     {:.4f} %".format(np.sum(predictions == y_test), accuracy * 100))
    print("Incorrectly Classified Instances  {:>10}     {:.4f} %".format(np.sum(predictions != y_test),
                                                                         (1 - accuracy) * 100))
    print("Kappa statistic                    {:.4f}".format(accuracy_score(y_test, predictions)))
    print("Mean absolute error                {:.4f}".format(np.mean(np.abs(predictions - y_test))))
    print("Root mean squared error            {:.4f}".format(np.sqrt(np.mean((predictions - y_test) ** 2))))
    print("Relative absolute error            {:.4f} %".format(
        (np.mean(np.abs(predictions - y_test)) / np.mean(y_test)) * 100))
    print("Root relative squared error        {:.4f} %".format(
        (np.sqrt(np.mean((predictions - y_test) ** 2)) / np.sqrt(np.mean(y_test))) * 100))
    print("Total Number of Instances          {:>10}\n".format(y_test.shape[0]))

    print("=== Detailed Accuracy By Class ===\n")
    print("                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class")
    print(
        "                 {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}     robo".format(
            TP_rate, FP_rate, precision, TP_rate, f_measure, MCC, ROC_area, PRC_area))
    print(
        "                 {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}     human".format(
            1 - TP_rate, 1 - FP_rate, 1 - precision, 1 - TP_rate, 1 - f_measure, 1 - MCC, 1 - ROC_area, 1 - PRC_area))
    print("Weighted Avg.    {:.3f}    {:.3f}    {:.3f}      {:.3f}    {:.3f}      {:.3f}    {:.3f}     {:.3f}\n".format(
        TP_rate, FP_rate, precision, TP_rate, f_measure, MCC, ROC_area, PRC_area))




    print("=== Confusion Matrix ===\n")
    print("   a   b   <-- classified as")
    print("{:>4} {:>4} |   a = robo".format(matrix[0][0], matrix[0][1]))
    print("{:>4} {:>4} |   b = human".format(matrix[1][0], matrix[1][1]))

else:
    print('ARFF file not found at the specified location.')
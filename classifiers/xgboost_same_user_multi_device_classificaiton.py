import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Define the specific user IDs
user_ids = ["0e02379c-4c6b-4c1f-9abd-d54ff1f91050-Oculus Quest", 
            "0e02379c-4c6b-4c1f-9abd-d54ff1f91050-Oculus Quest 2"]

# Define the path to the folder containing CSV files
folder_path = 'input_files/csv_user/'

# List all CSV files in the specified folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Check if there are CSV files in the folder
if len(csv_files) == 0:
    print('No CSV files found in the specified folder.')
else:
    # Initialize lists to hold the data
    all_X = []
    all_y = []

    feature_print_x = []
    feature_print_y = []

    # Load and concatenate all CSV files
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)

        # Ensure the last column is the target variable
        X = data.iloc[:, :-1].values  # All columns except the last one
        y = data.iloc[:, -1].values  # Last column

        # Filter data to include only the specified user IDs
        mask = np.isin(y, user_ids)
        X = X[mask]
        y = y[mask]

        all_X.append(X)
        all_y.append(y)

    # Concatenate all data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    feature_print_x = np.copy(X)
    feature_print_y = np.copy(y)

    # Create a binary target variable
    y_binary = (y == user_ids[0]).astype(int)

    # Print the total number of classes
    unique_classes = np.unique(y_binary)
    num_classes = len(unique_classes)
    print(f"Total number of classes: {num_classes}")
    print(f"Unique classes: {unique_classes}")

    # Check for NaN and infinity values and replace them
    X = np.nan_to_num(X, nan=0.0, posinf=1e308, neginf=-1e308)
    y_binary = np.nan_to_num(y_binary, nan=0.0, posinf=1e308, neginf=-1e308)

    # Normalize the input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Define DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters for binary classification
    params = {
        'objective': 'binary:logistic',  # For binary classification
        'max_depth': 10,
        'eta': 0.1,
        'eval_metric': 'logloss',        # Logarithmic loss
        'tree_method': 'hist',           # Use histogram-based algorithm
        'device': 'cuda'                 # Specify to use GPU
    }

    # Start timer
    start_time = time.time()

    # Train the classifier
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    predictions = bst.predict(dtest)
    predictions_binary = (predictions > 0.5).astype(int)  # Convert probabilities to binary class labels

    # Calculate accuracy for the specific user IDs
    accuracy = accuracy_score(y_test, predictions_binary)

    # Print detailed test results
    print("Detailed Test Results:")
    for i in range(len(y_test)):
        class_name = user_ids[y_test[i]]
        correctly_classified = "yes" if predictions_binary[i] == y_test[i] else "no"
        print(f"Class name: {class_name} - Correctly classified: {correctly_classified}")

    # Print overall accuracy for the specific classes
    print(f"\nOverall Accuracy for classes '{user_ids}': {accuracy * 100:.2f}%")

    # Perform information gain analysis
    info_gain = mutual_info_classif(X_train, y_train, discrete_features=False)
    top_features = np.argsort(info_gain)[-5:][::-1]  # Get indices of top 5 features

    test_index = [631, 14, 76, 25, 200]

    print(top_features)


    # Save detailed feature values for top features using feature_print_x
    with open('feature_details.txt', 'w') as f:
        for user_id, class_value in zip(user_ids, [0, 1]):
            f.write(f"Class {user_id}:\n")
            for feature_index in test_index:
                feature_values = feature_print_x[feature_print_y == user_id, feature_index]
                f.write(f"Column {feature_index} = {feature_values.tolist()}\n")
            f.write("\n")

    # Print information gain for each feature
    print("\nInformation Gain for each feature (column number: info gain):")
    for i, gain in enumerate(info_gain):
        print(f"Column {i}: {gain}")

    print(f"\nTop 5 features by information gain (column numbers): {top_features}")

    # Print the results
    print("Time taken to build model: {:.2f} seconds\n".format(elapsed_time))
    print("=== Stratified cross-validation ===")
    print("=== Summary ===\n")
    print("Correctly Classified Instances    {:>10}     {:.4f} %".format(np.sum(predictions_binary == y_test), accuracy * 100))
    print("Incorrectly Classified Instances  {:>10}     {:.4f} %".format(np.sum(predictions_binary != y_test), (1 - accuracy) * 100))
    print("Kappa statistic                    {:.4f}".format(accuracy))
    print("Total Number of Instances          {:>10}\n".format(y_test.shape[0]))

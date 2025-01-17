import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path, header=None)

def preprocess_data(data):
    """
    Preprocess the dataset by handling inf, NaN, and extremely large values.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number])

    # Replace inf and -inf with NaN
    numeric_cols.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with column means
    numeric_cols.fillna(numeric_cols.mean(numeric_only=True), inplace=True)

    # Cap excessively large values (optional)
    max_threshold = 1e12  # Adjust threshold as needed
    numeric_cols = numeric_cols.clip(upper=max_threshold)

    # Combine cleaned numeric columns with non-numeric columns (if needed)
    non_numeric_cols = data.select_dtypes(exclude=[np.number])
    cleaned_data = pd.concat([numeric_cols, non_numeric_cols], axis=1)

    return cleaned_data

def encode_labels(labels):
    """
    Encode categorical labels into numeric values.

    Args:
        labels (pd.Series): Categorical labels.

    Returns:
        np.ndarray: Encoded numeric labels.
        LabelEncoder: The encoder used for transformation.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def evaluate_combination(data, labels, combination_name):
    """
    Train an XGBoost classifier and evaluate its performance on a given combination of data.

    Args:
        data (pd.DataFrame): Feature matrix.
        labels (pd.Series): Target labels (0 or 1).
        combination_name (str): Name of the combination being evaluated.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Train XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"Metrics for {combination_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {"accuracy": accuracy, "auc": auc, "f1": f1}

# Load and preprocess Combination 1 and Combination 2 datasets
data_comb_1 = preprocess_data(load_data('output_test3.csv'))
data_comb_2 = preprocess_data(load_data('output_test4.csv'))

# Assume the last column is the target variable
X_comb_1 = data_comb_1.iloc[:, :-1]  # Features for Combination 1
y_comb_1_raw = data_comb_1.iloc[:, -1]   # Labels for Combination 1

X_comb_2 = data_comb_2.iloc[:, :-1]  # Features for Combination 2
y_comb_2_raw = data_comb_2.iloc[:, -1]   # Labels for Combination 2

# Encode target labels into numeric values
y_comb_1, encoder_comb_1 = encode_labels(y_comb_1_raw)
y_comb_2, encoder_comb_2 = encode_labels(y_comb_2_raw)

# Evaluate both combinations
metrics_comb_1 = evaluate_combination(X_comb_1, y_comb_1, "Combination 1")
metrics_comb_2 = evaluate_combination(X_comb_2, y_comb_2, "Combination 2")

# Compare results
print("\nComparison:")
if metrics_comb_1["auc"] > metrics_comb_2["auc"]:
    print("Combination 1 is easier to distinguish.")
else:
    print("Combination 2 is easier to distinguish.")

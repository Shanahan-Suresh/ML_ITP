import json
import os
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def custom_tokenizer(s):
    return re.findall(r'\w+|[^\w\s]', s)

# Load the dataset
directory = "C:/Users/shana/source/repos/ML_ITP/Proofs Dataset/"
file_name = "compiled_proofs.json"
file_path = os.path.join(directory, file_name)

with open(file_path, "r", encoding="utf-8") as infile:
    dataset = json.load(infile)

# Prepare the data
premises = [' '.join(proof['premises']) for proof in dataset]
conclusions = [proof['conclusion'] for proof in dataset]
rules = [proof['applied_rules'] for proof in dataset]

# Convert the list of applied rules to binary format
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(rules)

# Convert the data to numerical format using the Bag-of-Words method
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X = vectorizer.fit_transform(premises + conclusions)
scaler = StandardScaler(with_mean=False)  # with_mean=False to work with sparse matrices
X_scaled = scaler.fit_transform(X)
X_premises = X_scaled[:len(premises), :]
X_conclusions = X_scaled[len(premises):, :]


# Combine premises and conclusions features
X_combined = np.hstack([X_premises.toarray(), X_conclusions.toarray()])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_binary, test_size=0.2)

# Train the model
best_k = None
best_metric = None
best_accuracy = 0

for k in range(1, 21):  # Test K values from 1 to 10
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K={k}: {accuracy}")

    # Inside the loop for each K value
    y_pred = knn.predict(X_test)
    print(f"Classification report for K={k}:\n{classification_report(y_test, y_pred, zero_division=0)}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_knn_model = knn

print(f"Best K: {best_k}, Best Accuracy: {best_accuracy}")

# Save the best KNN model weights
joblib.dump(best_knn_model, "best_knn_model.pkl")


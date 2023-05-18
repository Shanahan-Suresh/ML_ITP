import json
import os
import re
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def custom_tokenizer(s):
    return re.findall(r'\w+|[^\w\s]', s)

# Load the dataset
directory = "C:/Users/shana/source/repos/ML_ITP/Proofs Dataset/"
file_name = "compiled_proofs.json"
file_path = os.path.join(directory, file_name)

with open(file_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

premises = [' '.join(item['premises']) for item in data]
conclusions = [item['conclusion'] for item in data]
inference_rules = [item['applied_rules'][0] for item in data]

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X = vectorizer.fit_transform(premises + conclusions)

X_train, X_test, y_train, y_test = train_test_split(X[:len(premises)], inference_rules, test_size=0.2)

# Hyperparameter grid for tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None] + list(range(2, 21)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_decision_tree = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")

# Train the best model on the full dataset
best_decision_tree.fit(X[:len(premises)], inference_rules)

# Make predictions
y_pred = best_decision_tree.predict(X_test)

# Generate classification report
print("Classification Report for Decision Tree:\n", classification_report(y_test, y_pred, zero_division=0))

with open("best_decision_tree.pkl", "wb") as f:
    pickle.dump(best_decision_tree, f)

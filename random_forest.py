import json
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def custom_tokenizer(s):
    return re.findall(r'\w+|[^\w\s]', s)

# Load the dataset
directory = "C:/Users/shana/source/repos/ML_ITP/Proofs Dataset/"
file_name = "compiled_proofs.json"
file_path = os.path.join(directory, file_name)

with open(file_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

premises = [" ".join(item["premises"]) for item in data]
conclusions = [item["conclusion"] for item in data]
applied_rules = [item["applied_rules"] for item in data]

# Vectorize premises and conclusions
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X_premises = vectorizer.fit_transform(premises)
X_conclusions = vectorizer.transform(conclusions)
X = X_premises + X_conclusions

# Binarize the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(applied_rules)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Specify the hyperparameters and their possible values for tuning
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Make predictions
y_pred = best_rf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest model accuracy: {accuracy:.2f}")

# Save the best model to a file
joblib.dump(best_rf, "best_random_forest_model.pkl")

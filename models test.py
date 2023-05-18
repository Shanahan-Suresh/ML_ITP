import joblib
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizer import custom_tokenizer

# Load the KNN, decision tree, and random forest models, vectorizer, and scaler objects
knn_model = joblib.load('Models/model_knn_0.37_model.pkl')
decision_tree_model = joblib.load('Models/model_decision_tree_0.62.pkl')
random_forest_model = joblib.load('Models/model_random_forest_0.56.pkl')
loaded_vectorizer = joblib.load('dataset_vectorizer.pkl')
scaler = joblib.load('dataset_scaler.pkl')

# Create a new instance of TfidfVectorizer with the custom_tokenizer function
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

# Copy the relevant attributes from the loaded vectorizer
vectorizer.vocabulary_ = loaded_vectorizer.vocabulary_
vectorizer.idf_ = loaded_vectorizer.idf_

# Load the dataset
directory = "C:/Users/shana/source/repos/ML_ITP/Proofs Dataset/"
file_name = "compiled_proofs.json"
file_path = os.path.join(directory, file_name)

with open(file_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)


# Function to predict the next inference rule using the decision tree model
def predict_next_rule(premise, conclusion):
    X_premise = vectorizer.transform([premise])
    X_conclusion = vectorizer.transform([conclusion])
    X_combined = X_premise
    predicted_rule = decision_tree_model.predict(X_combined)
    return predicted_rule

def test_sl_model():
    # Iterate over the dataset and display the predicted inference rule for each proof
    for i, item in enumerate(data):
        premises_str = ' '.join(item['premises'])
        conclusion_str = item['conclusion']
        true_rule = item['applied_rules'][0]
    
        predicted_rule = predict_next_rule(premises_str, conclusion_str)
        print(f"Proof {i + 1}:")
        print(f"Premises: {item['premises']}")
        print(f"Conclusion: {conclusion_str}")
        print(f"True rule: {true_rule}")
        print(f"Predicted rule: {predicted_rule[0]}")
        print("\nPress any key to display the prediction for the next proof...")
        input()

# Function to predict the next inference rule using multilabel models (KNN and Random Forest)
def predict_next_rule_multilabel(model, premise, conclusion, flag=False):
    X_premise = vectorizer.transform([premise])
    X_conclusion = vectorizer.transform([conclusion])
    X_combined = np.hstack([X_premise.toarray(), X_conclusion.toarray()])

    if flag :
        X_combined = X_premise
    y_binary_predicted = model.predict(X_combined)
    
    mlb = MultiLabelBinarizer()
    all_rules = ['modus_ponens', 'modus_tollens', 'disjunctive_syllogism', 'hypothetical_syllogism', 'addition', 'simplification']
    mlb.fit([all_rules])
    
    predicted_rules = mlb.inverse_transform(y_binary_predicted)
    return predicted_rules

def test_ml_models():
    # Iterate over the dataset and display the predicted inference rules for each proof using KNN and Random Forest models
    for i, item in enumerate(data):
        premises_str = ' '.join(item['premises'])
        conclusion_str = item['conclusion']
        true_rule = item['applied_rules'][0]
    
        predicted_rule_knn = predict_next_rule_multilabel(knn_model, premises_str, conclusion_str)
        predicted_rule_rf = predict_next_rule_multilabel(random_forest_model, premises_str, conclusion_str, flag=True)
    
        print(f"Proof {i + 1}:")
        print(f"Premises: {item['premises']}")
        print(f"Conclusion: {conclusion_str}")
        print(f"True rule: {true_rule}")
        print(f"KNN Predicted rules: {predicted_rule_knn[0]}")
        print(f"Random Forest Predicted rules: {predicted_rule_rf[0]}")
        print("\nPress any key to display the predictions for the next proof...")
        input()

def main():
    test_ml_models()

main()
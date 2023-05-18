import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizer import custom_tokenizer
import torch
import torch.nn as nn
import pickle


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

def predict_inference_rules(model, premises_list, conclusion, model_type='other', combine_features=True, apply_scaler=True, multilabel=True):
    premises_str = ' '.join(premises_list)

    if model_type == 'ann':
        X_premises = ann_vectorizer.transform([premises_str])
        X_conclusion = ann_vectorizer.transform([conclusion])
        X_combined = X_premises + X_conclusion
    else:
        X_premises = vectorizer.transform([premises_str])
        X_conclusion = vectorizer.transform([conclusion])

    if apply_scaler:
        X_premises = scaler.transform(X_premises)
        X_conclusion = scaler.transform(X_conclusion)

    if combine_features:
        X_combined = np.hstack([X_premises.toarray(), X_conclusion.toarray()])
    else:
                X_combined = X_premises

    if model_type == 'ann' or model_type == 'cnn':
        X_combined = torch.tensor(X_combined, dtype=torch.float32)

        if model_type == 'cnn':
            X_combined = X_combined.unsqueeze(1)

        with torch.no_grad():
            y_predicted = model(X_combined)

        y_predicted = (y_predicted.sigmoid() > 0.5).numpy()
    else:
        y_predicted = model.predict(X_combined)

    if multilabel:
        mlb = MultiLabelBinarizer()

        # All possible rules
        all_rules = ['modus_ponens', 'modus_tollens', 'disjunctive_syllogism', 'hypothetical_syllogism', 'addition', 'simplification']
        mlb.fit([all_rules])

        predicted_rules = mlb.inverse_transform(y_predicted)
        return predicted_rules[0]

    else:
        predicted_rules = y_predicted

    return predicted_rules




def knn_predict_inference_rules(premises_list, conclusion):
    return predict_inference_rules(knn_model, premises_list, conclusion, combine_features=True, apply_scaler=True, multilabel=True)

def decision_tree_predict_inference_rules(premises_list, conclusion):
    return predict_inference_rules(decision_tree_model, premises_list, conclusion, combine_features=False, apply_scaler=False, multilabel=False)

def random_forest_predict_inference_rules(premises_list, conclusion):
    return predict_inference_rules(random_forest_model, premises_list, conclusion, combine_features=False, apply_scaler=False, multilabel=True)

def ann_predict_inference_rules(premises_list, conclusion):
    return predict_inference_rules(ann_model, premises_list, conclusion, model_type='ann', combine_features=True, apply_scaler=True, multilabel=True)

def cnn_predict_inference_rules(premises_list, conclusion):
    return predict_inference_rules(cnn_model, premises_list, conclusion, model_type='cnn', combine_features=True, apply_scaler=True, multilabel=True)

class ProofPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProofPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ProofPredictorCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProofPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the saved ANN model state dictionary
ann_model_state_dict = torch.load('Models/model_ann_0.80.pth')

# Load the vectorizer used during training
with open('ann_vectorizer.pkl', 'rb') as f:
    ann_vectorizer = pickle.load(f)

# Set the input size, hidden size, and output size
hidden_size = 128
output_size = 6 # Number of trained rules

# Instantiate the ANN model
ann_input_size = 34  # You should set this to the correct input size based on your ANN training
ann_model = ProofPredictor(ann_input_size, hidden_size, output_size)

# Load the saved state dictionary into the ANN model
ann_model.load_state_dict(ann_model_state_dict)

# Set the ANN model to evaluation mode
ann_model.eval()


# Set the input size, hidden size, and output size [FOR CNN]
cnn_input_size = 34  #Obtained from X train set in model training module
hidden_size = 128
output_size = 6 #Number of trained rules

# Instantiate cnn model
cnn_model = ProofPredictorCNN(cnn_input_size, hidden_size, output_size)

# Load the saved state dictionaries
cnn_model.load_state_dict(torch.load('Models/model_cnn_0.81.pth'))

# Set CNN model to evaluation mode
cnn_model.eval()


'''
premises_input = ["(((Q ¬ (¬Z)) ¬ ((¬Z) ¬ Z)) → Z)", "(¬Z)"]
conclusion_input = "(¬((Q ¬ (¬Z)) ¬ ((¬Z) ¬ Z)))"
predicted_rules = cnn_predict_inference_rules(premises_input, conclusion_input)
print(predicted_rules)
'''


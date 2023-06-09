import torch
import torch.nn as nn
import torch.optim as optim
import json
import re
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

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

# Create a dictionary to map inference rule names to integer indices
unique_rules = list(set(rule for rules in applied_rules for rule in rules))
rule_to_idx = {rule: idx for idx, rule in enumerate(unique_rules)}

# Binarize the labels
mlb = MultiLabelBinarizer(classes=unique_rules)
y = mlb.fit_transform(applied_rules)

# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)

# Convert the dataset to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Add channel dimension to the input data
X_train_tensor = X_train_tensor.unsqueeze(1)
X_val_tensor = X_val_tensor.unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1)

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

# Set the input size, hidden size, and output size
input_size = X_train.shape[1]
hidden_size = 128
output_size = len(rule_to_idx)

# Create the model
model = ProofPredictorCNN(input_size, hidden_size, output_size)

# Set the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 55
batch_size = 1

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    train_loss = 0
    
    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x = X_train_tensor[indices, :]
        batch_y = y_train_tensor[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Compute the validation loss
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Store losses for this epoch
    train_losses.append(train_loss / (X_train.shape[0] // batch_size))
    val_losses.append(val_loss.item())


    # Print the losses for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = torch.sigmoid(test_outputs) > 0.5
    accuracy = (predicted == y_test_tensor).sum().item() / (len(y_test) * len(unique_rules))
    print(f"Accuracy: {accuracy}")

# Plot the train/validation loss graph
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get the ground truth labels
true_labels = mlb.inverse_transform(y_test)
# Get the predicted labels
pred_labels = mlb.inverse_transform(predicted.numpy())

# Compute the classification report
report = classification_report(y_test, predicted.numpy(), target_names=unique_rules)

print("\nClassification Report:")
print(report)

# Save the model
torch.save(model.state_dict(), "cnn_model.pth")


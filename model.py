import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load and preprocess the data
df = pd.read_csv('diabetes.csv')
df['Outcome'] = np.where(df['Outcome'] == 1, "Diabetic", "No Diabetic")
X = df.drop('Outcome', axis=1).values  # Independent features
y = df['Outcome'].values  # Dependent feature

# Convert categorical labels to numeric
y = np.where(y == "Diabetic", 1, 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the ANN model
class ANN_Model(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

# Instantiate and train the model
torch.manual_seed(20)
model = ANN_Model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500
final_losses = []

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss.item())
    if (epoch + 1) % 10 == 1:
        print(f"Epoch {epoch + 1} - Loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss function
plt.plot(range(epochs), final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model
with torch.no_grad():
    predictions = [model(data).argmax().item() for data in X_test]

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

score = accuracy_score(y_test, predictions)
print(f"Accuracy: {score}")

# Save the model
torch.save(model, 'diabetes.pt')

# flask_app.py

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

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

# Load the trained model
model = torch.load('diabetes.pt')
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    new_data = torch.FloatTensor([float(data['Pregnancies']),
                                  float(data['Glucose']),
                                  float(data['BloodPressure']),
                                  float(data['SkinThickness']),
                                  float(data['Insulin']),
                                  float(data['BMI']),
                                  float(data['DiabetesPedigreeFunction']),
                                  float(data['Age'])])
    with torch.no_grad():
        prediction = model(new_data).argmax().item()
    outcome = "Diabetic" if prediction == 1 else "No Diabetic"
    return jsonify({"Prediction": outcome})

if __name__ == '__main__':
    app.run(debug=True)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(" Loading Data ")
df = pd.read_csv("sales_data.csv")

# Inputs for AI
X = df[['competitor_price', 'our_price', 'is_weekend']].values

# Output for AI
y = df[['units_sold']].values

# Scaling: AI learns faster if numbers are small (between -1 and 1).
# E.g. treats $50.00 as just "0.5" relative to other prices.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pytorch Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Neural Network
class PricingModel(nn.Module):
    def __init__(self):
        super(PricingModel, self).__init__()
        self.layer1 = nn.Linear(3,64) # Takes our 3 inputs and expands to 64 neurons
        self.layer2 = nn.Linear(64,64) # Deep Learning
        self.output = nn.Linear(64,1)

        self.relu = nn.ReLU() # f(x) = max(0,x)

    # Forward Propagation
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)

        return x

model = PricingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05) # Updates weights to help learn, learning rate is 0.01

print(" Starting Training ")
epochs = 2000

for i in range(epochs):
    predictions = model(X_tensor)
    loss = criterion(predictions,y_tensor)

    # Backpropagation
    optimizer.zero_grad() # Clear old calculations
    loss.backward()       # Calculate how much to change each connection
    optimizer.step()      # Update the weights

    if (i+1) % 100 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}")

print(" Training Complete ")

# Save the state_dict which is the learned memory
torch.save(model.state_dict(), "pricing_model.pth")

import joblib
joblib.dump(scaler, "scaler.save") 

print("Model saved as 'pricing_model.pth'")
print("Scaler saved as 'scaler.save'")
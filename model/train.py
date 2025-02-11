import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from covid_model import CovidPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

csv_path = os.path.join(BASE_DIR, "output", "3.feature_engineering", "covid_feature_engineered.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File {csv_path} tidak ditemukan!")

df = pd.read_csv(csv_path)

feature_columns = ['confirmed_cases', 'deaths', 'recoveries', 'total_confirmed', 'total_deaths']
target_column = ['case_fatality_rate']

X = df[feature_columns]
y = df[target_column]

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

model = CovidPredictor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model_dir = os.path.join(BASE_DIR, "output", "deep_learning")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "covid_model.pth")

torch.save(model.state_dict(), model_path)
print(f"✅ Model berhasil disimpan di `{model_path}`")

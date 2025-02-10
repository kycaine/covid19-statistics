import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from covid_model import CovidPredictor

csv_path = "../output/feature_engineering/covid_feature_engineered.csv"
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

os.makedirs("output", exist_ok=True)

torch.save(model.state_dict(), "output/covid_model.pth")
print("✅ Model berhasil disimpan di `output/covid_model.pth`")

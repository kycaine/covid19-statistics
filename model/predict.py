import os
import torch
import pandas as pd
from covid_model import CovidPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

csv_path = os.path.join(BASE_DIR, "output", "3.feature_engineering", "covid_feature_engineered.csv")
model_path = os.path.join(BASE_DIR, "output", "deep_learning", "covid_model.pth")
predictions_path = os.path.join(BASE_DIR, "output", "deep_learning", "predictions.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File {csv_path} tidak ditemukan!")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model {model_path} tidak ditemukan!")

df = pd.read_csv(csv_path)
feature_columns = ['confirmed_cases', 'deaths', 'recoveries', 'total_confirmed', 'total_deaths']

X = df[feature_columns]
X_tensor = torch.tensor(X.values, dtype=torch.float32)

model = CovidPredictor()
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    predictions = model(X_tensor)

df['Predicted_Case_Fatality_Rate'] = predictions.numpy()
df.to_csv(predictions_path, index=False)

print(f"✅ Prediksi disimpan di `{predictions_path}`")

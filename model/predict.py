import torch
import pandas as pd
from covid_model import CovidPredictor

df = pd.read_csv("../output/feature_engineering/covid_feature_engineered.csv")  # Sesuaikan path
feature_columns = ['confirmed_cases', 'deaths', 'recoveries', 'total_confirmed', 'total_deaths']

X = df[feature_columns]
X_tensor = torch.tensor(X.values, dtype=torch.float32)

model = CovidPredictor() 
model.load_state_dict(torch.load("output/covid_model.pth"))
model.eval()

with torch.no_grad():
    predictions = model(X_tensor)

df['Predicted_Case_Fatality_Rate'] = predictions.numpy()
df.to_csv("output/predictions.csv", index=False)
print("✅ Prediksi disimpan di `model/output/predictions.csv`")

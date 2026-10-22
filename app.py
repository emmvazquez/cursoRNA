import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Arquitectura identica a la definida previamente
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

app = FastAPI()

# Permitir solicitudes locales desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifier().to(device)

# Cargar pesos guardados
try:
    model.load_state_dict(torch.load("digit_model.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    print("Aviso: digit_model.pth no encontrado. Entrena el modelo y guarda los pesos primero.")

class PredictRequest(BaseModel):
    # Matriz 8x8 con valores normalizados entre 0.0 y 1.0
    matrix: List[List[float]]

@app.post("/predict")
def predict_digit(data: PredictRequest):
    tensor_input = torch.tensor(data.matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = int(torch.argmax(logits, dim=1).item())

    return {
        "prediction": predicted_class,
        "probabilities": probabilities
    }
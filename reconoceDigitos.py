import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Carga y preparacion de matrices de entrada
digits = load_digits()
X_matrices = digits.images  # Dimension: (1797, 8, 8)
y_labels = digits.target    # Clases: 0 a 9

# Normalizacion al rango [0, 1] dividiendo entre el valor maximo de pixel (16)
X_norm = (X_matrices / 16.0).astype(np.float32)

# Division en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# Conversion a tensores y creacion de DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Definicion de la arquitectura
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Flatten convierte la matriz de (8, 8) en un vector de 64 elementos
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 10)  # 10 clases de salida (digitos 0 al 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 3. Inicializacion de modelo, criterio y optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 4. Ciclo de entrenamiento
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Monitoreo cada 10 epocas
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoca [{epoch+1:02d}/{epochs:02d}] - Perdida entrenamiento: {epoch_loss:.4f}")

# 5. Evaluacion en el conjunto de prueba
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_images)
        predictions = torch.argmax(outputs, dim=1)
        
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

accuracy = (correct / total) * 100
print(f"\nPrecision global en prueba: {accuracy:.2f}%")

# 6. Prueba puntual con una matriz individual
sample_idx = 7
sample_matrix = X_test[sample_idx]  # Matriz de 8x8
sample_true_label = y_test[sample_idx]

# Formatear la matriz con dimension de lote: (1, 8, 8)
input_tensor = torch.tensor(sample_matrix, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(input_tensor)
    predicted_digit = torch.argmax(output, dim=1).item()

print(f"\nPrueba individual:")
print(f"Etiqueta real: {sample_true_label} | Prediccion de la red: {predicted_digit}")
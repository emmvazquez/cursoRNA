import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Preparacion de datos sinteticos tabulares
X_raw, y_raw = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_classes=3,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Escalado estandar de caracteristicas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Conversion a tensores de PyTorch
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 2. Definicion de arquitectura con regularizacion integrada
class TabularClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# 3. Configuracion de dispositivo, modelo y optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularClassifier(in_features=20, num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

# 4. Ciclo de entrenamiento con Early Stopping
epochs = 100
patience = 8
best_val_loss = float("inf")
patience_counter = 0
best_model_weights = None

for epoch in range(epochs):
    # Modo entrenamiento
    model.train()
    running_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch_x.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)

    # Modo evaluacion
    model.eval()
    running_val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_val_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()

    val_loss = running_val_loss / len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)

    # Monitoreo
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoca [{epoch+1:03d}/{epochs:03d}] | Perdida Train: {train_loss:.4f} | Perdida Val: {val_loss:.4f} | Acc Val: {val_acc:.4f}")

    # Logica de Early Stopping y guardado de checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nParada temprana activada en epoca {epoch+1}. Mejor Perdida Val: {best_val_loss:.4f}")
            break

# Cargar los mejores pesos encontrados
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print("Modelo restaurado con el mejor punto de control.")
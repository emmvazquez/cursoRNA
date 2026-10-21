import numpy as np


class SimpleMLP:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5):
        self.lr = learning_rate

        # Inicializacion de pesos con valores aleatorios pequeños y sesgos en cero
        np.random.seed(42)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.5
        self.b1 = np.zeros((hidden_dim, 1))

        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.5
        self.b2 = np.zeros((output_dim, 1))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def sigmoid_derivative(cls, a):
        # Derivada expresada en funcion de la activacion ya calculada
        return a * (1.0 - a)

    def forward(self, X):
        # X tiene dimension (n_features, m)
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y):
        m = X.shape[1]

        # Derivada del error respecto a z2 (combinando BCE + Sigmoide)
        dZ2 = self.A2 - y
        dW2 = (1.0 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Propagacion del gradiente hacia la capa oculta
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = (1.0 / m) * np.dot(dZ1, X.T)
        db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Actualizacion de parametros
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=5000):
        # Entrada en formato (n_features, m)
        X_transposed = X.T
        y_transposed = y.reshape(1, -1)

        for epoch in range(epochs):
            predictions = self.forward(X_transposed)
            self.backward(X_transposed, y_transposed)

            if epoch % 1000 == 0:
                # Calculo de Binary Cross Entropy
                loss = -np.mean(
                    y_transposed * np.log(predictions + 1e-8)
                    + (1.0 - y_transposed) * np.log(1.0 - predictions + 1e-8)
                )
                print(f"Epoca {epoch:4d} | Perdida: {loss:.5f}")

    def predict(self, X):
        probabilities = self.forward(X.T)
        return (probabilities >= 0.5).astype(int).flatten()


# Evaluacion con la compuerta XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

mlp = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.8)
print("Entrenando red MLP sobre XOR...")
mlp.fit(X_xor, y_xor, epochs=5001)

preds = mlp.predict(X_xor)
print("\nResultados finales sobre XOR:")
for sample, target, pred in zip(X_xor, y_xor, preds):
    print(f"Entrada: {sample} | Esperado: {target} | Prediccion: {pred}")
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            errors_in_epoch = 0
            for xi, target in zip(X, y):
                # Combinación lineal
                linear_output = np.dot(xi, self.weights) + self.bias
                # Función escalón
                prediction = 1 if linear_output >= 0 else 0

                error = target - prediction
                if error != 0:
                    update = self.lr * error
                    self.weights += update * xi
                    self.bias += update
                    errors_in_epoch += 1

            # Criterio de parada temprana si no hay errores
            if errors_in_epoch == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


# Pruebas con compuertas logicas
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Compuerta AND (linealmente separable)
y_and = np.array([0, 0, 0, 1])
model_and = Perceptron(learning_rate=0.1, n_iterations=20)
model_and.fit(X, y_and)
preds_and = model_and.predict(X)

print("Resultados Compuerta AND:")
print(f"Predicciones: {preds_and} | Esperado: {y_and}")
print(f"Pesos finales: {model_and.weights}, Sesgo final: {model_and.bias}\n")

# Compuerta XOR (no separable linealmente)
y_xor = np.array([0, 1, 1, 0])
model_xor = Perceptron(learning_rate=0.1, n_iterations=50)
model_xor.fit(X, y_xor)
preds_xor = model_xor.predict(X)

print("Resultados Compuerta XOR (Demostracion del limite del perceptron):")
print(f"Predicciones: {preds_xor} | Esperado: {y_xor}")
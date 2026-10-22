import numpy as np


class BaseMLP:
    def __init__(self, layer_sizes, seed=42):
        self.layer_sizes = tuple(layer_sizes)
        self.loss_history = []
        self.initialize_parameters(seed)

    def initialize_parameters(self, seed):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for layer in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[layer]
            output_size = self.layer_sizes[layer + 1]
            scale = np.sqrt(2 / input_size) if layer < len(self.layer_sizes) - 2 else np.sqrt(1 / input_size)
            weight = rng.standard_normal((input_size, output_size)) * scale
            self.weights.append(weight.astype(np.float32))
            self.biases.append(np.zeros((1, output_size), dtype=np.float32))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def cross_entropy(output, y):
        prob = output[np.arange(len(y)), y]
        return -np.log(np.clip(prob, 1e-12, 1)).mean()

    def backward(self, outputs, hidden_inputs, y):
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        delta = outputs[-1].copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer] = outputs[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (hidden_inputs[layer - 1] > 0)
        return grad_weights, grad_biases

    def sgd_step(self, grad_weights, grad_biases, learning_rate):
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grad_weights[layer]
            self.biases[layer] -= learning_rate * grad_biases[layer]

    def fit(self, x, y, epochs=30, batch_size=64, learning_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.loss_history = []
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            total_loss = 0.0
            correct = 0
            for start in range(0, len(x), batch_size):
                batch = order[start:start + batch_size]
                x_batch, y_batch = x[batch], y[batch]
                outputs, hidden_inputs = self.forward(x_batch)
                output = outputs[-1]
                total_loss += self.cross_entropy(output, y_batch) * len(y_batch)
                correct += np.sum(np.argmax(output, axis=1) == y_batch)
                grad_weights, grad_biases = self.backward(outputs, hidden_inputs, y_batch)
                self.sgd_step(grad_weights, grad_biases, learning_rate)
            loss = float(total_loss / len(x))
            self.loss_history.append(loss)
            print(f"Epoch {epoch + 1}/{epochs} | loss: {loss:.4f} | accuracy: {correct / len(x):.4f}")
        return self

    def predict_proba(self, x):
        outputs, _ = self.forward(x)
        return outputs[-1]

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

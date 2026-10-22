from .base import BaseMLP


class MLP3Layer(BaseMLP):
    def __init__(self, input_size=784, hidden_sizes=(512, 256), num_classes=10, seed=42):
        if len(hidden_sizes) != 2:
            raise ValueError("MLP3Layer needs exactly two hidden layers")
        super().__init__([input_size, *hidden_sizes, num_classes], seed=seed)

    def forward(self, x):
        z1 = x @ self.weights[0] + self.biases[0]
        hidden1 = self.relu(z1)
        z2 = hidden1 @ self.weights[1] + self.biases[1]
        hidden2 = self.relu(z2)
        logits = hidden2 @ self.weights[2] + self.biases[2]
        output = self.softmax(logits)
        return [x, hidden1, hidden2, output], [z1, z2]

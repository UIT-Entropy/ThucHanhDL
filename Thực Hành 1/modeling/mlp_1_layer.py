from .base import BaseMLP


class MLP1Layer(BaseMLP):
    def __init__(self, input_size=784, num_classes=10, seed=42):
        super().__init__([input_size, num_classes], seed=seed)

    def forward(self, x):
        logits = x @ self.weights[0] + self.biases[0]
        output = self.softmax(logits)
        return [x, output], []

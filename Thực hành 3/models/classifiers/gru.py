from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GruModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, n_layer=5, n_labels=4, padding_idx=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, dropout=dropout if n_layer > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(self, inputs, lengths):
        embeddings = self.embedding(inputs)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.classifier(self.dropout(hidden[-1]))

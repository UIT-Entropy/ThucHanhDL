from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstmModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, n_layer=5, n_labels=21, padding_idx=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, bidirectional=True, dropout=dropout if n_layer > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_labels)

    def forward(self, inputs, lengths):
        embeddings = self.embedding(inputs)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        features, _ = pad_packed_sequence(encoded, batch_first=True, total_length=inputs.size(1))
        return self.classifier(self.dropout(features))

# ## LSTM implementation
# `EnsembleModel` is an inherited class that implements the ensemble model consisting of an LSTM network, which produces
# a context vector, and a feedforward neural network, which takes the context vector and outputs a binary prediction.

import torch


class EnsembleModel(torch.nn.Module):
    def __init__(self, embeddings_tensor,
                 hidden_size=256,
                 dropout=.5,
                 embedding_size=300, ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings_tensor)
        self.lstm = torch.nn.LSTM(embedding_size,
                                  hidden_size,
                                  batch_first=True,
                                  bidirectional=False,
                                  num_layers=3,
                                  dropout=dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.embedding(input)
        _, hidden = self.lstm(output)
        hidden = hidden[0]
        output = self.linear(hidden[-1])
        output = self.sigmoid(output)
        return output

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class NmosFcn(pl.LightningModule):
    def __init__(self, input_size, encoder_hidden_size, arg_comp_hidden_size, arg_comp_output_size):
        super(NmosFcn, self).__init__()
        self.__dict__.update(locals())
        self.head = CausalAttention(input_size, encoder_hidden_size)
        self.MLP = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.BatchNorm1d(encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, arg_comp_hidden_size),
            nn.ReLU(),
            nn.Linear(arg_comp_hidden_size, arg_comp_output_size),
        )

    def forward(self, seq):
        seq1, seq2 = seq["t1"], seq["t2"]
        seq1 = seq1.unsqueeze(1)
        seq2 = seq2.unsqueeze(1)

        seq = torch.concat((seq1, seq2), dim=1).permute(0, 2, 1)
        # do causal attention on every point in the sequence
        seq = torch.sum(torch.stack([self.head(seq[:, :i]).squeeze()/i for i in range(1, seq.shape[1])]), dim=0)
        seq = self.MLP(seq)
        return seq


class CausalAttention(pl.LightningModule):
    def __init__(self, input_size, encoder_hidden_size):
        super(CausalAttention, self).__init__()
        self.q = nn.Linear(input_size, encoder_hidden_size)
        self.k = nn.Linear(encoder_hidden_size, 1)
        self.v = nn.Linear(input_size, encoder_hidden_size)

    def forward(self, seq):
        # seq shape: (batch_size, seq_len, encoder_hidden_size)
        score = self.k(F.tanh(self.q(seq)))  # B*L*1
        score = F.softmax(score, dim=1)
        seq = self.v(score * seq).sum(dim=1)
        return seq



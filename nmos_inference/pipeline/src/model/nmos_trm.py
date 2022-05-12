import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# Transformer Network
class NmosTrm(pl.LightningModule):
    # TODO compositional network init
    def __init__(self, input_size=None, input_length=None,
                 encoder_name=None, encoder_hidden_size=None, encoder_num_layers=None,
                 arg_comp_hidden_size=None, arg_comp_output_size=None,
                 event_comp_hidden_size=None, event_comp_output_size=None,
                 dropout=None, margin=None,
                 aug=False, aug_prob=None):
        super().__init__()
        self.save_hyperparameters()
        self.__dict__.update(locals())
        # change the position encoding length due to the input length
        if aug == True and isinstance(self.aug_prob, list):
            if self.aug_prob[0] > 0:
                self.input_length = int(self.input_length * self.aug_prob[0])

        # self.emb = nn.Embedding(512, encoder_hidden_size)
        self.emb = nn.Linear(input_size, encoder_hidden_size)
        self.pos_encoder = PositionalEncoding(encoder_hidden_size, self.input_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_hidden_size, nhead=4,
                                                   dim_feedforward=self.encoder_hidden_size,
                                                   batch_first=True)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_num_layers)
        self.dropout = nn.Dropout(self.dropout)

        self.linear = nn.Linear(self.encoder_hidden_size, self.arg_comp_output_size)

    def forward(self, seq):
        # move input to GPU
        seq1, seq2 = seq["t1"], seq["t2"]

        # encode (batch_size, seq_len, input_size)
        seq1 = seq1.unsqueeze(1)
        seq2 = seq2.unsqueeze(1)

        seq = torch.concat((seq1, seq2), dim=1).permute(0, 2, 1)
        seq = self.emb(seq)
        seq = self.pos_encoder(seq)
        seq = self.trm(seq)
        seq = self.dropout(seq)
        # seq = torch.mean(seq, dim=1)

        seq = self.linear(seq)
        return seq


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.q = nn.Linear(input_size, hidden_size)
        self.k = nn.Linear(input_size, 1)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        # seq : (B,L,H*4)
        score = self.k(nn.functional.tanh(self.q(seq)))  # B*L*1
        score = nn.functional.softmax(score, dim=1)
        seq = torch.sum(score * seq, dim=1)
        return seq

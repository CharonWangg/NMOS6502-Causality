import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Compositional Network
class NmosLstm(pl.LightningModule):
    # TODO compositional network init
    def __init__(self, input_size=None,
                 encoder_name=None, encoder_hidden_size=None, encoder_num_layers=None,
                 arg_comp_hidden_size=None, arg_comp_output_size=None,
                 event_comp_hidden_size=None, event_comp_output_size=None,
                 dropout=None, margin=None):
        super().__init__()
        self.__dict__.update(locals())
        self.lstm = nn.LSTM(self.input_size, self.encoder_hidden_size, self.encoder_num_layers,
                            batch_first=True, dropout=self.dropout, bidirectional=True)
        # self.bn1 = nn.BatchNorm1d(self.encoder_hidden_size)
        self.linear = nn.Linear(self.arg_comp_hidden_size*2, self.arg_comp_output_size)
        # self.att = AttentionHead(self.event_comp_hidden_size, self.event_comp_output_size)

    def forward(self, seq):
        # move input to GPU
        seq1, seq2 = seq["t1"].to(self.device), seq["t2"].to(self.device)

        # encode
        seq1 = seq1.unsqueeze(1)
        seq2 = seq2.unsqueeze(1)
        seq = torch.concat((seq1, seq2), dim=1).permute(0, 2, 1)
        seq, (h, c) = self.lstm(seq)
        # seq = self.bn1(seq)

        # seq = torch.mean(seq, dim=1)

        # seq1, (h1,c1) = self.lstm1(seq1.unsqueeze(-1))
        # seq2, (h2,c2) = self.lstm2(seq2.unsqueeze(-1))
        # Simple concat
        # seq = torch.concat((seq1,seq2),dim=-1)[:,-1,:]
        # Attention
        # seq = torch.concat((seq1,seq2),dim=1)
        # seq = self.att(seq)
        # seq = self.fc1(seq)
        seq = self.linear(seq)
        return seq

class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.q = nn.Linear(input_size, hidden_size)
        self.k = nn.Linear(hidden_size, 1)
        self.v = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq):
        # seq : (B,L,H*4)
        score = self.k(nn.functional.tanh(self.q(seq)))  # B*L*1
        score = nn.functional.softmax(score, dim=1)
        seq = self.v(score * seq)
        return seq
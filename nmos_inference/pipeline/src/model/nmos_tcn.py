import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl

# TCN Network
class NmosTcn(pl.LightningModule):
    # TODO compositional network init
    def __init__(self, input_size=None,
                 encoder_name=None, encoder_hidden_size=None, encoder_num_layers=None,
                 arg_comp_hidden_size=None, arg_comp_output_size=None,
                 event_comp_hidden_size=None, event_comp_output_size=None,
                 dropout=0.2, margin=None):
        super().__init__()
        self.__dict__.update(locals())
        self.encoder = TemporalConvNet(self.input_size, self.encoder_hidden_size, dropout=self.dropout)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.encoder_hidden_size[-1], self.arg_comp_output_size)

    def forward(self, seq):
        # move input to GPU
        seq1, seq2 = seq["t1"].to(self.device), seq["t2"].to(self.device)

        # encode
        seq1 = seq1.unsqueeze(1)
        seq2 = seq2.unsqueeze(1)
        seq = torch.concat((seq1, seq2), dim=1)  # (batch_size, 2,seq_len)
        seq = self.encoder(seq).permute(0, 2, 1)  # (batch_size, seq_len, encoder_hidden_size)
        seq = self.linear(seq)  # (batch_size, seq_len, arg_comp_output_size)
        return seq



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNExtractor(nn.Module):
    ''' A simple 2-layer CNN extractor for acoustic feature down-sampling'''

    def __init__(self, input_dim, out_dim):
        super(CNNExtractor, self).__init__()

        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
        )

    def forward(self, feature, feat_len):
        # Fixed down-sample ratio
        feat_len = feat_len//4
        # Channel first
        feature = feature.transpose(1,2) 
        # Foward
        feature = self.extractor(feature)
        # Channel last
        feature = feature.transpose(1, 2)

        return feature, feat_len


class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.out_dim = sample_rate * \
            rnn_out_dim if sample_rate > 1 and sample_style == 'concat' else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj

        if self.sample_style not in ['drop', 'concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size, timestep, feature_dim = output.shape
            x_len = x_len//self.sample_rate

            if self.sample_style == 'drop':
                # Drop the unselected timesteps
                output = output[:, ::self.sample_rate, :].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep % self.sample_rate != 0:
                    output = output[:, :-(timestep % self.sample_rate), :]
                output = output.contiguous().view(batch_size, int(
                    timestep/self.sample_rate), feature_dim*self.sample_rate)

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len



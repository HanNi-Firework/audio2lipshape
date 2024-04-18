import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model.util import init_weights, init_gate
from model.module import CNNExtractor, RNNLayer


class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''

    def __init__(self, input_size, vocab_size, init_adadelta, ctc_weight):
        super(ASR, self).__init__()

        # Setup
        assert 0 <= ctc_weight <= 1
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None

        # Modules
        self.encoder = Encoder(input_size)

        self.ctc_layer = nn.Linear(self.encoder.out_dim, vocab_size)

        # Init
        if init_adadelta:
            self.apply(init_weights)
            if self.enable_att:
                for l in range(self.decoder.layer):
                    bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
                    bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
            self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append(
                '           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.encoder.cnn:
            msg.append(
                '           | CNN Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.enable_ctc:
            msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
                self.ctc_weight))
        if self.enable_att:
            msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
                self.attention.mode, 1 - self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len,  get_dec_state=False):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape 
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxL] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
        # bs = audio_feature.shape[0]
        #
        # Encode
        encode_feature, encode_len = self.encoder(audio_feature, feature_len)

        # CTC based decoding

        ctc_output = F.log_softmax(self.ctc_layer(encode_feature), dim=-1)

        return ctc_output, encode_len,


class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''

    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout

        # Init
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        self.layers = getattr(nn, module)(
            input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim * self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim * self.layer)

    def forward(self, x):
        ''' Decode and transform into vocab '''
        if not self.training:
            self.layers.flatten_parameters()
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        char = self.char_trans(self.final_dropout(x))
        return char, x


class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, module="LSTM", bidirection=False):
        super(Encoder, self).__init__()

        # Hyper-parameters checking

        self.sample_rate = 1
        self.dim= [512, 512, 512, 512, 512]
        self.dropout = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.layer_norm = [True, True, True, True, True]
        self.proj = [True, True, True, True, True]
        self.sample_rate = [1, 1, 1, 1, 1]
        self.sample_style = 'drop'
        assert len(self.sample_rate) == len(self.dropout), 'Number of layer mismatch'
        assert len(self.dropout) == len(self.dim), 'Number of layer mismatch'
        num_layers = len(self.dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        # Prenet on audio feature

        cnn_extractor = CNNExtractor(input_size, out_dim=self.dim[0])
        module_list.append(cnn_extractor)
        input_dim = cnn_extractor.out_dim
        self.sample_rate = self.sample_rate * 4

        # Recurrent encoder
        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, self.dim[l], bidirection, self.dropout[l], self.layer_norm[l],
                                            self.sample_rate[l], self.sample_style, self.proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate * self.sample_rate[l]
        else:
            raise NotImplementedError

        # Build model
        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len

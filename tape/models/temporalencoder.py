# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import VIBE_DATA_DIR
import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)

class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights) 
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1).squeeze()
        return representations, scores

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            hidden_size=2048,
            add_linear=True,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual
        self.self_attention = SelfAttention(attention_size=2048,
                                       layers=n_layers,
                                       dropout=.0)

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        out,scores=self.self_attention(y)

        #out = torch.mul(y.permute(1,0,2), scores[:, :, None])
        #out = torch.sum(out, dim=1)  # N x 2048
        
        return out.reshape(y.shape[0],2048) #y
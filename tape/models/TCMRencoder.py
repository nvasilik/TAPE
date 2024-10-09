
import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores


class TemporalEncoderTCMR(nn.Module):
    def __init__(
            self,
            n_layers=2,
            seq_len=16,
            hidden_size=1024
    ):
        super(TemporalEncoderTCMR, self).__init__()

        self.gru_cur = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers
        )
        self.gru_bef = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.gru_aft = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.mid_frame = int(seq_len/2)
        self.hidden_size = hidden_size

        self.linear_cur = nn.Linear(hidden_size * 2, 2048)
        self.linear_bef = nn.Linear(hidden_size, 2048)
        self.linear_aft = nn.Linear(hidden_size, 2048)

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        # NTF -> TNF
        y, state = self.gru_cur(x.permute(1,0,2))  # y: Tx N x (num_dirs x hidden size)

        x_bef = x[:, :self.mid_frame]
        x_aft = x[:, self.mid_frame+1:]
        x_aft = torch.flip(x_aft, dims=[1])
        y_bef, _ = self.gru_bef(x_bef.permute(1,0,2))
        y_aft, _ = self.gru_aft(x_aft.permute(1,0,2))

        # y_*: N x 2048
        y_cur = self.linear_cur(F.relu(y[self.mid_frame]))
        y_bef = self.linear_bef(F.relu(y_bef[-1]))
        y_aft = self.linear_aft(F.relu(y_aft[-1]))

        #y = torch.cat((y_bef[:, None, :]+x_bef.mean(axis=1)[:, None, :], y_cur[:, None, :], y_aft[:, None, :]+x_aft.mean(axis=1)[:, None, :]), dim=1)
        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)

        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)  # N x 2048

        if not is_train:
            return out, scores
        else:
            y = torch.cat((y[:, 0:1], y[:, 2:], out[:, None, :]), dim=1)
            return y, scores
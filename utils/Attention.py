import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, init_size, hidden_size, output_size):

        super(Attention, self).__init__()

        self.init_size = init_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.line_q = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_k = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_v = nn.Linear(self.init_size, self.output_size, bias=False)

    def forward(self, query, key, value, mask=None, dropout=None):

        query = self.line_q(query)
        key = self.line_k(key)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = F.dropout(p_attn,p=dropout,training=self.training)

        return self.line_v(torch.matmul(p_attn, value))
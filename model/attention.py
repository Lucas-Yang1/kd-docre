import torch.nn as nn
import torch


class AxisAttention(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super(AxisAttention, self).__init__()

        self.Wq = nn.Linear(input_dim, attn_dim)
        self.Wk = nn.Linear(input_dim, attn_dim)
        self.Wv = nn.Linear(input_dim, input_dim)

        self.reset()

    def reset(self):
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

        nn.init.zeros_(self.Wq.bias)
        nn.init.zeros_(self.Wk.bias)
        nn.init.zeros_(self.Wv.bias)

    def forward(self, g):
        q, k, v = self.Wq(g), self.Wk(g), self.Wv(g)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-2)

        row_att = torch.mul(torch.mul(q, k.transpose(0, 2)).sum(-1, True).softmax(-1), v.transpose(0, 2)).sum(0)
        col_att = torch.mul(torch.mul(q, k.transpose(1, 2)).sum(-1, True).softmax(-1), v.transpose(1, 2)).sum(1)

        return g + row_att + col_att


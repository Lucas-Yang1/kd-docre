import torch
import torch.nn as nn

class Bilinear(nn.Module):
    def __init__(
            self,
            input_1_size,
            input_2_size,
            output_size,
            bias=True,
                 ):
        super(Bilinear, self).__init__()
        self.weight = nn.Parameter(torch.ones(output_size, input_1_size, input_2_size))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, 1, 1))


    def forward(self, input_1:torch.Tensor, input_2:torch.Tensor):

        intermediate = torch.matmul(input_1.unsqueeze(-3), self.weight)
        final = torch.matmul(intermediate, input_2.unsqueeze(-3).transpose(-1, -2))

        if hasattr(self, 'bias'):
            final += self.bias

        return final


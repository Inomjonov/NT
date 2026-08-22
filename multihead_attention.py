import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, row_dim = 0, col_dim = 1, num_head = 1):
        super().__init__()

        self.heads = nn.ModuleList([Attention(d_model, row_dim, col_dim) for _ in range(num_head)])

        self.row_dim = row_dim
        self.col_dim = col_dim


    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v):
        return torch.cat([head(encodings_for_q, encodings_for_k, encodings_for_v) for head in self.heads], dim = self.col_dim)



torch.manual_seed(42)
multihead_Attention = MultiHeadAttention(2, 0, 1, 3)

encodings_for_q = torch.tensor([[1.16, 0.23], 
                                [0.57, 1.36], 
                                [4.41, -2.16]])


encodings_for_k = torch.tensor([[1.16, 0.23], 
                                [0.57, 1.36], 
                                [4.41, -2.16]])


encodings_for_v = torch.tensor([[1.16, 0.23], 
                                [0.57, 1.36], 
                                [4.41, -2.16]])


print(multihead_Attention(encodings_for_q, encodings_for_k, encodings_for_v))
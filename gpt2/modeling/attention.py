# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import math
import torch
from gpt2.modeling import Linear


class BaseAttention(torch.nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        x += mask.type_as(x) * x.new_tensor(-1e4)
        x = self.dropout(x.softmax(-1))
        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    def __init__(self, n_heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.n_heads = n_heads
    
    def forward(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        
        # split the tensors to multi-heads
        q = q.view(q.size()[:-1] + (self.n_heads, q.size(-1) // self.n_heads))
        k = k.view(k.size()[:-1] + (self.n_heads, k.size(-1) // self.n_heads))
        v = v.view(v.size()[:-1] + (self.n_heads, v.size(-1) // self.n_heads))
        
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        
        mask = mask.unsqueeze(-3)
        
        # calculate multi-headed attentions and merge them into one
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.n_heads)))


class AttentionLayer(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, dropout)
        assert dim % n_heads == 0
        self.proj_q = Linear(dim, dim)
        self.proj_k = Linear(dim, dim)
        self.proj_v = Linear(dim, dim)
        self.proj_o = Linear(dim, dim)
    
    def reset_parameters(self):
        self.proj_q.reset_parameters()
        self.proj_k.reset_parameters()
        self.proj_v.reset_parameters()
        self.proj_o.reset_parameters()
    
    def forward(self,
              x: torch.Tensor,
              k_cache: torch.Tensor,
              v_cache: torch.Tensor,
              mask: torch.Tensor):
        
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        
        # reuse attention keys and values by concatenating to the current ones 
        k = torch.cat((k_cache, k), dim=-2)
        v = torch.cat((v_cache, v), dim=-2)
        
        scores = self.attention(q, k, v, mask)
        
        x = self.proj_o(scores)
        return x, k, v


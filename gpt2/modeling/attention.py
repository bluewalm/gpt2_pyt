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
from bluewalm.softplus_attention import QueryProjection, KeyProjection, ValueProjection, OutProjection, attention_operator


class AttentionLayer(torch.nn.Module):
    def __init__(self, dim: int, core_dim: int, dropout: float):
        super().__init__()
        # assert dim % n_heads == 0
        self.proj_q = QueryProjection(dim, core_dim)
        self.proj_k = KeyProjection(dim, core_dim)
        self.proj_v = ValueProjection(dim, core_dim)
        self.proj_o = OutProjection(dim, core_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.proj_q.reset_parameters()
        self.proj_k.reset_parameters()
        self.proj_v.reset_parameters()
        self.proj_o.reset_parameters()
    
    def forward(self,
              x: torch.Tensor,
              k_cache: torch.Tensor,
              v_cache: torch.Tensor):
        
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        
        # reuse attention keys and values by concatenating to the current ones 
        k = torch.cat((k_cache, k), dim=2)
        v = torch.cat((v_cache, v), dim=2)
        
        # q, k and v are contiguous here 
        scores = attention_operator(q, k, v)
        scores = self.dropout(scores)
        
        x = self.proj_o(scores)
        return x, k, v


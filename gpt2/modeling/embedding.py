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
from typing import Dict


class Embedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight.numel() == 0:
            self.weight.requires_grad = False  # WAR : apex bug
    
    def __repr__(self):
        return "Embedding(" + self.extra_repr() + ")"
    
    def extra_repr(self):
        s = 'num_embeddings={num_embeddings}, embedding_dim={embedding_dim}'
        precision = {torch.float32 : 'fp32', torch.float16 : 'fp16', torch.bfloat16 : 'bf16'}
        precision = precision[self.weight.dtype]
        s += ', precision={precision}'
        size = self.weight.nelement() * self.weight.element_size()
        size = round(size / 1024**2, 4)
        s += ', size=' + str(size) + ' MB'
        return s.format(**self.__dict__, precision=precision)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()
        self.embedding = Embedding(max_seq_len, dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.02)
    
    def _load_from_state_dict(self,
                          state_dict: Dict[str, torch.Tensor],
                          prefix: str,
                          *args,
                          **kwargs):
        weight = state_dict[f'{prefix}embedding.weight']
        # Reduce or expand the positional embedding matrix to increase or decrease the total sequence length. 
        if weight.size(0) < self.embedding.num_embeddings:
            weight = torch.cat((weight, self.embedding.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.embedding.num_embeddings:
            weight = weight[:self.embedding.num_embeddings]
        
        state_dict[f'{prefix}embedding.weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1), dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x).contiguous()
        return self.embedding.forward(position)


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.02)
    
    def forward(self, 
              x: torch.Tensor, 
              transposed: bool = False) -> torch.Tensor:
        if transposed:
            wT = self.embedding.weight
            wT = wT.transpose(0, 1)
            return torch.matmul(x, wT)
        else:
            ret = self.embedding.forward(x)
            return ret


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

import torch
import torch.utils.checkpoint
from typing import Optional, Tuple, List, Union
# 
from gpt2.modeling import FutureMasking
from gpt2.modeling import AttentionLayer
from gpt2.modeling import PositionalEmbedding
from gpt2.modeling import TokenEmbedding
from gpt2.modeling import ModelArgs
from gpt2.modeling import Linear
from gpt2.modeling import PositionwiseFeedForward


class TransformerLayer(torch.nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dim)
    past (*)        float           (..., past_len, dim)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dim)
    output 2 (*)    float           (..., past_len + seq_len, dim)
    ===========================================================================
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = AttentionLayer(args.dim, args.n_heads, args.dropout)
        self.combinator = PositionwiseFeedForward(args.dim, args.rate, args.dropout)
        self.ln_attention = torch.nn.LayerNorm(args.dim)
        self.ln_combinator = torch.nn.LayerNorm(args.dim)
    
    def reset_parameters(self):
        self.attention.reset_parameters()
        self.combinator.reset_parameters()
        self.ln_attention.reset_parameters()
        self.ln_combinator.reset_parameters()
    
    def forward(self,
              v: torch.Tensor,
              k_cache: torch.Tensor,
              v_cache: torch.Tensor,
              mask: torch.Tensor):
        # Layer normalizations are performed before the layers respectively.
        v = self.ln_attention(v)
        a, k_cache, v_cache = self.attention(v, k_cache, v_cache, mask)
        v = v + a
        y = self.ln_combinator(v)
        v = v + self.combinator(y)
        return v, k_cache, v_cache


class Transformer(torch.nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    k_past (**)       float           (layers, ..., past_len, dim)
    v_past (**)       float           (layers, ..., past_len, dim)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dim)
    output 2 (**)   float           (layers, ..., past_len + seq_len, dim)
    output 3 (**)   float           (layers, ..., past_len + seq_len, dim)
    ===========================================================================
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.future_masking = FutureMasking()
        
        self.positional_embedding = PositionalEmbedding(args.max_seq_len, args.dim)
        self.token_embedding = TokenEmbedding(args.vocab_size, args.dim)
        self.dropout_embedding = torch.nn.Dropout(args.dropout)
        
        layers = [TransformerLayer(args) for _ in range(args.n_layers)]
        self.layers = torch.nn.ModuleList(layers)
        self.ln_head = torch.nn.LayerNorm(args.dim)
    
    def reset_parameters(self):
        self.positional_embedding.reset_parameters()
        self.token_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.ln_head.reset_parameters()
    
    def get_empty_k_cache(self, batch_size, cache_len):
        shape = (len(self.layers), batch_size, cache_len, self.dim)
        device=self.ln_head.weight.device
        dtype=self.ln_head.weight.dtype
        return torch.empty(shape, device=device, dtype=dtype)
    
    def get_empty_v_cache(self, batch_size, cache_len):
        shape = (len(self.layers), batch_size, cache_len, self.dim)
        device=self.ln_head.weight.device
        dtype=self.ln_head.weight.dtype
        return torch.empty(shape, device=device, dtype=dtype)
    
    def forward(self, input: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        # get cache length 
        cache_len = k_cache.size(-2)
        
        # get query length
        seq_len = input.size(1)

        # create masking tensor 
        mask = self.future_masking(seq_len, cache_len, input.device)
        
        # use token embedding and positional embedding layers 
        positions = self.positional_embedding(input, cache_len)
        v = self.token_embedding(input) + positions
        v = self.dropout_embedding(v)
        
        # prepare empty caches 
        updated_k_cache = []
        updated_v_cache = []
        # apply transformer layers sequentially 
        for layer, k_c, v_c in zip(self.layers, k_cache, v_cache):
            v, k_c, v_c = layer(v, k_c, v_c, mask)
            updated_k_cache.append(k_c)
            updated_v_cache.append(v_c)
        # tensorify 
        updated_k_cache = torch.stack(updated_k_cache)
        updated_v_cache = torch.stack(updated_v_cache)
        # layer normalization 
        v = self.ln_head(v)
        # transposed embedding 
        v = self.token_embedding(v, transposed=True)
        # the only return point 
        return v, updated_k_cache, updated_v_cache


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


class Linear(torch.nn.Linear):
    def __repr__(self):
        return "Linear(" + self.extra_repr() + ")"
    
    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}'
        precision = {torch.float32 : 'fp32', torch.float16 : 'fp16', torch.bfloat16 : 'bf16'}
        precision = precision[self.weight.dtype]
        s += ', precision={precision}'
        size = self.weight.nelement() * self.weight.element_size()
        size = round(size / 1024**2, 4)
        s += ', size=' + str(size) + ' MB'
        return s.format(**self.__dict__, precision=precision)


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class PositionwiseFeedForward(torch.nn.Sequential):
    def __init__(self, dim: int, rate: int, dropout: float):
        super().__init__(
            Linear(dim, dim * rate),
            Swish(),
            torch.nn.Dropout(dropout),
            Linear(dim * rate, dim))
    
    def reset_parameters(self):
        self[0].reset_parameters()
        self[3].reset_parameters()


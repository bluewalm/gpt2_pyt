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


class FutureMasking(torch.nn.Module):
    def forward(self, seq_len: int, offset: int, device: torch.device) -> torch.Tensor:
        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset), dtype=torch.int32, device=device)
        future = future.triu(offset + 1)
        mask = future.unsqueeze(0)
        return mask


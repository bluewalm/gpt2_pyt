#!/bin/bash

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

for N_LAYERS in 4
do
for BATCH_SIZE in 32
do
for DIM in 128
do
for N_HEADS in 4
do
for STEPS in 160000
do
for SEQLEN in 2048
do
python -m gpt2 generate --tokenizer            tokenizer.model \
                      --n_layers               ${N_LAYERS} \
                      --dim                    ${DIM} \
                      --n_heads                ${N_HEADS} \
                      --from_checkpoint        gpt2-softmax-layers${N_LAYERS}-dim${DIM}-heads${N_HEADS}-seqlen${SEQLEN}-batchsize${BATCH_SIZE}-steps${STEPS}.pth \
                      --max_seq_len            ${SEQLEN} \
                      --nucleus_prob           0.85 \
                      --allow_tf32 \
                      --amp_bf16
done
done
done
done
done
done


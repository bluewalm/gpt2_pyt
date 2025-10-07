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

# splits *.txt files present in current directory into train and test set

# set directories
ROOT_DIR="$(pwd)/data/en"
TRAIN_DIR="$(pwd)/pretrain_data/train"
EVAL_DIR="$(pwd)/pretrain_data/eval"

# create train and test dirs if they don't exist
mkdir -p "$ROOT_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$EVAL_DIR"

# count .txt files
NBFiles=$(find "$ROOT_DIR" -type f -name "*.txt" | wc -l)

echo "Found ${NBFiles} .txt files. "

# move 80% to train dir and the rest to test dir
index=0
threshold=$(( NBFiles * 9 / 10 ))
for file in "$ROOT_DIR"/*; do
    if [ -f "$file" ] && [ ${file: -4} = ".txt" ]; then
        if (( index < threshold )); then
            mv "$file" "$TRAIN_DIR"
            ((index++))
        else
            mv "$file" "$EVAL_DIR"
        fi
    fi
done

echo "Dataset split into train and eval subfolders. "
NBFiles=$(find "$TRAIN_DIR" -type f -name "*.txt" | wc -l)
echo "Number of files in train subfolder: ${NBFiles} "
NBFiles=$(find "$EVAL_DIR" -type f -name "*.txt" | wc -l)
echo "Number of files in eval subfolder: ${NBFiles} "


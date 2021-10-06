#!/usr/bin/python3

# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

# STEP 1 - Import libraries and install dependencies ---------------------------------------
import torch
import os
from pip._internal import main as pipmain

from os import environ

# Install `transformers` from master
# pipmain(['install', 'git+https://github.com/huggingface/transformers'])
# pipmain(['install', 'datasets'])

# STEP 2 - Check GPU connection ------------------------------------------------------------
output=os.system("nvidia-smi")
print(output)

# Check if you can access the GPU (should be 'True')
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

#initialize your GPU
torch.cuda.init()

# STEP 3 - Train a language model from roberta_base ----------------------------------------
output=os.system("python myrun_mlm.py "
                 "--model_name_or_path=roberta-base "
                 "--overwrite_output_dir "
                 "--train_file='data/training_wikibookabstract.txt' "
                 "--validation_file='data/testing_wikibookabstract.txt' "
                 "--per_device_train_batch_size=16 "
                 "--per_device_eval_batch_size=16 "
                 "--gradient_accumulation_steps=16 "
                 "--do_train "
                 "--do_eval "
                 "--line_by_line "
                 "--save_steps=2034 "
                 "--num_train_epochs=100 "
                 "--output_dir='./spaceROBERTA_eval/' "
                 "--logging_steps=226 ")









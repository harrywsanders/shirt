#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:27:26 2024

@author: eochoa
"""

import json
from tqdm import tqdm
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def query_encoder(queries):
    torch.cuda.empty_cache()

    input_data = tokenizer.batch_encode_plus(queries,
                                           return_tensors="pt",
                                           padding=True)

    input_ids = input_data['input_ids']
    mask = input_data['attention_mask']
    if torch.cuda.is_available():
        input_ids = input_ids.cuda(device=cuda_devices[0])
        mask = mask.cuda(device=cuda_devices[0])
    encoder_outputs = encoder(input_ids=input_ids,
                                          attention_mask=mask,
                                          return_dict=True)
    hidden_states = encoder_outputs["last_hidden_state"]
    pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
    pooled_hidden_states_np = pooled_hidden_states.detach().cpu().numpy()
  
    return pooled_hidden_states_np


file_path='../data_QA.json'

with open(file_path, "r") as datafile:
    data = json.load(datafile)
    
data_new = {bench:{'Qs':[x.split('\nQ: ')[-1] for x in data[bench]['Qs']], 
        'As':[x for x in data[bench]['As']]} for bench in data.keys()}


model = AutoModelForSeq2SeqLM.from_pretrained('google/t5-xl-lm-adapt')
cuda_devices = [0]
print(f"Using CUDA devices {cuda_devices}")
if torch.cuda.is_available():
    model.cuda(device=cuda_devices[0])
model.eval()

encoder = torch.nn.DataParallel(model.encoder, device_ids=cuda_devices)
tokenizer = AutoTokenizer.from_pretrained('google/t5-xl-lm-adapt')



with torch.inference_mode():
  for k in tqdm(data.keys()):

    queries = data[k]['Qs']
    #pooled_hidden_states_np_list = []
    pooled_hidden_states_np = query_encoder(queries)
    data_new[k]['Embs'] = pooled_hidden_states_np.tolist()
    

with open('data_QA_processed.json', 'w') as f:
    json.dump(data_new, f)
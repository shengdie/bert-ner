import os
import copy
import argparse
import time
import torch
import numpy as np
from model import model
from torch.utils.data import DataLoader
from dataloader import load_data, load_vocab, Config, pad_batch, NERDataSet

config = Config('./config.json')
config.batch_size = 45
#config.inc_dig = True
train_dataset = NERDataSet(os.path.join(config.data_path, 'train.tsv'), os.path.join(config.data_path, 'train_rel.npz'), config )
test_dataset = NERDataSet(os.path.join(config.data_path, 'test.tsv'), os.path.join(config.data_path, 'test_rel.npz'), config )
padfn = lambda x: pad_batch(x, max_len=80)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=padfn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=padfn)

#lr1 = 0.001
lr2 = 0.0001
record = {}
for m in ['cls', 'ave', 'pair_ave', 'pair_transformer']:
    config.method = m
    net = model(config, ner_state_dict_path=None)
    torch.cuda.empty_cache()
    # train
    #net.train(train_dataloader, test_dataloader, 1, config.idx2tag, 
    #          save_weight_path=None, start_save=1, lr=lr1)
    net.finetune_bert(True)
    net.train(train_dataloader, test_dataloader, 5, config.idx2tag, 
              save_weight_path=None, start_save=1, lr=lr2)
    record[m] = copy.deepcopy(net.record)
print(record)
with open('./records_{}.txt'.format(lr2), 'w') as f:
    f.write(str(record))
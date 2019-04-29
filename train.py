import os
import argparse
import time
import numpy as np
from model import model
from torch.utils.data import DataLoader
from dataloader import load_data, load_vocab, Config, pad_batch, NERDataSet

if __name__ == "__main__":
    config = Config('./config.json')
    #config.inc_dig = True
    print(config.inc_dig)
    train_dataset = NERDataSet(os.path.join(config.data_path, 'train.tsv'), os.path.join(config.data_path, 'train_rel.npz'), config )
    test_dataset = NERDataSet(os.path.join(config.data_path, 'test.tsv'), os.path.join(config.data_path, 'train_rel.npz'), config )
    padfn = lambda x: pad_batch(x, max_len=75)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=padfn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=padfn)
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-ep', dest='epochs', default=None, type=int,
                         help='epochs')
    parser.add_argument('--batch-size', '-bs', dest='batch_size', default=None, type=int,
                         help='batch size')
    parser.add_argument('--lr', dest='lr', default=None, type=float,
                         help='lr')
    parser.add_argument('--warmup', dest='lr_warmup', default=None, type=float,
                         help='lr warmup, to potion of total steps')
    parser.add_argument('--save', '-s', dest='save', action='store_true',
                         help='save the best weight')
    parser.add_argument('--start_save', '-ss', dest='start_save', default=2, type=int,
                         help='start save model after these epochs')
    parser.add_argument('--ner-weight', '-nw', dest='ner_weight', default=None,
                         help='The ner model weight, which contains bert weight and classifier weight')                      

    pargs = parser.parse_args()
    if pargs.epochs is not None:
        config.num_epochs = pargs.epochs
    if pargs.batch_size is not None:
        config.batch_size = pargs.batch_size
    if pargs.lr is not None:
        config.lr = pargs.lr
    if pargs.lr_warmup is not None:
        config.lr_warmup = pargs.lr_warmup
    if pargs.ner_weight is not None:
        config.ner_weight = pargs.ner_weight
        config.bert_weight_path = 'None'
    if pargs.save:
        if not os.path.exists('model_save'): os.makedirs('model_save')
        save_path = os.path.join('model_save', time.strftime("%m-%d-%H-%M-%S", time.localtime()))
        os.makedirs(save_path)
        config.save(os.path.join(save_path, 'model_conf.json'))
        weight_path = os.path.join(save_path, 'ner_weight')
        hist_path = os.path.join(save_path, 'history.txt')
    else:
        weight_path = None
        hist_path = None
    

    # new model
    net = model(config, ner_state_dict_path=pargs.ner_weight)
    
    # train
    net.train(train_dataloader, test_dataloader, config.num_epochs_cls, config.idx2tag, save_weight_path=weight_path, start_save=pargs.start_save)
    net.finetune_bert(True)
    net.train(train_dataloader, test_dataloader, config.num_epochs_full, 
          config.idx2tag, save_weight_path=weight_path, start_save=1, lr=0.0001)
    net.save_hist(hist_path)
    
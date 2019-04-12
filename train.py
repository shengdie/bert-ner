import argparse
import numpy as np
from model import model
from dataloader import load_data

if __name__ == "__main__":
    vocab_path = 'weights/pubmed_pmc_470k/vocab.txt'
    bert_conf_path = 'weights/pubmed_pmc_470k/bert_config.json'
    bert_weight= 'weights/pytorch_weight'
    train_data_path = 'data/data_BI.npy'
    test_data_path = 'data/test_data_BI.npy'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-size', '-ts', dest='train_size', type=int, default=-1,
                        help='train data size')
    parser.add_argument('--val-size', '-vs', dest='val_size', default='0.1', #type=int,
                        help='val data size')
    parser.add_argument('--max-len', '-ml', dest='max_len', default=120, type=int,
                        help='max len of the sequence')
    parser.add_argument('--epochs', '-ep', dest='epochs', default=5, type=int,
                        help='epochs')
    parser.add_argument('--full-train', '-ft', dest='full_trainning', action='store_true',
                        help='if fully train the whole model')                             

    pargs = parser.parse_args()
    train_size = None if pargs.train_size < 0 else pargs.train_size
    val_size = eval(pargs.val_size)

    classes = [0, 1, 2, 3, 4, 5, 6]
    tags = ['O', 'B-E', 'B-P', 'B-T', 'I-E', 'I-P', 'I-T']
    #classes = [tags['']]

    idx2token = np.loadtxt(vocab_path, dtype='str')
    vocab = {idx2token[i]:i for i in range(len(idx2token))}

    train_dataloader, val_dataloader = load_data(train_data_path, test_data_path, vocab, classes=classes, max_len=pargs.max_len,
                                                 train_size=train_size, val_size=val_size)
    # new model
    net = model(bert_conf_path, bert_weight, num_class=len(classes))
    if pargs.full_trainning:
        net.full_trainning(True)
    
    # train
    net.train(train_dataloader, val_dataloader, pargs.epochs, tags)
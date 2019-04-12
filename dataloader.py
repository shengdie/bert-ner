import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split


def process_data(raw_txts, raw_labels, vocab, max_len=120, pad='[PAD]', unk='[UNK]', cls='[CLS]', add_cls=False, classes=[0,1,2,3]):
    id_pad = vocab[pad]
    id_cls = vocab[cls]
    id_unk = vocab[unk]
    padded_txt_idx = torch.tensor([([id_cls] if add_cls else []) + [vocab[w] if w in vocab.keys() else id_unk for w in s] + [id_pad] * (max_len - len(s)) \
                                 if len(s) < max_len else \
                                 ([id_cls] if add_cls else [])  + [vocab[w] if w in vocab.keys() else id_unk for w in s[:max_len]]
                                 for s in raw_txts]).long()
    padded_labels = torch.tensor([([classes[0]] if add_cls else []) + list(l) + [classes[0]] * (max_len - len(l)) \
                                  if len(l) < max_len else \
                                  ([classes[0]] if add_cls else []) + list(l[:max_len])
                                  for l in raw_labels]).long()
    # add attention mask to ignore padding
    att_masks = (padded_txt_idx != id_pad).long()
    return padded_txt_idx, padded_labels, att_masks

def load_data(train_data_path, test_data_path, vocab, 
              batch_size=32, val_size = 0.1, train_size=None, classes=[0,1,2,3],
              max_len=120, pad='[PAD]', unk='[UNK]', cls='[CLS]', add_cls=False, resplit=[0.75, 0.17,0.08]):
    """load data
    
    Arguments:
        train_data_path {str} -- train data path, numpy npy file
        test_data_path {str} -- test data 
        vocab {dict} -- token to id
    
    Keyword Arguments:
        batch_size {int} -- batch size (default: {32})
        val_size {float} -- split part of test data to validation, can be int or ratio. not used when resplit is not None (default: {0.1})
        train_size {int} -- only use this size of trainning data, for quick test, not used when resplit is not None (default: {None})
        classes {list} -- classes id (default: {[0,1,2,3]})
        max_len {int} -- max len of sequence, (sentence) (default: {120})
        pad {str} -- pad token (default: {'[PAD]'})
        unk {str} -- unknown token (default: {'[UNK]'})
        cls {str} -- [cls] token (default: {'[CLS]'})
        add_cls {bool} -- if add [CLS] in the beginning (default: {False})
        resplit {list} -- if not None, then combine train and test, then randomly resplit to [train, test, validation]. (default: {[0.75, 0.17,0.08]})
    
    Returns:
        [type] -- [description]
    """

    data = np.load(train_data_path).item()
    data_test = np.load(test_data_path).item()
    train_txts = []
    train_labels = []
    temp_txts = []
    temp_labels = []
    for v in data.values():
        for r in v.values():
            #if 'txt' in r.keys():
            train_txts.extend(r['txt'])
            train_labels.extend(r['output'])
    for v in data_test.values():
        temp_txts.extend(r['txt'])
        temp_labels.extend(r['output'])
    if resplit is not None:
        temp_txts.extend(train_txts)
        temp_labels.extend(train_labels)
        train_size = int(len(temp_txts) * resplit[0])
        test_size = int(len(temp_txts) * resplit[1])
        val_size = len(temp_txts) - train_size - test_size
        temp_input, temp_output, temp_att_masks = process_data(temp_txts, temp_labels, vocab, max_len=max_len,
                                                            pad=pad, unk=unk, cls=cls, add_cls=add_cls,classes=classes)
        temp_dataset = TensorDataset(temp_input, temp_output, temp_att_masks)                                        
        train_dataset, test_dataset, val_dataset = random_split(temp_dataset, [train_size, test_size, val_size])
    else:
        train_input, train_output, train_att_masks =  process_data(train_txts, train_labels, vocab, max_len=max_len,
                                                                    pad=pad, unk=unk, cls=cls, add_cls=add_cls, classes=classes)
        temp_input, temp_output, temp_att_masks = process_data(temp_txts, temp_labels, vocab, max_len=max_len,
                                                                pad=pad, unk=unk, cls=cls, add_cls=add_cls,classes=classes)
        if train_size is None:
            train_size = len(train_input)
        
        train_dataset = TensorDataset(train_input[:train_size], train_output[:train_size], train_att_masks[:train_size])
        temp_dataset = TensorDataset(temp_input, temp_output, temp_att_masks)
        if type(val_size) is float:
            val_size = int(val_size * temp_input.size()[0])
            test_size = temp_input.size()[0] - val_size
        else:
            val_size = 32
            test_size = temp_input.size()[0] - val_size

        test_dataset, val_dataset = random_split(temp_dataset, [test_size, val_size])
    print('data size [train, test, val] = [{}, {}, {}]'.format(len(train_dataset), len(test_dataset), len(val_dataset)))

    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    val_sampler = SequentialSampler(val_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader, val_dataloader
import json
import copy
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split, Dataset
from pytorch_pretrained_bert import BertTokenizer, BertConfig

def load_tsv(path, cls='[CLS]', sep='[SEP]', add_cls=False, add_sep=False, cls_num=1):
    """load tsv data"""
    with open(path, 'r') as f:
        sent_tags = f.read().strip().split('\n\n')
    sents, tags = [], []
    for s in sent_tags:
        st = [w.split() for w in s.splitlines()]
        sents.append(([cls] * cls_num if add_cls else []) + [w[0] for w in st if len(w) == 2] + ([sep] if add_sep else []))
        tags.append((['O'] * cls_num if add_cls else []) + [w[1] for w in st if len(w) == 2] + (['O'] if add_sep else []))
    return sents, tags

def extract_num_concept(xx, yy, rr, num, remove_non=0, remove_one=0):
    ne = 5 - num
    nm = ne * 5 + ne * num
    #nall = num ** 2
    new_x, new_y, new_r = [], [], []
    for x , y, r in zip(xx, yy, rr):
        if remove_non > 0 and r.sum() == 0:
            if np.random.random() > remove_non:
                new_x.append(x)
                new_y.append(y)
                new_r.append(r[:num, :num])
        elif remove_one > 0 and r.sum() == 1:
            if np.random.random() > remove_one:
                new_x.append(x)
                new_y.append(y)
                new_r.append(r[:num, :num])
        elif (r[num:,:] == 0).sum() + (r[:num,num:] == 0).sum() == nm:
            new_x.append(x)
            new_y.append(y)
            new_r.append(r[:num, :num])
    return new_x, new_y, np.array(new_r)

def load_relation(path):
    return np.load(path)['arr_0'] +1

def triu_relation(d, inc_dig=True, inc_first=False):
    a = d[0].shape[0]
    t1, t2 = np.triu_indices(a, 0) if inc_dig else np.triu_indices(a, 1)
    if not inc_dig and inc_first:
        t1 = np.insert(t1, 0, 0)
        t2 = np.insert(t2, 0, 0) 
    return d[:, t1, t2]

def load_vocab(path):
    idx2token = np.loadtxt(path, dtype='str', comments=None)
    vocab =  {idx2token[i]:i for i in range(len(idx2token))}
    return idx2token, vocab

def pad_batch(batch, max_len=None, pad=0, idO=0):
    """pad a batch to max len of seq if max_len is None"""
    sents = [s[0].copy() for s in batch]
    sents_len = [len(s) for s in sents]
    labels = [s[1].copy() for s in batch]
    relations = [s[-1] for s in batch]
    #atts = [s[-1] for s in batch]
    atts = [None] * len(sents)
    max_slen = max(sents_len)
    if max_len is None: 
        max_len = max_slen
    if max_len >= max_slen:
        for i in range(len(sents)):
            atts[i] = [1] * sents_len[i] + [0] * (max_slen - sents_len[i])
            sents[i].extend([pad] * (max_slen - sents_len[i]))
            labels[i].extend([idO] * (max_slen - sents_len[i]))
    else:
        for i in range(len(sents)):
            if max_len >= sents_len[i]:
                atts[i] = [1] * sents_len[i] + [0] * (max_len - sents_len[i])
                sents[i].extend([pad] * (max_len - sents_len[i]))
                labels[i].extend([idO] * (max_len - sents_len[i]))
            else:
                atts[i] = [1] * max_len
                sents[i] = sents[i][:max_len]
                labels[i] = labels[i][:max_len]
    return torch.LongTensor(sents), torch.LongTensor(labels), torch.LongTensor(atts), torch.LongTensor(relations)

class Config(object):
    def __init__(self, config_json):
        """json should contain data_path, vocab_path, tags_vocab, batch_size, num_epochs, lr, bert_weight_path, bert_conf_path"""
        if isinstance(config_json, str):
            with open(config_json, 'r', encoding='utf-8') as f:
                json_config = json.loads(f.read())
            assert all(v in json_config.keys() for v in ['data_path', 'vocab_path', 'tags_vocab', 
                                                        'batch_size', 'num_epochs_cls', 'lr', 'bert_weight_path', 'bert_conf_path', 'lr_warmup'])
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            raise ValueError('Must be str path of config')
        #
        self.idx2tag = ['O', 'I-<P>'] + [t for t in self.tags_vocab]
        self.tag2idx = {v:i for i, v in enumerate(self.idx2tag)}
        self.piece_tag = 'I-<P>'
  
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.to_json_string())

class NERDataSet(Dataset):
    def __init__(self, data_path, relation_path, config, add_cls=True, add_sep=False):
        self.config = config
        self.relations = load_relation(relation_path).astype(int)#.tolist()
        #print(self.relations)
        #add_cls = True
        self.sents, self.tags = load_tsv(data_path, add_cls=add_cls, add_sep=add_sep, cls_num=config.cls_num)
        if config.max_concept <5 or config.remove_non > 0 or config.remove_one > 0:
            self.sents, self.tags, self.relations = extract_num_concept(self.sents, self.tags, self.relations, config.max_concept, remove_non=config.remove_non, remove_one=config.remove_one)
        # relation
        #print(self.relations)
        if config.triu:
            self.relations = triu_relation(self.relations, config.inc_dig, config.inc_first)
        
        self.tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)
        self.tokenize()

    def __len__(self):
        return len(self.sents)

    def tokenize(self):
        alltok_sents, alltok_tags = [], []
        for sent_words, sent_tags in zip(self.sents, self.tags):
            tok_sent, tok_tag = [], []
            for w, t in zip(sent_words, sent_tags): # tokenize the words
                tokens = self.tokenizer.tokenize(w)
                tok_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                tok_tags = [t] + [self.config.piece_tag] * (len(tokens) - 1)
                ttags_ids = [self.config.tag2idx[tt] for tt in tok_tags]
                tok_sent.extend(tok_ids)
                tok_tag.extend(ttags_ids)
            alltok_sents.append(tok_sent)
            alltok_tags.append(tok_tag)
        self.tok_sents = alltok_sents
        self.tok_tags = alltok_tags
    
    def __getitem__(self, idx):
        return self.tok_sents[idx], self.tok_tags[idx], self.relations[idx]

class RelationDataSet(Dataset):
    def __init__(self, data_path, config, add_cls=True, add_sep=True):
        self.config = config
        #self.relations = load_relation(data_path).astype(int)#.tolist()
        #print(self.relations)
        #add_cls = True
        self.concepts, self.relations = load_pure_relations(data_path, add_cls=add_cls, add_sep=add_sep, remove_no=config.remove_no)
        
        self.tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)
        self.tokenize()

    def __len__(self):
        return len(self.relations)

    def tokenize(self):
        cons, cons_types = [], []
        for c1, c2 in self.concepts:
            c1token, c2token = [], []
            for w1 in c1:
                w1token = self.tokenizer.tokenize(w1)
                w1token_idx = self.tokenizer.convert_tokens_to_ids(w1token)
                c1token.extend(w1token_idx)
            for w2 in c2:
                w2token = self.tokenizer.tokenize(w2)
                w2token_idx = self.tokenizer.convert_tokens_to_ids(w2token)
                c2token.extend(w2token_idx)
            t = [0] * len(c1token) + [1] * len(c2token)
            c1token.extend(c2token)
            cons.append(c1token)
            cons_types.append(t)
            
        self.concepts = cons
        self.cons_types = cons_types
    
    def __getitem__(self, idx):
        return self.concepts[idx], self.cons_types[idx], self.relations[idx]

# def pad_batch_rels(batch, max_len=None, pad=0, idO=0):
#     """pad a batch to max len of seq if max_len is None"""
#     cons = [s[0].copy() for s in batch]
#     cons_len = [len(s) for s in cons]
#     cons_t = [s[1].copy() for s in batch]
#     relations = [s[-1] for s in batch]
#     #atts = [s[-1] for s in batch]
#     atts = [None] * len(cons)
#     max_slen = max(cons_len)
#     if max_len is None: max_len = max_slen
#     if max_len >= max_slen:
#         for i in range(len(cons)):
#             atts[i] = [1] * cons_len[i] + [0] * (max_len - cons_len[i])
#             cons[i].extend([pad] * (max_len - cons_len[i]))
#             cons_t[i].extend([idO] * (max_len - cons_len[i]))
#     else:
#         for i in range(len(cons)):
#             if max_len >= cons_len[i]:
#                 atts[i] = [1] * cons_len[i] + [0] * (max_len - cons_len[i])
#                 cons[i].extend([pad] * (max_len - cons_len[i]))
#                 cons_t[i].extend([idO] * (max_len - cons_len[i]))
#             else:
#                 atts[i] = [1] * max_len
#                 cons[i] = cons[i][:max_len]
#                 cons_t[i] = labels[i][:max_len]
#     return torch.LongTensor(sents), torch.LongTensor(labels), torch.LongTensor(atts), torch.LongTensor(relations)

def load_pure_relations(file, cls='[CLS]', sep='[SEP]', add_cls=False, add_sep=False, remove_no=0):
    with open(file, 'r') as f:
        l = f.read()
    instances = l.split('\n\n')
    concept_pair, rels = [], []
    for ins in instances:
        c1,c2, r = ins.split('\n')
        r = int(r)
        c1 = c1.split()
        c2 = c2.split()
        #t = [0] * len(c1) + [1] * len(c2)
        if add_cls: c1.insert(0, cls)
        if add_sep: c1.append(sep)
        if remove_no > 0 and r == 1:
            if np.random.random() > remove_no:
                concept_pair.append([c1, c2])
                #type_idx.append(t)
                rels.append(r)
        else:
            concept_pair.append([c1, c2])
                #type_idx.append(t)
            rels.append(r)
    return concept_pair, rels

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

def get_numpy_dump(data):
    if type(data) is np.lib.npyio.NpzFile:
        d = data[list(data)[0]]
        if type(d) is np.ndarray: return d
        return d.item()
    else:
        return data.item()

def load_data(train_data_path, test_data_path, vocab, 
              batch_size=32, val_size = 0.1, train_size=None, classes=[0,1,2,3],
              max_len=120, pad='[PAD]', unk='[UNK]', cls='[CLS]', add_cls=False, resplit=[0.75, 0.17,0.08], labelkey='output'):
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

    data = get_numpy_dump(np.load(train_data_path)) #.item()
    data_test = get_numpy_dump(np.load(test_data_path)) #.item()
    train_txts = []
    train_labels = []
    temp_txts = []
    temp_labels = []
    for v in data.values():
        for r in v.values():
            #if 'txt' in r.keys():
            train_txts.extend(r['txt'])
            train_labels.extend(r[labelkey])
    for v in data_test.values():
        temp_txts.extend(r['txt'])
        temp_labels.extend(r[labelkey])
    if resplit is not None:
        temp_txts.extend(train_txts)
        temp_labels.extend(train_labels)
        train_size = int(len(temp_txts) * resplit[0])
        test_size = int(len(temp_txts) * resplit[1])
        val_size = len(temp_txts) - train_size - test_size
        temp_input, temp_output, temp_att_masks = process_data(temp_txts, temp_labels, vocab, max_len=max_len,
                                                            pad=pad, unk=unk, cls=cls, add_cls=add_cls,classes=classes)
        print((temp_input == 100).sum(), (temp_att_masks).sum() )                                                   
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
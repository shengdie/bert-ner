import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import rnn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from dataloader import load_tsv, load_vocab, Config, load_relation, triu_relation
from tqdm.auto import tqdm
from help_func import f1_score

class NerLSTM(nn.Module):
    """LSTM base line for named entity recognition."""
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.lstm = nn.LSTM(config.embed_size, config.hidden_size//2, num_layers=config.num_layers, batch_first=True, bidirectional=True)
        self.classifer =  nn.Linear(config.hidden_size, len(config.idx2tag))
        
        self.relation = nn.Linear(config.hidden_size, 10*10)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, y, relations, idx2tag=None):
        packx = rnn.pack_sequence(x, enforce_sorted=False)
        packy = rnn.pack_sequence(y, enforce_sorted=False)
        embed = rnn.PackedSequence(self.embed(packx.data), 
                                    packx.batch_sizes, 
                                    sorted_indices=packx.sorted_indices, 
                                    unsorted_indices=packx.unsorted_indices)
        #print(packx.data)
        sq_out, (hn, _) = self.lstm(embed)
        sq_out= self.dropout(sq_out.data)
        #print(hn.size())
        #rout = self.relation(self.dropout(hn[-1]))
        rout = self.relation(sq_out[:len(x)])

        logits = self.classifer(sq_out)

        if idx2tag is None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits[len(x):], packy.data)
            
            mask = relations > 0
            #print(relations.size())
            #print(rout.size())
            loss_fctr = nn.CrossEntropyLoss()
            if len(relations[mask]) > 0:
                loss += loss_fctr(rout.view(-1, 10, 10)[mask], relations[mask])
            return loss
        else:
            logits.detach()
            pred = torch.argmax(logits, 1)
            pred_rec, _ = rnn.pad_packed_sequence(rnn.PackedSequence(pred, packx.batch_sizes, 
                                            sorted_indices=packx.sorted_indices, 
                                            unsorted_indices=packx.unsorted_indices), batch_first=True)
            pred_rec = pred_rec.to('cpu').numpy()
            true = [[idx2tag[w.item()] for w in l] for l in y]
            pred_f = [[idx2tag[w] for w in p[:len(s)]] for s, p in zip(y, pred_rec[:,1:])]
            rout.detach()
            mask = relations > 0
            
            r_pred = torch.argmax(rout.view(-1,10,10)[mask], -1).to('cpu').numpy() if mask.sum() > 0 else []
            
            return true, pred_f, relations[mask].cpu().numpy(), r_pred


class BaselineDataSet(Dataset):
    def __init__(self, data_path, relation_path, config, device, unk='[UNK]'):
        x, y = load_tsv(data_path, add_cls=True)
        idx2token, token2idx = load_vocab(config.vocab_path)
        tag2idx = config.tag2idx
        self.x = [torch.LongTensor([token2idx[w] if w in token2idx.keys() else token2idx[unk] for w in l]).to(device) for l in x] 
        self.y = [torch.LongTensor([tag2idx[t] for t in l[1:]]).to(device) for l in y] # remove the first cls
        self.relations = load_relation(relation_path).astype(int)
        self.relations = triu_relation(self.relations, False, False)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.relations[idx]


class Baseline_Learner(object):
    def __init__(self, config):
        #config = Config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = BaselineDataSet(os.path.join(config.data_path, 'train.tsv'), os.path.join(config.data_path, 'train_rel.npz'), config, self.device)
        test_dataset = BaselineDataSet(os.path.join(config.data_path, 'test.tsv'), os.path.join(config.data_path, 'test_rel.npz'), config, self.device)
        cfn = lambda b: ([s[0] for s in b], [s[1] for s in b], torch.LongTensor([s[2] for s in b]).to(self.device)) # we need an empty collate_fn
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=cfn)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=cfn)
        
        self.model = NerLSTM(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.total_epoch = 0
        
        if torch.cuda.is_available():
            self.model.cuda()
        self.config = config
        self.train_loss = []
        self.val_loss = []
        self.f1score = []
        self.recall = []
        self.prec = []
        self.rel_acc = []
        self.rel_acc_r = []
        self.train_time = []
        self.record = {'train_loss': self.train_loss, 'val_loss': self.val_loss, 
                        'ner_f1score': self.f1score, 'ner_recall': self.recall, 
                        'ner_prec': self.prec, 'train_time': self.train_time, 
                        'rel_acc': self.rel_acc, 'rel_acc_r': self.rel_acc_r}

    def train(self, epochs=None, train_dataloader=None, test_dataloader=None, save_weight_path=None, start_save=3, lr=None, quiet=False):
        if lr is not None: self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if epochs is None: epochs = self.config.num_epochs
        if train_dataloader is None: train_dataloader = self.train_dataloader
        if test_dataloader is None: test_dataloader = self.test_dataloader

        total_ep = epochs + self.total_epoch

        ebatches = len(train_dataloader) // 10
        for i in range(self.total_epoch, total_ep):
            print('* [Epoch {}/{}]'.format(i+1, total_ep))
            start_time = time.time()
            self.model.train()
            with tqdm(total=len(train_dataloader), desc="Trainning", bar_format="{l_bar}{bar} [ time left: {remaining} ]", leave=False) as pbar:
                for step, batch in enumerate(train_dataloader):
                    # add batch to gpu
                    #batch = tuple(t.to(self.device) for t in batch)
                    #print(batch)
                    b_input_ids, b_labels, b_relations = batch

                    self.optimizer.zero_grad()

                    # forward pass
                    loss = self.model(b_input_ids, b_labels, b_relations)
                    # backward pass
                    loss.backward()
                    # clip grad
                    #torch.nn.utils.clip_grad_norm_(parameters=self.b_model.parameters(), max_norm=max_grad_norm)
                    # update parameters
                    self.optimizer.step()
                    #self.b_model.zero_grad()
                    pbar.update(1)
                    if step % ebatches == 0 and not quiet:
                        pbar.write("Step [{}/{}] train loss: {}".format(step, len(train_dataloader), loss.item()))
            self.train_time.append(time.time() - start_time)
            print('========== * Evaluating * ===========')
            ret_dic = self.predict(test_dataloader)
            if not quiet:
                print('- Time elasped: {:.5f} seconds\n'.format(self.train_time[-1]))
                # VALIDATION on validation set
                print('========== * Evaluating * ===========')
                print('Relation acc: {}'.format(ret_dic['rel_acc']))
                print('Relation acc with r: {}'.format(ret_dic['rel_acc_withr']))

                print("Validation loss: {}".format(ret_dic['loss']))
                print("Validation precision: {}".format(ret_dic['prec']))
                print("F1-Score: {}".format(ret_dic['f1-score']))
            
            # if ret_dic['f1-score'] > self.best_f1score:
            #     self.best_f1score = ret_dic['f1-score']
            #     if save_weight_path is not None and i - total_ep + epochs + 1 >= start_save:
            #         print('Saving weight...\n')
            #         self.save(save_weight_path)
            self.train_loss.append(loss.item())
            self.val_loss.append(ret_dic['loss'])
            self.f1score.append(ret_dic['f1-score'])
            self.recall.append(ret_dic['recall'])
            self.prec.append(ret_dic['prec'])
            self.rel_acc.append(ret_dic['rel_acc'])
            self.rel_acc_r.append(ret_dic['rel_acc_withr'])
            self.train_time_epoch = sum(self.train_time)/len(self.train_time)

            self.total_epoch += 1

    def predict(self, test_dataloader=None):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        self.model.eval()
        eval_loss, nb_eval_steps = 0, 0
        predictions , true_labels = [], []
        r_preds, r_trues = [], []
        ret_dic = {}
        for batch in test_dataloader:
            #batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_labels, b_relations = batch
            
            with torch.no_grad():
                tmp_eval_loss = self.model(b_input_ids, b_labels, b_relations)
                true, pred, r_true, r_pred = self.model(b_input_ids, b_labels, b_relations, self.config.idx2tag)

            predictions.extend(pred)

            true_labels.extend(true)
            r_preds.extend(r_pred)
            r_trues.extend(r_true)
            
            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        r_preds = np.array(r_preds)
        r_trues = np.array(r_trues)
        mask = r_trues > 1

        ret_dic['loss'] = eval_loss
        ret_dic['rel_acc'] = (r_preds == r_trues).mean()
        ret_dic['rel_acc_withr'] = (r_preds[mask] == r_trues[mask]).mean()
        ret_dic['prec'], ret_dic['recall'], ret_dic['f1-score'] = f1_score(true_labels, predictions)
        ret_dic['pred'] = predictions
        ret_dic['truth'] = true_labels
        ret_dic['r_true'] = r_trues
        ret_dic['r_pred'] = r_preds
        
        return ret_dic

if __name__ == "__main__":
    bs = Baseline_Learner('./baseline_config.json')
    bs.train()
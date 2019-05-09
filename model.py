import math
import time
import torch
import numpy as np
from collections import OrderedDict
from torch.optim import Adam
#from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam
#from berttokenclassification import BertForTokenClassification
#from bertrelationcls import BertForTokenClassification
#from bertrelationbert import BertForTokenClassification

from tqdm.auto import tqdm
from help_func import f1_score


class model(object):
    def __init__(self, config, ner_state_dict_path=None):
        """ner_state_dict is whole weight of BertForTokenClassification, which contains BERT weight + classifier weight,
        if it is not None, then pretrained bert weight will not be loaded.
        """
        self.config = config
        if config.triu:
            self.relnum = (config.max_concept + 1) * (config.max_concept)//2 if config.inc_dig else (config.max_concept) * (config.max_concept - 1)//2
            if config.inc_first: self.relnum += 1
        else:
            self.relnum = config.max_concept ** 2
        #b_conf = BertConfig.from_json_file(config.bert_conf_path)
        if ner_state_dict_path is None:
            tmp_d = torch.load(config.bert_weight_path, map_location='cpu')
            
            if len(tmp_d.keys()) > 201:
                bert_state_dict = OrderedDict()
                for i in list(tmp_d.keys())[:199]:
                    x = i
                    if i.find('bert') > -1:
                        x = '.'.join(i.split('.')[1:])
                    bert_state_dict[x] = tmp_d[i]
                #for i in list(tmp_d.keys())[:199]:
                #    state_dict[i] = tmp_d[i]
                # cls_weight = torch.Tensor(num_class, config.hidden_size)
                # cls_bias = torch.Tensor(num_class)
                # torch.nn.init.kaiming_uniform_(cls_weight, a=math.sqrt(5))
                # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(cls_weight)
                # bound = 1 / math.sqrt(fan_in)
                # torch.nn.init.uniform_(cls_bias, -bound, bound)
            
                # state_dict['classifier.weight'] = cls_weight
                # state_dict['classifier.bias'] = cls_bias
            else:
                bert_state_dict = tmp_d
        else:
            bert_state_dict = None

        if config.method == 'cls':
            from bertrelationcls import BertForTokenClassification
        elif config.method == 'bert':
            from bertrelationbert import BertForTokenClassification
        elif config.method == 'ave':
            from bertrelationave import BertForTokenClassification
        elif config.method == 'pair_ave':
            from bertrelationpair import BertForTokenClassification
        elif config.method == 'pair_transformer':
            from bertrelationpairtransformer import BertForTokenClassification
        
        self.b_model = BertForTokenClassification(config, self.relnum, bert_state_dict=bert_state_dict)
        if ner_state_dict_path is not None:
            self.load(ner_state_dict_path)

        
        #self.b_model.load_state_dict(state_dict)
        self.optimizer = None
        self.optim_params = None
        self.finetune_bert(False) # default fix bert
        #self.optimizer = Adam(self.b_model.parameters(), lr = config.lr, weight_decay=0.01)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            self.b_model.cuda()

        self.total_epoch = 0
        self.best_f1score = -0.1
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

    def train(self, train_dataloader, val_dataloader, epochs, tags, save_weight_path=None, start_save=3, lr=None):
        if lr is None:
            if self.optimizer is None:
                lr = self.config.lr
        if self.optimizer is None or lr is not None:        
            self.optimizer = BertAdam(self.b_model.parameters(), lr=lr, warmup=self.config.lr_warmup, t_total=len(train_dataloader) * epochs)
        total_ep = epochs + self.total_epoch

        ebatches = len(train_dataloader) // 10
        for i in range(self.total_epoch, total_ep):
            print('* [Epoch {}/{}]'.format(i+1, total_ep))
            start_time = time.time()
            self.b_model.train()
            with tqdm(total=len(train_dataloader), desc="Trainning", bar_format="{l_bar}{bar} [ time left: {remaining} ]", leave=False) as pbar:
                for step, batch in enumerate(train_dataloader):
                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_labels, b_input_mask, b_rels = batch

                    self.optimizer.zero_grad()

                    # forward pass
                    loss = self.b_model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels, relations=b_rels)
                    # backward pass
                    loss.backward()
                    # clip grad
                    #torch.nn.utils.clip_grad_norm_(parameters=self.b_model.parameters(), max_norm=max_grad_norm)
                    # update parameters
                    self.optimizer.step()
                    #self.b_model.zero_grad()
                    pbar.update(1)
                    if step % ebatches == 0:
                        pbar.write("Step [{}/{}] train loss: {}".format(step, len(train_dataloader), loss.item()))
            self.train_time.append(time.time() - start_time)
            print('- Time elasped: {:.5f} seconds\n'.format(self.train_time[-1]))
            # VALIDATION on validation set
            print('========== * Evaluating * ===========')
            ret_dic = self.predict(val_dataloader, tags)

            #print("Validation loss: {}".format(ret_dic['loss']))
            print("Validation precision: {}".format(ret_dic['prec']))
            print("F1-Score: {}".format(ret_dic['f1-score']))
            print('Relation acc: {}'.format(ret_dic['rel_acc']))
            print('Relation acc with r: {}\n'.format(ret_dic['rel_acc_withr']))
            
            if ret_dic['f1-score'] > self.best_f1score:
                self.best_f1score = ret_dic['f1-score']
                if save_weight_path is not None and i - total_ep + epochs + 1 >= start_save:
                    print('Saving weight...\n')
                    self.save(save_weight_path)
            self.train_loss.append(loss.item())
            #self.val_loss.append(ret_dic['loss'])
            self.f1score.append(ret_dic['f1-score'])
            self.recall.append(ret_dic['recall'])
            self.prec.append(ret_dic['prec'])
            self.rel_acc.append(ret_dic['rel_acc'])
            self.rel_acc_r.append(ret_dic['rel_acc_withr'])

            self.total_epoch += 1
        self.record['train_time_epoch'] = sum(self.train_time) /len(self.train_time)
        self.record['dataset_size'] = len(train_dataloader) * self.config.batch_size

    def predict(self, test_dataloader, tags):
        self.b_model.eval()
        eval_loss, nb_eval_steps = 0, 0
        predictions , true_labels = [], []
        relations_pred, relations_true, none_z = [], [], []
        
        ret_dic = {}
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_rels = batch
            
            with torch.no_grad():
                #tmp_eval_loss = self.b_model(b_input_ids, token_type_ids=None,
                #                    attention_mask=b_input_mask, labels=b_labels)
                logits, rout = self.b_model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            mask = b_input_mask.to('cpu').numpy().astype(bool)
            logits = logits.detach().cpu().numpy()

            rout = rout.detach().cpu().numpy()
            rels_true = b_rels.to('cpu').numpy().astype(int)
            rels_pred = np.argmax(rout.reshape((-1, self.relnum, 10)), axis=-1)
            nonez = rels_true != 0
            if self.config.inc_first:
                nonez[:, 0] = False
            if self.config.add_cls:
                mask[:,0] = False
            #(rels_pred[nonez] == rels_true[nonez]).sum() / nonez.sum()
            relations_pred.extend(rels_pred)
            relations_true.extend(rels_true)
            none_z.extend(nonez)

            label_ids = b_labels.to('cpu').numpy()
            pred = np.argmax(logits, axis=2)
            # don't count I-<P>
            mask[label_ids == 1] = False
            predictions.extend([p[m] for p, m in zip(pred, mask)])

            true_labels.extend([l[m] for l, m in zip(label_ids, mask)])
            
            #eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
        #return relations_pred, relations_true
        #eval_loss = eval_loss/nb_eval_steps
        pred_tags = [[tags[pi] for pi in p] for p in predictions]
        valid_tags = [[tags[li] for li in l] for l in true_labels]
        
        # relation accuray with no relation
        rel_acc = np.array([[(p[m] == t[m]).sum(), m.sum()] for p, t, m in zip(relations_pred, relations_true, none_z)])
        rel_acc = rel_acc[:,0].sum() / rel_acc[:,1].sum()
        # for relation accuray without relation
        relations_pred = np.array(relations_pred) 
        relations_true = np.array(relations_true)
        mask = relations_true > 1
        
        #ret_dic['loss'] = eval_loss
        ret_dic['prec'], ret_dic['recall'], ret_dic['f1-score'] = f1_score(valid_tags, pred_tags)
        ret_dic['rel_acc'] = rel_acc
        ret_dic['rel_acc_withr'] = (relations_pred[mask] == relations_true[mask]).mean()
        ret_dic['relat_pred'] = relations_pred
        ret_dic['relat_true'] = relations_true
        ret_dic['pred'] = pred_tags
        ret_dic['truth'] = valid_tags
        
        
        return ret_dic
    
    def finetune_bert(self, fineturn, layers=None, embed=False, namedp=None):
        if not fineturn:
            for params in self.b_model.bert.parameters():
                params.requires_grad = False
        elif layers is not None or embed:
            nlen = len('encoder.layer.1.')
            layer_name = ['encoder.layer.{}.'.format(i) if i < 10 else 'encoder.layer.{}'.format(i) for i in layers] if layers is not None else []
            emb = ['embeddings.word_embeddings.weight'] if embed else []
            for n, p in self.b_model.bert.named_parameters():
                if n[:nlen] in layer_name or n in emb: 
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif namedp is not None:
            for n, p in self.b_model.bert.named_parameters():
                if n in namedp: 
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for params in self.b_model.bert.parameters():
                params.requires_grad = True




    # def full_trainning(self, full_trainning, lr=1e-3):
    #     """If full_training, then train the classifier + BERT, else only train the classifier
    #     """

    #     if full_trainning:
    #         param_optimizer = list(self.b_model.named_parameters())
    #         no_decay = ['bias', 'gamma', 'beta']
    #         optimizer_grouped_parameters = [
    #             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #             'weight_decay_rate': 0.01},
    #             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #             'weight_decay_rate': 0.0}
    #         ]
    #     else:
    #         param_optimizer = list(self.b_model.classifier.named_parameters()) 
    #         optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer], 'weight_decay_rate': 0.01}]
    #     self.optimizer = Adam(optimizer_grouped_parameters, lr=lr)

    def save(self, path):
        torch.save(self.b_model.state_dict(), path)
    def save_hist(self, path):
        if path is not None:
            hist = np.array([self.train_loss, self.rel_acc, self.prec, self.recall, self.f1score])
            np.savetxt(path, hist.T, header='train loss, rel acc, precision, recall, f1-score')
    def load(self, path):
        self.b_model.load_state_dict(torch.load(path, map_location='cpu'))



class RelationLearner(object):
    def __init__(self, config, ner_state_dict_path=None):
        """ner_state_dict is whole weight of BertForTokenClassification, which contains BERT weight + classifier weight,
        if it is not None, then pretrained bert weight will not be loaded.
        """
        self.config = config
        #b_conf = BertConfig.from_json_file(config.bert_conf_path)
        if ner_state_dict_path is None:
            tmp_d = torch.load(config.bert_weight_path, map_location='cpu')
            
            if len(tmp_d.keys()) > 201:
                bert_state_dict = OrderedDict()
                for i in list(tmp_d.keys())[:199]:
                    x = i
                    if i.find('bert') > -1:
                        x = '.'.join(i.split('.')[1:])
                    bert_state_dict[x] = tmp_d[i]
                #for i in list(tmp_d.keys())[:199]:
                #    state_dict[i] = tmp_d[i]
                # cls_weight = torch.Tensor(num_class, config.hidden_size)
                # cls_bias = torch.Tensor(num_class)
                # torch.nn.init.kaiming_uniform_(cls_weight, a=math.sqrt(5))
                # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(cls_weight)
                # bound = 1 / math.sqrt(fan_in)
                # torch.nn.init.uniform_(cls_bias, -bound, bound)
            
                # state_dict['classifier.weight'] = cls_weight
                # state_dict['classifier.bias'] = cls_bias
            else:
                bert_state_dict = tmp_d
        else:
            bert_state_dict = None
        from bertpurerelation import BertForRelationClassification
        
        self.b_model = BertForRelationClassification(config, bert_state_dict=bert_state_dict)
        if ner_state_dict_path is not None:
            self.load(ner_state_dict_path)

        
        #self.b_model.load_state_dict(state_dict)
        self.optimizer = None
        self.optim_params = None
        self.finetune_bert(False) # default fix bert
        #self.optimizer = Adam(self.b_model.parameters(), lr = config.lr, weight_decay=0.01)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            self.b_model.cuda()

        self.total_epoch = 0
        self.best_acc = -0.1
        self.train_loss = []
        self.val_loss = []
        self.rel_acc = []
        self.rel_raw_acc = []

    def train(self, train_dataloader, val_dataloader, epochs, save_weight_path=None, start_save=3, lr=None):
        if lr is None:
            if self.optimizer is None:
                lr = self.config.lr
        if self.optimizer is None or lr is not None:        
            self.optimizer = BertAdam(self.b_model.parameters(), lr=lr, warmup=self.config.lr_warmup, t_total=len(train_dataloader) * epochs)
        total_ep = epochs + self.total_epoch

        ebatches = len(train_dataloader) // 10
        for i in range(self.total_epoch, total_ep):
            print('* [Epoch {}/{}]'.format(i+1, total_ep))
            start_time = time.time()
            self.b_model.train()
            with tqdm(total=len(train_dataloader), desc="Trainning", bar_format="{l_bar}{bar} [ time left: {remaining} ]", leave=False) as pbar:
                for step, batch in enumerate(train_dataloader):
                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_type_ids, b_input_mask, b_rels = batch

                    self.optimizer.zero_grad()

                    # forward pass
                    loss = self.b_model(b_input_ids, token_type_ids=b_type_ids,
                                attention_mask=b_input_mask, relations=b_rels)
                    # backward pass
                    loss.backward()
                    # clip grad
                    #torch.nn.utils.clip_grad_norm_(parameters=self.b_model.parameters(), max_norm=max_grad_norm)
                    # update parameters
                    self.optimizer.step()
                    #self.b_model.zero_grad()
                    pbar.update(1)
                    if step % ebatches == 0:
                        pbar.write("Step [{}/{}] train loss: {}".format(step, len(train_dataloader), loss.item()))

            print('- Time elasped: {:.5f} seconds\n'.format(time.time() - start_time))
            # VALIDATION on validation set
            print('========== * Evaluating * ===========')
            ret_dic = self.predict(val_dataloader)

            print("Validation loss: {}".format(ret_dic['loss']))
            #print("Validation precision: {}".format(ret_dic['prec']))
            #print("F1-Score: {}".format(ret_dic['f1-score']))
            print('Relation acc: {}'.format(ret_dic['rel_acc']))
            print('Relation raw acc: {}\n'.format(ret_dic['raw_rel_acc']))
            
            if ret_dic['rel_acc'] > self.best_acc:
                self.best_acc = ret_dic['rel_acc']
                if save_weight_path is not None and i - total_ep + epochs + 1 >= start_save:
                    print('Saving weight...\n')
                    self.save(save_weight_path)
            self.train_loss.append(loss.item())
            #self.val_loss.append(ret_dic['loss'])
            #self.f1score.append(ret_dic['f1-score'])
            #self.recall.append(ret_dic['recall'])
            #self.prec.append(ret_dic['prec'])
            self.rel_acc.append(ret_dic['rel_acc'])
            self.rel_raw_acc.append(ret_dic['raw_rel_acc'])

            self.total_epoch += 1

    def predict(self, test_dataloader):
        self.b_model.eval()
        eval_loss, nb_eval_steps = 0, 0
        relations_pred, relations_true= [], []
        
        ret_dic = {}
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_type_ids, b_input_mask, b_rels = batch
            
            with torch.no_grad():
                tmp_eval_loss = self.b_model(b_input_ids, token_type_ids=b_type_ids,
                                    attention_mask=b_input_mask, relations=b_rels)
                logits = self.b_model(b_input_ids, token_type_ids=b_type_ids,
                                attention_mask=b_input_mask)

            #mask = b_input_mask.to('cpu').numpy().astype(bool)
            logits = logits.detach().cpu().numpy()

            rels_true = b_rels.to('cpu').numpy().astype(int)
            rels_pred = np.argmax(logits.reshape((-1, 10)), axis=-1)
            #nonez = rels_true != 0
            #(rels_pred[nonez] == rels_true[nonez]).sum() / nonez.sum()
            relations_pred.extend(rels_pred)
            relations_true.extend(rels_true)
            #none_z.extend(nonez)
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
        #print(relations_true, none_z)
        eval_loss /= nb_eval_steps
        #sreturn relations_pred, relations_true, none_z
        #rel_acc = np.array([[(p[m] == t[m]).sum(), m.sum()] for p, t, m in zip(relations_pred, relations_true, none_z)])
        relations_true = np.array(relations_true)
        relations_pred = np.array(relations_pred)
        mask = relations_true > 1
        rel_acc = (relations_true[mask] == relations_pred[mask]).sum() / mask.sum()
        ret_dic['loss'] = eval_loss
        #ret_dic['prec'], ret_dic['recall'], ret_dic['f1-score'] = f1_score(valid_tags, pred_tags)
        ret_dic['rel_acc'] = rel_acc
        ret_dic['raw_rel_acc'] = np.average(relations_true == relations_pred)
        ret_dic['relat_pred'] = relations_pred
        ret_dic['relat_true'] = relations_true
        
        return ret_dic
    
    def finetune_bert(self, fineturn, layers=None, embed=False, namedp=None):
        if not fineturn:
            for params in self.b_model.bert.parameters():
                params.requires_grad = False
        elif layers is not None or embed:
            nlen = len('encoder.layer.1.')
            layer_name = ['encoder.layer.{}.'.format(i) if i < 10 else 'encoder.layer.{}'.format(i) for i in layers] if layers is not None else []
            emb = ['embeddings.word_embeddings.weight'] if embed else []
            for n, p in self.b_model.bert.named_parameters():
                if n[:nlen] in layer_name or n in emb: 
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif namedp is not None:
            for n, p in self.b_model.bert.named_parameters():
                if n in namedp: 
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for params in self.b_model.bert.parameters():
                params.requires_grad = True


    def save(self, path):
        torch.save(self.b_model.state_dict(), path)
    def save_hist(self, path):
        if path is not None:
            hist = np.array([self.train_loss, self.rel_acc, self.rel_raw_acc])
            np.savetxt(path, hist.T, header='train loss, rel acc, rel raw acc')
    def load(self, path):
        self.b_model.load_state_dict(torch.load(path, map_location='cpu'))




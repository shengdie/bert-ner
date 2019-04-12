import math
import time
import torch
import numpy as np
from collections import OrderedDict
from torch.optim import Adam
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert import BertTokenizer, BertConfig

from tqdm.auto import tqdm
from help_func import f1_score


class model(object):
    def __init__(self, config_json, bert_weight, num_class=4):
        config = BertConfig.from_json_file(config_json)
        self.b_model = BertForTokenClassification(config=config, num_labels=num_class)

        tmp_d = torch.load(bert_weight, map_location='cpu')
        
        if len(tmp_d.keys()) > 201:
            state_dict = OrderedDict()
            for i in list(tmp_d.keys())[:199]:
                state_dict[i] = tmp_d[i]
            cls_weight = torch.Tensor(num_class, config.hidden_size)
            cls_bias = torch.Tensor(num_class)
            torch.nn.init.kaiming_uniform_(cls_weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(cls_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(cls_bias, -bound, bound)
        
            state_dict['classifier.weight'] = cls_weight
            state_dict['classifier.bias'] = cls_bias
        else:
            state_dict = tmp_d
        
        self.b_model.load_state_dict(state_dict)
        # default train the classifier only
        self.full_trainning(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.b_model.cuda()

        self.total_epoch = 0
        self.best_loss = 1e10

    def train(self, train_dataloader, val_dataloader, epochs, tags, save_weight_path=None, start_save=3):
        total_ep = epochs + self.total_epoch
        for i in range(self.total_epoch, total_ep):
            print('* Epoch {}/{}'.format(i+1, total_ep))
            start_time = time.time()
            self.b_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=len(train_dataloader), desc="Trainning", bar_format="{l_bar}{bar} [ time left: {remaining} ]", leave=False) as pbar:
                for step, batch in enumerate(train_dataloader):
                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_labels, b_input_mask = batch

                    # forward pass
                    loss = self.b_model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
                    # backward pass
                    loss.backward()
                    # track train loss
                    tr_loss += loss.item()
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1
                    # update parameters
                    self.optimizer.step()
                    self.b_model.zero_grad()
                    pbar.update(1)

            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            # VALIDATION on validation set
            ret_dic = self.predict(val_dataloader, tags)

            print("Validation loss: {}".format(ret_dic['loss']))
            print("Validation Accuracy: {}".format(ret_dic['accuracy']))
            print("F1-Score: {}".format(ret_dic['f1-score']))
            print('- Time elasped: {:.5f} seconds\n'.format(time.time() - start_time))
            if save_weight_path is not None and ret_dic['loss'] < self.best_loss and i - total_ep + epochs + 1 >= start_save:
                print('Saving weight...\n')
                self.best_loss = ret_dic['loss']
                self.save(save_weight_path)

            self.total_epoch += 1

    def predict(self, test_dataloader, tags):
        self.b_model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        ret_dic = {}
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_labels,b_input_mask = batch
            
            with torch.no_grad():
                tmp_eval_loss = self.b_model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                logits = self.b_model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=2)
            predictions.extend(pred_flat)
            pred_flat = pred_flat.flatten()
            true_labels.extend(label_ids)
            labels_flat = label_ids.flatten()
            
            tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
            
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps

        pred_tags = [[tags[pi] for pi in p] for p in predictions]
        valid_tags = [[tags[li] for li in l] for l in true_labels]

        ret_dic['loss'] = eval_loss
        ret_dic['accuracy'] = eval_accuracy/nb_eval_steps
        ret_dic['f1-score'] = f1_score(valid_tags, pred_tags)
        ret_dic['pred'] = np.array(predictions)
        ret_dic['truth'] = np.array(true_labels)
        
        return ret_dic

    def full_trainning(self, full_trainning):
        """If full_training, then train the classifier + BERT, else only train the classifier
        """

        if full_trainning:
            param_optimizer = list(self.b_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.b_model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    def save(self, path):
        torch.save(self.b_model.state_dict(), path)




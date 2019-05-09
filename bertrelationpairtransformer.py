import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from help_func import pair, get_e
from modules import Transformer


class BertForTokenClassification(nn.Module):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.
    Params:
        `config`: a config.
        `bert_config`: bert config
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, relnum, bert_state_dict=None):
        super().__init__()
        self.num_labels = len(config.idx2tag)
        self.idx2tag = config.idx2tag
        self.config = config
        bert_config = BertConfig.from_json_file(config.bert_conf_path)
        self.bert = BertModel(bert_config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        # we don't fine tune bert, it requires large GPU mem
        #self.bert.eval()
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        rel_bert_conf = BertConfig.from_dict({
                                            "attention_probs_dropout_prob": 0.1,
                                            "hidden_act": "gelu",
                                            "hidden_dropout_prob": 0.1,
                                            "hidden_size": 768,
                                            "initializer_range": 0.02,
                                            "intermediate_size": 3072,
                                            "max_position_embeddings": 512,
                                            "num_attention_heads": 12,
                                            "num_hidden_layers": 1,
                                            "type_vocab_size": 2,
                                            "vocab_size": 28996})

        self.transformer = Transformer(rel_bert_conf)

        #self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
        #                   input_size=768, hidden_size=768//2, batch_first=True)
        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.relnum = relnum
        self.relation = nn.Linear(bert_config.hidden_size, 10)
        #self.relation = nn.Bilinear(bert_config.hidden_size, bert_config.hidden_size, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, relations=None):
        # with torch.no_grad():
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #loss = torch.tensor(0.).cuda()
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #loss = torch.argmax(logits, -1) > 0
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # return loss
            rout = []
            rel = []
            loss_fctr = nn.CrossEntropyLoss()
            for i in range(len(labels)):
                entities = get_e(labels[i])
                #entities.extend([None])
                #temp_batch = []
                if len(entities) > 1:
                    ent_o = [sequence_output[i, slice(*ent)] for ent in entities] + [None] * (self.config.max_concept - len(entities))
                    paris = pair(*ent_o)
                    #rel.extend(relations[i][:len(paris)])
                    #print(labels[i], entities, relations[i])
                    for (ent1, ent2), r in zip(paris, relations[i]):
                        if ent1 is not None and ent2 is not None:
                            t_type_ids = [0] * (len(ent1) + 1) + [1] * len(ent2) # add cls
                            inp = torch.cat((sequence_output[i,0:1], ent1, ent2)).unsqueeze(0)
                            #print(inp.size(), len(t_type_ids))
                            _, pooler = self.transformer(inp, torch.LongTensor(t_type_ids).cuda().unsqueeze(0), torch.ones(len(t_type_ids)).cuda().unsqueeze(0))
                            
                            rout.append(self.relation(pooler)[0])
                            #print(rout[-1].size())
                            #rout.append(self.relation(ent1 , ent2))
                            rel.append(r)
                            #print(rout[-1], r)
            #return    

            if len(rel) > 0:
                #print (torch.rout, rel)
                rout = torch.stack(rout)
                rel = torch.stack(rel)
                #print(rout.size(), rel.size())
                #print(rout.size(), rel.size())
                loss += loss_fctr(rout, rel)
            return loss
        else:
            plabel = torch.argmax(logits, -1)
            rout = []
            rfake = torch.zeros(10).cuda()
            rfake[0] = torch.tensor(1).cuda()
            for i in range(len(plabel)):
                entities = get_e(plabel[i])[:self.config.max_concept]
                #le = len(entities)
                #print(entities)
                
                if len(entities) > 1:
                    temp = []
                    ent_o = [sequence_output[i, slice(*ent)] for ent in entities] + [None] * (self.config.max_concept - len(entities))
                    #print(ent_o)
                    paris = pair(*ent_o)
                    #print(paris)
                    for ent1, ent2 in paris:
                        if ent1 is not None and ent2 is not None:
                            t_type_ids = [0] * (len(ent1) + 1) + [1] * len(ent2) # add cls
                            #inp = 
                            _, pooler = self.transformer(torch.cat((sequence_output[i,0:1], ent1, ent2)).unsqueeze(0), 
                                                        torch.LongTensor(t_type_ids).cuda().unsqueeze(0), torch.ones(len(t_type_ids)).cuda().unsqueeze(0))
                            temp.append(self.relation(pooler)[0])
                            #temp.append(self.relation(ent1 + ent2))
                            #print(temp)
                        else:
                            temp.append(rfake)
                            #print(temp)
                    #print(temp)
                else:
                    temp = [rfake]* self.relnum
                #print(temp)
                rout.append(torch.stack(temp))    
            rout = torch.stack(rout)

        return logits, rout

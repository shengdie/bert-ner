import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from help_func import pair, get_e
from modules import Transformer


class BertForTokenClassification(nn.Module):
    """BERT model for token-level classification, use pair transformer."""

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

        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.relnum = relnum
        self.relation = nn.Linear(bert_config.hidden_size, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, relations=None):
        # with torch.no_grad():
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if self.config.add_cls: attention_mask[:,0] = 0
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

                if len(entities) > 1:
                    ent_o = [sequence_output[i, slice(*ent)] for ent in entities] + [None] * (self.config.max_concept - len(entities))
                    paris = pair(*ent_o)
                    for (ent1, ent2), r in zip(paris, relations[i]):
                        if ent1 is not None and ent2 is not None:
                            t_type_ids = [0] * (len(ent1) + 1) + [1] * len(ent2) # add cls
                            inp = torch.cat((sequence_output[i,0:1], ent1, ent2)).unsqueeze(0)
                            
                            _, pooler = self.transformer(inp, torch.LongTensor(t_type_ids).cuda().unsqueeze(0), torch.ones(len(t_type_ids)).cuda().unsqueeze(0))
                            rout.append(self.relation(pooler)[0])
                            rel.append(r)

            if len(rel) > 0:
                rout = torch.stack(rout)
                rel = torch.stack(rel)
                loss += loss_fctr(rout, rel)
            return loss
        else:
            plabel = torch.argmax(logits, -1)
            rout = []
            rfake = torch.zeros(10).cuda()
            rfake[0] = torch.tensor(1).cuda()
            for i in range(len(plabel)):
                entities = get_e(plabel[i])[:self.config.max_concept]
                
                if len(entities) > 1:
                    temp = []
                    ent_o = [sequence_output[i, slice(*ent)] for ent in entities] + [None] * (self.config.max_concept - len(entities))
                    
                    paris = pair(*ent_o)
                    
                    for ent1, ent2 in paris:
                        if ent1 is not None and ent2 is not None:
                            t_type_ids = [0] * (len(ent1) + 1) + [1] * len(ent2) # add cls
                            #inp = 
                            _, pooler = self.transformer(torch.cat((sequence_output[i,0:1], ent1, ent2)).unsqueeze(0), 
                                                        torch.LongTensor(t_type_ids).cuda().unsqueeze(0), torch.ones(len(t_type_ids)).cuda().unsqueeze(0))
                            temp.append(self.relation(pooler)[0])
                        else:
                            temp.append(rfake)
                else:
                    temp = [rfake]* self.relnum
                rout.append(torch.stack(temp))    
            rout = torch.stack(rout)

        return logits, rout

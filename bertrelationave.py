import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
#from help_func import get_entities_p, pair


class BertForTokenClassification(nn.Module):
    """BERT model for token-level classification. and relation extraction use ave pooling."""

    def __init__(self, config, relnum, bert_state_dict=None):
        super().__init__()
        self.config = config
        self.num_labels = len(config.idx2tag)
        self.idx2tag = config.idx2tag
        bert_config = BertConfig.from_json_file(config.bert_conf_path)
        self.bert = BertModel(bert_config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.relnum = relnum
        self.relation = nn.Linear(bert_config.hidden_size, relnum * 10)

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

            plabel = labels.unsqueeze(-1) > 1
            sequence_output = sequence_output.masked_fill(plabel == False, 0).sum(dim=1)
            plabel = plabel.sum(dim=1)
            plabel = plabel.masked_fill(plabel == 0, 1)
            sequence_output = sequence_output / plabel.float()
            rout = self.relation(sequence_output)

            loss_fctr = nn.CrossEntropyLoss()
            mask = relations > 0
            if len(relations[mask]) > 0:
                loss += loss_fctr(rout.view(-1, self.relnum, 10)[mask], relations[mask])

            #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
            return loss
        else:
            plabel = torch.argmax(logits, -1)
            plabel = plabel.masked_fill(attention_mask == 0, 0) > 1
            plabel = plabel.unsqueeze(-1)
            
            sequence_output = sequence_output.masked_fill(plabel == False, 0).sum(dim=1)# / plabel.sum(dim=1)
            plabel = plabel.sum(dim=1)
            plabel = plabel.masked_fill(plabel == 0, 1)
            sequence_output = sequence_output / plabel.float()
            rout = self.relation(sequence_output)

        return logits, rout

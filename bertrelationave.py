import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
#from help_func import get_entities_p, pair


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
        bert_config = BertConfig.from_json_file(config.bert_conf_path)
        self.bert = BertModel(bert_config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        # we don't fine tune bert, it requires large GPU mem
        #self.bert.eval()
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        #self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
        #                   input_size=768, hidden_size=768//2, batch_first=True)
        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.relnum = relnum
        self.relation = nn.Linear(bert_config.hidden_size, relnum * 10)
        #self.relation = nn.Bilinear(bert_config.hidden_size, bert_config.hidden_size, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, relations=None):
        # with torch.no_grad():
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        #rout = self.relation(sequence_output[:, 0, :])
        #tags_output, _ = self.rnn(sequence_output)
        #tags_output = self.dropout(tags_output)
        logits = self.classifier(sequence_output)
        #torch.argmax(logits, dim=)
        #plabel = torch.argmax(logits, -1)
        #plabel = plabel.masked_fill(attention_mask == 0, 0) > 1
        #plabel = plabel.unsqueeze(-1)
        
        #sequence_output = sequence_output.masked_fill(plabel == False, 0).sum(dim=1)# / plabel.sum(dim=1)
        #plabel = plabel.sum(dim=1)
        #plabel = plabel.masked_fill(plabel == 0, 1)
        #sequence_output = sequence_output / plabel.float()
        #print(sequence_output.size())
        #rout = self.relation(sequence_output)

        #loss = torch.tensor(0.).cuda()
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            attention_mask[:,0] = 0
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
            #if relations is not None:
            #label_tags = [self.idx2tag(l) for l in (l > 0)]
           # plabel = labels.masked_fill(attention_mask == 0, 0) > 1
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
        #elif relations is not None:
        #    loss_fct = nn.CrossEntropyLoss()
        #    loss = loss_fct(rout.view(-1, 10), relations.view(-1))
        #    return loss

        return logits, rout

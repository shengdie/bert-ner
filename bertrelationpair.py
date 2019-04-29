import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from help_func import pair, get_e


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

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
                           input_size=768, hidden_size=768//2, batch_first=True)
        self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.relnum = relnum
        #self.relation = nn.Linear(bert_config.hidden_size, 10)
        self.relation = nn.Bilinear(bert_config.hidden_size, bert_config.hidden_size, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, relations=None):
        # with torch.no_grad():
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        #rout = self.relation(sequence_output[:, 0, :])
        tags_output, _ = self.rnn(sequence_output)
        tags_output = self.dropout(sequence_output)
        logits = self.classifier(tags_output)


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

                le = len(entities)
                
                if le == 2:
                    rout.append(self.relation(sequence_output[i, slice(*entities[0])].mean(dim=0), sequence_output[i, slice(*entities[1])].mean(dim=0)))
                    rel.append(relations[i,0])
                    #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
                elif le == 3:
                    c1 = sequence_output[i, slice(*entities[0])].mean(dim=0)
                    c2 = sequence_output[i, slice(*entities[1])].mean(dim=0)
                    c3 = sequence_output[i, slice(*entities[2])].mean(dim=0)
                    rout1 = self.relation(c1, c2)
                    rout2 = self.relation(c1, c3)
                    rout3 = self.relation(c2, c3)
                    rout.extend([rout1, rout2, rout3])
                    rel.extend(relations[i])
            
                    #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
            #print (torch.rout, rel)
            rout = torch.stack(rout)
            rel = torch.stack(rel)
            #print(rout.size(), rel.size())

            loss += loss_fctr(rout, rel)
            return loss
        else:
            plabel = torch.argmax(logits, -1)
            rout = []
            rel = []
            for i in range(len(plabel)):
                entities = get_e(plabel[i])
                le = len(entities)
                if le == 2:
                    ro = self.relation(sequence_output[i, slice(*entities[0])].mean(dim=0), sequence_output[i, slice(*entities[1])].mean(dim=0))
                    rfake = torch.zeros_like(ro).cuda()
                    rfake[0] = torch.tensor(1).cuda()
                    ro = torch.stack((ro, rfake, rfake))
                    rout.append(ro)
                    #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
                elif le >= 3:
                    c1 = sequence_output[i, slice(*entities[0])].mean(dim=0)
                    c2 = sequence_output[i, slice(*entities[1])].mean(dim=0)
                    c3 = sequence_output[i, slice(*entities[2])].mean(dim=0)
                    rout1 = self.relation(c1, c2)
                    rout2 = self.relation(c1, c3)
                    rout3 = self.relation(c2, c3)
                    rout.append(torch.stack((rout1, rout2, rout3)))
                    #rel.extend(relations[i])
                else:
                    rfake = torch.zeros(10).cuda()
                    rfake[0] = torch.tensor(1).cuda()
                    rout.append(torch.stack((rfake, rfake, rfake)))
            rout = torch.stack(rout)
                    #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
                #loss += loss_fctr(rout.view(-1, 10), relations.view(-1))
        #elif relations is not None:
        #    loss_fct = nn.CrossEntropyLoss()
        #    loss = loss_fct(rout.view(-1, 10), relations.view(-1))
        #    return loss

        return logits, rout

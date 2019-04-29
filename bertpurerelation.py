import json
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from modules import Transformer, PoolerClassifier

class BertForRelationClassification(nn.Module):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.
    Params:
        `config`: a config.
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

    def __init__(self, config, bert_state_dict=None):
        super().__init__()
        self.num_labels = len(config.idx2tag)
        bert_config = BertConfig.from_json_file(config.bert_conf_path)
        self.bert = BertModel(bert_config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        # we don't fine tune bert, it requires large GPU mem
        #self.bert.eval()
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        #self.relation1 = nn.Linear(bert_config.hidden_size, 3072)
        #self.activation = nn.Relu()
        self.relation = nn.Linear(bert_config.hidden_size, 10)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, relations=None):
        # with torch.no_grad():
        _, pooler = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #sequence_output = self.dropout(sequence_output)
        #sequence_output, _ = self.rnn(sequence_output)
        #sequence_output, _ = self.rnn(sequence_output, token_type_ids, attention_mask)
        
        #rout = self.relation(sequence_output)
        
        #sequence_output = self.dropout(sequence_output)
        #logits = self.classifier(sequence_output)

        pooler = self.dropout(pooler)
           
        #print(sequence_output.size())
        #rout = nn.functional.relu(self.relation1(pooler))
        #rout = self.dropout(rout)
        rout = self.relation(pooler)


        #loss = torch.tensor(0.).cuda()
        if relations is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(rout.view(-1, 10), relations.view(-1))
            return loss

        return rout

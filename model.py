import torch as torch
from transformers import MPNetTokenizer, MPNetModel
from transformers.models.mpnet import MPNetPreTrainedModel
from util import *
from torch import nn
from transformers import AutoModelForMaskedLM,AutoModel

class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, centroids=None,
                labeled=False, feature_ext=False):

        encoded_layer_12, pooled_output_ori = self.bert(input_ids, token_type_ids, attention_mask,
                                                        output_all_encoded_layers=False)

        pooled_output = self.mean_pooling(encoded_layer_12, attention_mask)

        pooled_output = self.dense(pooled_output)
        proj_output = self.proj(pooled_output)

        logits=self.classifier(pooled_output)
        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits

class MPNetForModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.backbone.config.hidden_size
        self.dropout_prob = self.backbone.config.hidden_dropout_prob

        self.dense = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.backbone.config.hidden_dropout_prob)
        )

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            mode=None,
            feature_ext=False
    ):
        if 'bert' in self.model_name:
            outputs = self.backbone(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids, attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        pooled_output = self.dense(pooled_output)

        proj_output = self.proj(pooled_output)
        logits = self.classifier(pooled_output)

        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        elif mode == "train":
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits

import numpy as np
import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, BertModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

class Bart_summarization(nn.Module):
    def __init__(self):
        super(Bart_summarization, self).__init__()
        
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        # self.model_config = BartConfig.from_pretrained('facebook/bart-base')
        
        # bias
        # self.encoder = self.model.get_encoder()
        # self.decoder = self.model.get_decoder()

        # self.hidden_dim = self.encoder.embed_tokens.embedding_dim
        # self.vocab_size = self.model.lm_head.out_features

    def forward(self, input_ids, attetion_mask, labels, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attetion_mask, labels=labels) # input_ids 만들기 (attention mask?)

        return outputs

    def generate(self, b_input_ids, max_length=1024):
        return self.model.generate(b_input_ids)

class BERTClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2 # bert hidden_dim=768
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.classifier = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask) # (batch_size, sequence_length, hidden_size)

        last_hidden_state_cls = outputs[0][:, 0, :] # 'latent_hidden_state' (batch_size, hidden_size) CLS token 추출

        logits = self.classifier(last_hidden_state_cls)

        return logits
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BertEncodingNetwork(nn.Module):

    def __init__(self, bert, bert_size, cnn_size, dropout):
        super(BertEncodingNetwork, self).__init__()
        self.bert = bert
        self.bert_size = bert_size
        self.sentence_transform = nn.Sequential(
            nn.Linear(bert_size, cnn_size),
            nn.Dropout(dropout)
        )

    def forward(self, bert_token, bert_segment):
        # BERT encoding
        encoder_layer, _ = self.bert(bert_token, bert_segment, output_all_encoded_layers=False)
        batch_size, segment_len = bert_segment.size()
        max_segment_len = bert_segment.argmax(dim=-1, keepdim=True)
        batch_arrange = torch.arange(segment_len).unsqueeze(0).expand(batch_size, segment_len).to(bert_segment.device)
        segment_mask = batch_arrange <= max_segment_len
        sentence_mask = segment_mask & (1 - bert_segment).byte()
        sentence_lens = sentence_mask.long().sum(dim=1, keepdim=True)

        # sentence encode layer
        max_len = sentence_lens.max().item()
        sentence = encoder_layer[:, 0: max_len].contiguous()
        sentence_mask = sentence_mask[:, 0: max_len].contiguous()
        sentence = sentence.masked_fill(sentence_mask.unsqueeze(-1) == 0, 0)
        sentence = self.sentence_transform(sentence)

        return sentence


class BertCNNNetwork(nn.Module):

    def __init__(self, bert, bert_size, dropout, cnn_size, filter_size, filter_nums, num_categories):
        super(BertCNNNetwork, self).__init__()
        self.bert = BertEncodingNetwork(bert, bert_size, cnn_size, dropout)
        self.dropout = dropout
        self.num_categories = num_categories
        self.cnn_size = cnn_size
        self.filter_size = filter_size
        self.filter_nums = filter_nums
        self.cnnDrop = nn.Dropout(0.5)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.cnn_size, self.filter_nums, kernel_size=k),
                # nn.BatchNorm1d(num_features=feature_size),
                nn.ReLU(),
            ) for k in self.filter_size])
        self.linear = nn.Linear(self.filter_nums * len(self.filter_size), self.num_categories, bias=False)

    def load_sentiment(self, path):
        sentiment = np.load(path)
        e1 = np.mean(sentiment)
        d1 = np.std(sentiment)
        e2 = 0
        d2 = np.sqrt(2.0 / (sentiment.shape[0] + sentiment.shape[1]))
        sentiment = (sentiment - e1) / d1 * d2 + e2
        self.guide_capsule.data.copy_(torch.tensor(sentiment))

    @staticmethod
    def pooling(conv, x):
        x = conv(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        return x

    def forward(self, bert_token, bert_segment):
        bertout = self.bert(bert_token, bert_segment)  # [b,l,dim]
        bertout = bertout.transpose(1, 2)
        out = [conv(bertout) for conv in self.convs]
        out = [F.max_pool1d(o, kernel_size=o.size(-1)).squeeze(-1) for o in out]  # [B,D,L] -> [B,O,L]*len -> [B,O,1]*len -> [B,O]*len
        out = torch.cat(out, -1)  # [B, len(k)*O]
        # out = self.cnnDrop(out)
        out = self.linear(out)
        # out = torch.softmax(out, -1)
        return out


class BertFcNetwork(nn.Module):

    def __init__(self, bert, bert_size, dropout, cnn_size, num_categories):
        super(BertFcNetwork, self).__init__()
        self.num_categories = num_categories
        self.cnn_size = cnn_size
        self.bert = bert
        self.bert_size = bert_size
        self.sentence_transform = nn.Sequential(
            nn.Linear(bert_size, cnn_size),
            nn.Dropout(dropout)
        )
        self.linear = nn.Linear(self.cnn_size, self.num_categories, bias=False)

    def forward(self, bert_token, bert_segment):
        _, bertout = self.bert(bert_token, bert_segment, output_all_encoded_layers=False)
        out = self.sentence_transform(bertout)
        out = self.linear(out)
        # out = torch.softmax(out, -1)
        return out

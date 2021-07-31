import torch
from torch import nn
import torch.nn.functional as F
from src.module.utils.squash import squash
from src.module.attention.bilinear_attention import BilinearAttention
import numpy as np


class XlnetEncodingNetwork(nn.Module):

    def __init__(self, xlnet, xlnet_size, hidden_size, dropout):
        super(XlnetEncodingNetwork, self).__init__()
        self.xlnet = xlnet
        self.xlnet_size = xlnet_size
        self.sentence_transform = nn.Sequential(
            nn.Linear(xlnet_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, xlnet_token, xlnet_segment):
        # xlnet encoding
        encoder_layer = self.xlnet(input_ids=xlnet_token, token_type_ids=xlnet_segment)[0]
        batch_size, segment_len = xlnet_segment.size()
        max_segment_len = xlnet_segment.argmax(dim=-1, keepdim=True)
        batch_arrange = torch.arange(segment_len).unsqueeze(0).expand(batch_size, segment_len).to(xlnet_segment.device)
        segment_mask = batch_arrange <= max_segment_len
        sentence_mask = segment_mask & (1 - xlnet_segment).byte()
        sentence_lens = sentence_mask.long().sum(dim=1, keepdim=True)

        # sentence encode layer
        max_len = sentence_lens.max().item()
        sentence = encoder_layer[:, 0: max_len].contiguous()
        sentence_mask = sentence_mask[:, 0: max_len].contiguous()
        sentence = sentence.masked_fill(sentence_mask.unsqueeze(-1) == 0, 0)
        sentence = self.sentence_transform(sentence)

        return sentence


class XlnetCapsuleNetwork(nn.Module):

    def __init__(self, xlnet, xlnet_size, capsule_size, dropout, num_categories):
        super(XlnetCapsuleNetwork, self).__init__()
        self.xlnet = xlnet
        self.xlnet_size = xlnet_size
        self.capsule_size = capsule_size
        self.aspect_transform = nn.Sequential(
            nn.Linear(xlnet_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.sentence_transform = nn.Sequential(
            nn.Linear(xlnet_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.norm_attention = BilinearAttention(capsule_size, capsule_size)
        self.guide_capsule = nn.Parameter(
            torch.Tensor(num_categories, capsule_size)
        )
        self.guide_weight = nn.Parameter(
            torch.Tensor(capsule_size, capsule_size)
        )
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(xlnet_size, xlnet_size * num_categories)
        self.dropout = dropout
        self.num_categories = num_categories
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.guide_capsule)
        nn.init.xavier_uniform_(self.guide_weight)

    def load_sentiment(self, path):
        sentiment = np.load(path)
        e1 = np.mean(sentiment)
        d1 = np.std(sentiment)
        e2 = 0
        d2 = np.sqrt(2.0 / (sentiment.shape[0] + sentiment.shape[1]))
        sentiment = (sentiment - e1) / d1 * d2 + e2
        self.guide_capsule.data.copy_(torch.tensor(sentiment))

    def forward(self, xlnet_token, xlnet_segment):
        # xlnet encoding
        encoder_layer = self.xlnet(input_ids=xlnet_token, token_type_ids=xlnet_segment)[0]  # [b,l,dim]
        batch_size, segment_len = xlnet_segment.size()
        max_segment_len = xlnet_segment.argmax(dim=-1, keepdim=True)
        batch_arrange = torch.arange(segment_len).unsqueeze(0).expand(batch_size, segment_len).to(xlnet_segment.device)
        segment_mask = batch_arrange <= max_segment_len
        sentence_mask = segment_mask & (1 - xlnet_segment).byte()
        aspect_mask = xlnet_segment
        sentence_lens = sentence_mask.long().sum(dim=1, keepdim=True)
        # aspect average pooling
        aspect_lens = aspect_mask.long().sum(dim=1, keepdim=True)
        aspect = encoder_layer.masked_fill(aspect_mask.unsqueeze(-1) == 0, 0)
        aspect = aspect.sum(dim=1, keepdim=False) / aspect_lens.float()
        # sentence encode layer
        max_len = sentence_lens.max().item()
        sentence = encoder_layer[:, 0: max_len].contiguous()
        sentence_mask = sentence_mask[:, 0: max_len].contiguous()
        sentence = sentence.masked_fill(sentence_mask.unsqueeze(-1) == 0, 0)
        # primary capsule layer
        sentence = self.sentence_transform(sentence)
        primary_capsule = squash(sentence, dim=-1)
        aspect = self.aspect_transform(aspect)
        aspect_capsule = squash(aspect, dim=-1)
        # aspect aware normalization
        norm_weight = self.norm_attention.get_attention_weights(aspect_capsule, primary_capsule, sentence_mask)
        # capsule guided routing
        category_capsule = self._capsule_guided_routing(primary_capsule, norm_weight)
        category_capsule_norm = torch.sqrt(torch.sum(category_capsule * category_capsule, dim=-1, keepdim=False))
        return category_capsule_norm

    def _capsule_guided_routing(self, primary_capsule, norm_weight):
        guide_capsule = squash(self.guide_capsule)
        guide_matrix = primary_capsule.matmul(self.guide_weight).matmul(guide_capsule.transpose(0, 1))
        guide_matrix = F.softmax(guide_matrix, dim=-1)
        guide_matrix = guide_matrix * norm_weight.unsqueeze(-1) * self.scale  # (batch_size, time_step, num_categories)
        category_capsule = guide_matrix.transpose(1, 2).matmul(primary_capsule)
        category_capsule = F.dropout(category_capsule, p=self.dropout, training=self.training)
        category_capsule = squash(category_capsule)
        return category_capsule


class XlnetCNNNetwork(nn.Module):
    def __init__(self, xlnet, xlnet_size, dropout, cnn_size, filter_size, filter_nums, num_categories):
        super(XlnetCNNNetwork, self).__init__()
        self.xlnet = XlnetEncodingNetwork(xlnet, xlnet_size, cnn_size, dropout)
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

    @staticmethod
    def pooling(conv, x):
        x = conv(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        return x

    def forward(self, xlnet_token, xlnet_segment):
        xlnetout = self.xlnet(xlnet_token, xlnet_segment)  # [b,l,dim]
        xlnetout = xlnetout.transpose(1, 2)
        out = [conv(xlnetout) for conv in self.convs]
        out = [F.max_pool1d(o, kernel_size=o.size(-1)).squeeze(-1) for o in out]  # [B,D,L] -> [B,O,L]*len -> [B,O,1]*len -> [B,O]*len
        out = torch.cat(out, -1)  # [B, len(k)*O]
        # out = self.cnnDrop(out)
        out = self.linear(out)
        # out = torch.softmax(out, -1)
        return out


class XlnetFcNetwork(nn.Module):

    def __init__(self, xlnet, xlnet_size, dropout, hidden_size, num_categories):
        super(XlnetFcNetwork, self).__init__()
        self.xlnet = xlnet
        self.xlnet_size = xlnet_size
        self.hidden_size = hidden_size
        self.num_categories = num_categories
        self.sentence_transform = nn.Sequential(
            nn.Linear(xlnet_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.linear = nn.Linear(self.hidden_size, self.num_categories, bias=False)
    # 0.首先写出一个test xml版本 1.怎么变成last_hidden_size 2.token type ids 0/1/2 3.attention mask
    def forward(self, xlnet_token, xlnet_segment):
        output = self.xlnet(input_ids=xlnet_token, token_type_ids=xlnet_segment)
        out = self.sentence_transform(output[0][:, -1])
        out = self.linear(out)
        # out = torch.softmax(out, -1)
        return out

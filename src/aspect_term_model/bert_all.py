import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


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


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        # 双向LSTM，也就是序列从左往右算一次，从右往左又算一次，这样就可以两倍的输出 bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, num_layers=layers, bidirectional=True)
        self.lstm_drop = nn.Dropout(dropout)
        self.init_weight()

    def init_weight(self):
        stdv = math.sqrt(2.0 / (self.input_dim * 2))
        stdv_1 = math.sqrt(2.0 / (self.input_dim * 3))
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, val=0)
            elif 'weight' in name:
                if 'l0' in name:
                    nn.init.normal_(param, 0, stdv)
                else:
                    nn.init.normal_(param, 0, stdv_1)

    def forward(self, inputs, bert_segment):
        """
        :param inputs: Char Embedding
        :param seq_lengths:
        :return:
        """
        max_segment_len = bert_segment.argmax(dim=-1).view(-1)
        seq_lengths, idx1 = torch.sort(max_segment_len, descending=True)
        _, idx2 = torch.sort(idx1)
        inputs = inputs.index_select(0, idx1)

        pack_input = pack_padded_sequence(inputs, seq_lengths, True)
        lstm_out, (h_n, c_n) = self.lstm(pack_input, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, True)  # [bs, n, 2u]

        h_n = h_n.permute(1, 0, 2)
        # lstm_out = h_n.index_select(0, idx2)
        a = seq_lengths.index_select(0, idx2)

        # lstm_out = self.lstm_drop(h_n)  # [bs, n, d]
        return lstm_out


class SelfAttentive(nn.Module):
    def __init__(self, hidden_size, attn_hops=1, mlp_d=350):
        super(SelfAttentive, self).__init__()
        r = attn_hops
        d = mlp_d
        initrange = 0.1
        self.Ws1 = nn.Parameter(torch.Tensor(1, d, hidden_size).uniform_(-initrange, initrange))
        self.Ws2 = nn.Parameter(torch.Tensor(1, r, d).uniform_(-initrange, initrange))
        self.fc1 = nn.Linear(r * hidden_size, mlp_d)

    def forward(self, H):
        """
        A Structured Self-attentive Sentence Embedding
        :param H: rnn out (bs, n, 2u)
        :return: out: (bs, mlp_hidden)  A: (bs, r, n)
        """
        bs, n, _ = H.size() # batch size, sentence length, dim=2u+d
        H_T = torch.transpose(H, 2, 1).contiguous()  # (bs, n, 2u) -> (bs, 2u, n)
        A = torch.tanh(torch.bmm(self.Ws1.repeat(bs, 1, 1), H_T))  # (bs, d, n)
        A = torch.bmm(self.Ws2.repeat(bs, 1, 1), A)  # (bs, r, n)
        A = F.softmax(A.view(-1, n), -1).view(bs, -1, n)  # (bs, r, n)
        M = torch.bmm(A, H)  # (bs, r, 2u)
        out = F.relu(self.fc1(M.view(bs, -1)))  # (bs, mlp_hidden)
        # out = M.view(bs, -1)
        return out, A


class BertAttnNetwork(nn.Module):
    def __init__(self, bert, bert_size, hidden_size, attn_hops, mlp_d, dropout, num_categories):
        super(BertAttnNetwork, self).__init__()
        self.bert = BertEncodingNetwork(bert, bert_size, hidden_size, dropout)
        self.attn = SelfAttentive(hidden_size, attn_hops, mlp_d)
        self.linear = nn.Linear(mlp_d, num_categories)

    def forward(self, bert_token, bert_segment):
        bertout = self.bert(bert_token, bert_segment)  # [b,l,dim]
        out, _ = self.attn(bertout)
        out = self.linear(out)
        return out

    def get_weight(self, bert_token, bert_segment):
        bertout = self.bert(bert_token, bert_segment)  # [b,l,dim]
        out, weight = self.attn(bertout)
        return weight


class BertRNNNetwork(nn.Module):
    def __init__(self, bert, bert_size, hidden_size, hidden_dim, layers, lstm_dropout, dropout, num_categories):
        super(BertRNNNetwork, self).__init__()
        self.bert = BertEncodingNetwork(bert, bert_size, hidden_size, dropout)
        self.bilstm = BiLSTM(hidden_size, hidden_dim, layers, lstm_dropout)
        self.linear = nn.Linear(hidden_dim, num_categories)

    def forward(self, bert_token, bert_segment):
        bertout = self.bert(bert_token, bert_segment)  # [b,l,dim]
        out = self.bilstm(bertout, bert_segment)
        out = self.linear(out)
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

import torch
from torch import nn
import numpy as np
import os
import yaml
from pytorch_pretrained_bert import BertModel
from transformers import XLNetModel
from src.aspect_category_model.recurrent_capsnet import RecurrentCapsuleNetwork
from src.aspect_category_model.bert_capsnet import BertCapsuleNetwork
from src.aspect_category_model.bert_all import BertCNNNetwork, BertFcNetwork
from src.aspect_category_model.xlnet_all import XlnetFcNetwork, XlnetCNNNetwork, XlnetCapsuleNetwork


def make_model(config):
    model_type = config['aspect_category_model']['type']
    if model_type == 'recurrent_capsnet':
        return make_recurrent_capsule_network(config)
    elif 'bert' in model_type:
        return make_bert_all_network(config)
    elif 'xlnet' in model_type:
        return make_xlnet_all_network(config)

def make_xlnet_all_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    model_type = config['aspect_category_model']['type']
    config = config['aspect_category_model'][model_type]
    xlnet_path = '/home/xuwd/data/xlnet_cased_L-12_H-768_A-12/'
    xlnet = XLNetModel.from_pretrained(xlnet_path)
    if model_type == 'xlnet_fc':
        model = XlnetFcNetwork(
            xlnet=xlnet,
            xlnet_size=config['xlnet_size'],
            dropout=config['dropout'],
            hidden_size=config['hidden_size'],
            num_categories=log['num_categories']
        )
    elif model_type == 'xlnet_cnn':
        model = XlnetCNNNetwork(
            xlnet=xlnet,
            xlnet_size=config['xlnet_size'],
            dropout=config['dropout'],
            cnn_size=config['cnn_size'],
            filter_size=config['filter_size'],
            filter_nums=config['filter_nums'],
            num_categories=log['num_categories']
        )
    elif model_type == 'xlnet_capsule':
        model = XlnetCapsuleNetwork(
            xlnet=xlnet,
            xlnet_size=config['xlnet_size'],
            capsule_size=config['capsule_size'],
            dropout=config['dropout'],
            num_categories=log['num_categories']
        )
    return model


def make_bert_all_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    model_type = config['aspect_category_model']['type']
    config = config['aspect_category_model'][model_type]
    bert_path = '/home/xuwd/data/bert-base-uncased/'
    bert = BertModel.from_pretrained(bert_path)
    if model_type == 'bert_capsnet':
        model = BertCapsuleNetwork(
            bert=bert,
            bert_size=config['bert_size'],
            capsule_size=config['capsule_size'],
            dropout=config['dropout'],
            num_categories=log['num_categories']
        )
        model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix1.npy'))
    elif model_type == 'bert_cnn':
        model = BertCNNNetwork(
            bert=bert,
            bert_size=config['bert_size'],
            dropout=config['dropout'],
            cnn_size=config['cnn_size'],
            filter_size=config['filter_size'],
            filter_nums=config['filter_nums'],
            num_categories=log['num_categories']
        )
    elif model_type == 'bert_fc':
        model = BertFcNetwork(
            bert=bert,
            bert_size=config['bert_size'],
            dropout=config['dropout'],
            cnn_size=config['cnn_size'],
            num_categories=log['num_categories']
        )

    return model

'''
def make_model(config):
    model_type = config['aspect_category_model']['type']
    if model_type == 'recurrent_capsnet':
        return make_recurrent_capsule_network(config)
    elif model_type == 'bert_capsnet':
        return make_bert_capsule_network(config)
    elif 'bert' in model_type:
        return make_bert_all_network(config)

def make_bert_all_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    model_type = config['aspect_category_model']['type']
    config = config['aspect_category_model'][model_type]
    bert_path = '/home/xuwd/data/bert-base-uncased/'
    bert = BertModel.from_pretrained(bert_path)
    if model_type == 'bert_cnn':
        model = BertCNNNetwork(
            bert=bert,
            bert_size=config['bert_size'],
            dropout=config['dropout'],
            cnn_size=config['cnn_size'],
            filter_size=config['filter_size'],
            filter_nums=config['filter_nums'],
            num_categories=log['num_categories']
        )
    elif model_type == 'bert_fc':
        model = BertFcNetwork(
            bert=bert,
            bert_size=config['bert_size'],
            dropout=config['dropout'],
            cnn_size=config['cnn_size'],
            num_categories=log['num_categories']
        )
    return model
'''

def make_bert_fc_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    bert_path = '/home/xuwd/data/bert-base-uncased/'
    bert = BertModel.from_pretrained(bert_path)
    model = BertFcNetwork(
        bert=bert,
        bert_size=config['bert_size'],
        dropout=config['dropout'],
        cnn_size=config['cnn_size'],
        num_categories=log['num_categories']
    )
    # model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model


def make_bert_cnn_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    print(config)
    bert_path = '/home/xuwd/data/bert-base-uncased/'
    bert = BertModel.from_pretrained(bert_path)
    print("hrere")
    model = BertCNNNetwork(
        bert=bert,
        bert_size=config['bert_size'],
        dropout=config['dropout'],
        cnn_size=config['cnn_size'],
        filter_size=config['filter_size'],
        filter_nums=config['filter_nums'],
        num_categories=log['num_categories']
    )
    # model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model

def make_bert_capsule_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BertCapsuleNetwork(
        bert=bert,
        bert_size=config['bert_size'],
        capsule_size=config['capsule_size'],
        dropout=config['dropout'],
        num_categories=log['num_categories']
    )
    model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model

def make_recurrent_capsule_network(config):
    embedding = make_embedding(config)
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    aspect_embedding = nn.Embedding(num_embeddings=8, embedding_dim=config['embed_size'])
    model = RecurrentCapsuleNetwork(
        embedding=embedding,
        aspect_embedding=aspect_embedding,
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        capsule_size=config['capsule_size'],
        dropout=config['dropout'],
        num_categories=log['num_categories']
    )
    model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model

def make_embedding(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    vocab_size = log['vocab_size']
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    embed_size = config['embed_size']
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    glove = np.load(os.path.join(base_path, 'processed/glove.npy'))
    embedding.weight.data.copy_(torch.tensor(glove))
    return embedding
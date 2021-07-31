import os
from torch.utils.data import DataLoader
from data_process.dataset import ABSADataset

input_list = {
    'base': ['context', 'aspect'],
    'bert': ['bert_token', 'bert_segment'],
    'xlnet': ['xlnet_token', 'xlnet_segment'],
    'test': ['bert_token', 'bert_segment', 'xlnet_token', 'xlnet_segment']
}


def make_term_data(config):
    base_path = config['base_path']
    train_path = os.path.join(base_path, 'processed/train_xlnet.npz')
    print('train path:', train_path)
    val_path = os.path.join(base_path, 'processed/val_xlnet.npz')
    print('val path:', val_path)
    key = 'base'
    if 'bert' in config['aspect_term_model']['type']:
        key = 'bert'
    elif 'xlnet' in config['aspect_term_model']['type']:
        key = 'xlnet'
    train_data = ABSADataset(train_path, input_list[key])
    val_data = ABSADataset(val_path, input_list[key])
    print(train_path)
    config = config['aspect_term_model'][config['aspect_term_model']['type']]
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader


def make_term_test_data(config, key, batch_size):
    base_path = config['base_path']
    test_path = os.path.join(base_path, 'processed/test_xlnet.npz')
    test_data = ABSADataset(test_path, input_list[key])
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    return test_loader

def make_category_data(config):
    base_path = config['base_path']
    train_path = os.path.join(base_path, 'processed/train_xlnet.npz')
    print('train path:', train_path)
    val_path = os.path.join(base_path, 'processed/val_xlnet.npz')
    print('val path:', val_path)
    key = 'base'
    if 'bert' in config['aspect_category_model']['type']:
        key = 'bert'
    elif 'xlnet' in config['aspect_category_model']['type']:
        key = 'xlnet'
    train_data = ABSADataset(train_path, input_list[key])
    val_data = ABSADataset(val_path, input_list[key])
    print(train_path)
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader


def make_category_test_data(config, key, batch_size):
    base_path = config['base_path']
    test_path = os.path.join(base_path, 'processed/test.npz')
    test_data = ABSADataset(test_path, input_list[key])
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    return test_loader

def make_category_test_data1(config):
    model_type = config['aspect_category_model']['type']
    if 'bert' in model_type:
        i_list = ['bert_token', 'bert_segment']
    else:
        i_list = ['sentence', 'aspect']
    base_path = config['base_path']
    test_path = os.path.join(base_path, 'processed/test.npz')
    test_data = ABSADataset(test_path, i_list)
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    return test_loader

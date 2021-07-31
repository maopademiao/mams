import torch
from torch import nn
from torch import optim
from train import make_aspect_term_model, make_aspect_category_model
from train.make_data import make_term_data, make_category_data
from train.make_optimizer import make_optimizer
from train.eval import eval, eval_f1
import os
import time
import pickle
from src.module.utils.loss import CapsuleLoss


def train(config):
    mode = config['mode']
    print('mode:', mode)
    print('type:', config['aspect_' + mode + '_model']['type'])
    if mode == 'term':
        model = make_aspect_term_model.make_model(config)
        train_loader, val_loader = make_term_data(config)
    else:
        model = make_aspect_category_model.make_model(config)
        train_loader, val_loader = make_category_data(config)
    model = model.cuda()
    base_path = config['base_path']
    model_path = os.path.join(base_path, 'checkpoints/%s_%s.pth' % (config['aspect_' + mode + '_model']['type'], config['aspect_' + mode + '_model']['save']))
    print('model save path:', model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    criterion = CapsuleLoss()
    optimizer = make_optimizer(config, model)
    max_val_accuracy = 0
    min_val_loss = 100
    best_f1 = -1
    global_matrix = None
    best_epoch, best_step = 0, 0
    global_step = 0
    config = config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]
    for epoch in range(config['num_epoches']):
        total_loss = 0
        total_samples = 0
        correct_samples = 0
        start = time.time()
        for i, data in enumerate(train_loader):
            global_step += 1
            model.train()
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            optimizer.zero_grad()
            logit = model(input0, input1)
            loss = criterion(logit, label)
            batch_size = input0.size(0)
            total_loss += batch_size * loss.item()
            total_samples += batch_size
            pred = logit.argmax(dim=1)
            correct_samples += (label == pred).long().sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if i % 10 == 0 and i > 0 or i == len(train_loader) - 1:
                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                # total_loss = 0
                # total_samples = 0
                # correct_samples = 0
                val_accuracy, f1, matrix, val_loss = eval_f1(model, val_loader, criterion)
                print('[epoch %2d] [step %3d] train_loss: %.4f train_acc: %.4f val_loss: %.4f val_acc: %.4f f1: %.4f'
                      % (epoch, i, train_loss, train_accuracy, val_loss, val_accuracy, f1))
                if f1 > best_f1:
                    best_f1 = f1
                    global_matrix = matrix
                    best_epoch = epoch
                    best_step = i
                    max_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), model_path)
                # if val_accuracy > max_val_accuracy:
                #     max_val_accuracy = val_accuracy
                #     torch.save(model.state_dict(), model_path)
                # if val_loss < min_val_loss:
                #     min_val_loss = val_loss
                #     if epoch > 0:
                #         torch.save(model.state_dict(), model_path)
            #break
       # break
        print('positive -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(global_matrix[0]))
        print('negative -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(global_matrix[1]))
        print('neutral  -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(global_matrix[2]))
        print('best_epoch: %d  best_step: %d  best_f1: %.4f' % (best_epoch, best_step, best_f1))
        end = time.time()
        print('time: %.4fs' % (end - start))
    print('max_val_accuracy:', max_val_accuracy)

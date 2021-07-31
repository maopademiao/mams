import torch
import torch.nn.functional as F
import os
from train import make_aspect_term_model, make_aspect_category_model
from train.make_data import make_term_test_data, make_category_test_data
import codecs
import json
import xml.etree.ElementTree as et
import pickle

# positive -- correct: 661  predict: 767  real: 803  f1: 0.8420
# negative -- correct: 555  predict: 686  real: 654  f1: 0.8284
# neutral  -- correct: 1060 predict: 1215 real: 1211 f1: 0.8739
# best_f1: 0.8481
# max_val_accuracy: 0.8530734632683659

# positive -- correct: 661  predict: 773  real: 803  f1: 0.8388
# negative -- correct: 552  predict: 684  real: 654  f1: 0.8251
# neutral  -- correct: 1060 predict: 1211 real: 1211 f1: 0.8753
# best_f1: 0.8464
# max_val_accuracy: 0.8519490254872564

# positive -- correct: 662  predict: 771  real: 803  f1: 0.8412
# negative -- correct: 552  predict: 682  real: 654  f1: 0.8263
# neutral  -- correct: 1060 predict: 1215 real: 1211 f1: 0.8739
# best_f1: 0.8471
# max_val_accuracy: 0.8523238380809596


def test(config):
    bert_model_path = [
        'bert_capsnet_all.pth',
        'bert_cnn_all.pth',
        'bert_fc_all.pth'
     #   'bert_attn_all.pth'
    ]
    xlnet_model_path = [
        #'xlnet_capsnet_all.pth',
        'xlnet_cnn_all.pth',
        'xlnet_fc_all.pth'
    ]
    flag = 1
    models = []
    if config['mode'] == 'term':
        make_aspect_model = make_aspect_term_model
        test_loader = make_term_test_data(config, 'test', 32)
    else:
        make_aspect_model = make_aspect_category_model
        test_loader = make_category_test_data(config, 'test', 32)
    for m in bert_model_path + xlnet_model_path:
        if 'bert_capsnet' in m:
            config['aspect_term_model']['type'] = 'bert_capsnet'
        elif 'bert_cnn' in m:
            config['aspect_term_model']['type'] = 'bert_cnn'
        elif 'bert_fc' in m:
            config['aspect_term_model']['type'] = 'bert_fc'
    #    elif 'bert_attn' in m:
      #      config['aspect_term_model']['type'] = 'bert_attn'
        elif 'xlnet_fc' in m:
            config['aspect_term_model']['type'] = 'xlnet_fc'
        elif 'xlnet_cnn' in m:
            config['aspect_term_model']['type'] = 'xlnet_cnn'
      #  elif 'xlnet_capsnet' in m:
        #    config['aspect_term_model']['type'] = 'xlnet_capsnet'
        model = make_aspect_model.make_model(config)
        model = model.cuda()
        model_path = os.path.join(config['base_path'], 'checkpoints/%s' % m)
        model.load_state_dict(torch.load(model_path))
        print(model_path, 'load successfully!')
        models.append(model)

    # with open('temp.pkl', 'wb') as f:
    #     pickle.dump(models, f)
    # with open('temp.pkl', 'rb') as f:
    #     models = pickle.load(f)

    test_ensemble(models, test_loader, len(bert_model_path), config)
    # test_vote(models, test_loader, len(bert_model_path), config['result_path'])


def test_ensemble(models, test_loader, bert_len, config):
    total_samples = 0
    correct_samples = 0
    pred_all = None
    label_all = None
    for model in models:
        model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 10 == 0:
                print(i)
            input0, input1, input2, input3, label = data
            input0, input1, input2, input3, label = \
                input0.cuda(), input1.cuda(), input2.cuda(), input3.cuda(), label.cuda()
            for j, model in enumerate(models):
                if j < bert_len:
                    x0, x1 = input0, input1
                else:
                    x0, x1 = input2, input3
                if j == 0:
                    logit = F.softmax(model(x0, x1), dim=-1)
                else:
                    logit += F.softmax(model(x0, x1), dim=-1)
                # if j == 0:
                #     logit = model(x0, x1)
                # else:
                #     logit += model(x0, x1)
            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)
            correct_samples += (label == pred).long().sum().item()
            if pred_all is None:
                pred_all = pred
            else:
                pred_all = torch.cat([pred_all, pred], -1)
            if label_all is None:
                label_all = label
            else:
                label_all = torch.cat([label_all, label], -1)
    result = {
        'pred': pred_all.cpu().numpy().tolist(),
        'real': label_all.cpu().numpy().tolist()
    }
    if -1 in result['real']:
        # with codecs.open('./task2/ATSA/test.json', 'w', encoding='utf-8') as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        print_result(config['test_path'], result['pred'], config['result_path'])
    else:
        accuracy = correct_samples / total_samples
        correct_mask = (pred_all == label_all).long()
        correct_all = (label_all + 1).mul(correct_mask)
        matrix = []
        for i in range(3):
            c = torch.sum(torch.eq(correct_all, i + 1)).item()
            p = torch.sum(torch.eq(pred_all, i)).item()
            l = torch.sum(torch.eq(label_all, i)).item()
            f1 = c * 2 / (p + l)
            matrix.append([c, p, l, f1])
        assert sum([i[0] for i in matrix]) == correct_samples
        f1 = sum([i[3] for i in matrix]) / 3
        print(matrix)
        print(
            'positive -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[0]))
        print(
            'negative -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[1]))
        print(
            'neutral  -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[2]))
        print('best_f1: %.4f' % f1)
        print('max_val_accuracy:', accuracy)
        return accuracy, f1, matrix


def test_vote(models, test_loader, bert_len, config):
    pred_all = None
    label_all = None
    logit_all = None
    for model in models:
        model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 10 == 0:
                print(i)
            input0, input1, input2, input3, label = data
            input0, input1, input2, input3, label = \
                input0.cuda(), input1.cuda(), input2.cuda(), input3.cuda(), label.cuda()
            for j, model in enumerate(models):
                if j < bert_len:
                    x0, x1 = input0, input1
                else:
                    x0, x1 = input2, input3
                m = model(x0, x1)
                if j == 0:
                    logit = m
                    # logit = F.softmax(m, dim=-1)
                    pred = m.argmax(dim=1).view(-1, 1)
                else:
                    logit += m
                    # logit += F.softmax(m, dim=-1)
                    pred = torch.cat([pred, m.argmax(dim=1).view(-1, 1)], -1)
            if pred_all is None:
                pred_all = pred
            else:
                pred_all = torch.cat([pred_all, pred], 0)
            if label_all is None:
                label_all = label
            else:
                label_all = torch.cat([label_all, label], 0)
            if logit_all is None:
                logit_all = logit
            else:
                logit_all = torch.cat([logit_all, logit], 0)
    pred_new_all = []
    for pred, logit in zip(pred_all, logit_all):
        num = len(models) // 2
        if torch.sum(torch.eq(pred, 0)).item() > num:
            p = 0
        elif torch.sum(torch.eq(pred, 1)).item() > num:
            p = 1
        elif torch.sum(torch.eq(pred, 2)).item() > num:
            p = 2
        else:
            p = logit.argmax(-1).item()
        pred_new_all.append(p)
    result = {
        'pred': pred_new_all,
        'real': label_all.cpu().numpy().tolist()
    }
    if -1 in result['real']:
        # with codecs.open('./task2/ATSA/test.json', 'w', encoding='utf-8') as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        print_result(config['test_path'], result['pred'], config['result_path'])
    else:
        pred_all = torch.LongTensor(pred_new_all).cuda()
        total_samples = pred_all.size(0)
        correct_samples = (label_all == pred_all).long().sum().item()
        accuracy = correct_samples / total_samples
        correct_mask = (pred_all == label_all).long()
        correct_all = (label_all + 1).mul(correct_mask)
        matrix = []
        for i in range(3):
            c = torch.sum(torch.eq(correct_all, i + 1)).item()
            p = torch.sum(torch.eq(pred_all, i)).item()
            l = torch.sum(torch.eq(label_all, i)).item()
            f1 = c * 2 / (p + l)
            matrix.append([c, p, l, f1])
        assert sum([i[0] for i in matrix]) == correct_samples
        f1 = sum([i[3] for i in matrix]) / 3
        print(matrix)
        print(
            'positive -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[0]))
        print(
            'negative -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[1]))
        print(
            'neutral  -- correct: {0[0]:<4d} predict: {0[1]:<4d} real: {0[2]:<4d} f1: {0[3]:.4f}'.format(matrix[2]))
        print('best_f1: %.4f' % f1)
        print('max_val_accuracy:', accuracy)
        return accuracy, f1, matrix


def print_result(data_path, preds, out_path):
    index2p = {
        0: 'positive',
        1: 'negative',
        2: 'neutral',
    }
    if isinstance(preds, str):
        preds = json.load(open(preds))['pred']
    tree = et.parse(data_path)
    sentences = tree.getroot()
    i = 0
    for sentence in sentences:
        aspectTerms = sentence.find('aspectCategories')
        for aspectTerm in aspectTerms:
            aspectTerm.set('pred', index2p[preds[i]])
            i += 1
    assert i == len(preds)
    tree.write(out_path)


if __name__ == '__main__':
    print("oooo")
    preds = json.load(open('../task2/ACSA/result.json'))['pred']
    print_result('../task2/ACSA/dev.xml', preds)

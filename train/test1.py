import torch
import os
from train import make_aspect_term_model, make_aspect_category_model
from train.make_data import make_term_test_data, make_category_test_data
from train.eval import eval
import codecs
import json
import xml.etree.ElementTree as et

def test(config):
    mode = config['mode']
    if mode == 'term':
        model = make_aspect_term_model.make_model(config)
    else:
        model = make_aspect_category_model.make_model(config)
    model = model.cuda()
    model_path = os.path.join(config['base_path'], 'checkpoints/%s.pth' % config['aspect_' + mode + '_model']['type'])
    model.load_state_dict(torch.load(model_path))
    if mode == 'term':
        test_loader = make_term_test_data(config)
    else:
        test_loader = make_category_test_data(config)
    test_accuracy = eval(model, test_loader)
    print('test:\taccuracy: %.4f' % (test_accuracy))


def test_ensemble(config):
    path = '/home/xuwd/projects/absa/all/MAMS-for-ABSA-master/task2/ACSA/checkpoints'
    model_path = [
        'xlnet_cnn_all.pth'
        #'bert_capsnet.pth',
        #'bert_cnn_all.pth',
        #'bert_fc_all.pth',
        #'bert_attn_all.pth',
    ]
    flag = 4
    models = []
    if config['mode'] == 'term':
        make_aspect_model = make_aspect_term_data
        test_loader = make_term_test_data(config, 'test', 32)
    else:
        make_aspect_model = make_aspect_category_model
        test_loader = make_category_test_data(config, 'test', 32)
    for m in model_path:
        if 'bert_capsnet' in m:
            config['aspect_term_model']['type'] = 'bert_capsnet'
        elif 'bert_cnn' in m:
            config['aspect_term_model']['type'] = 'bert_cnn'
        elif 'bert_fc' in m:
            config['aspect_term_model']['type'] = 'bert_fc'
        elif 'bert_attn' in m:
            config['aspect_term_model']['type'] = 'bert_attn'
        elif 'xlnet_fc' in m:
            config['aspect_term_model']['type'] = 'xlnet_fc'
        elif 'xlnet_cnn' in m:
            config['aspect_term_model']['type'] = 'xlnet_cnn'
        model = make_aspect_term_model.make_model(config)
        model = model.cuda()
        model_path = os.path.join(config['base_path'], 'checkpoints/%s' % m)
        model.load_state_dict(torch.load(model_path))
        print(model_path, 'load successfully!')
        models.append(model)

    total_samples = 0
    correct_samples = 0
    total_loss = 0
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
                if j < flag:
                    x0, x1 = input0, input1
                else:
                    x0, x1 = input2, input3
                if j == 0:
                    logit = model(x0, x1)
                else:
                    logit += model(x0, x1)
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
    if -1 in result['pred']:
        # with codecs.open('./task2/ATSA/test.json', 'w', encoding='utf-8') as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        print_result(config['result_path'], result['pred'])
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

# 0.8441286372373226
# 0.848575712143928


def print_result(data_path, preds):
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
        aspectTerms = sentence.find('aspectTerms')
        for aspectTerm in aspectTerms:
            aspectTerm.set('pred', index2p[preds[i]])
            i += 1
    assert i == len(preds)
    tree.write('../task2/ATSA/output.xml')


if __name__ == '__main__':
    preds = json.load(open('../task2/ATSA/result.json'))['pred']
    print_result('../task2/ATSA/dev.xml', preds)

import torch

def eval(model, data_loader, criterion=None):
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            logit = model(input0, input1)
            loss = criterion(logit, label).item() if criterion is not None else 0
            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)
            correct_samples += (label == pred).long().sum().item()
            total_loss += loss * input0.size(0)
    accuracy = correct_samples / total_samples
    avg_loss = total_loss / total_samples
    if criterion is not None:
        return accuracy, avg_loss
    else:
        return accuracy


def eval_f1(model, data_loader, criterion=None):
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    pred_all = None
    label_all = None
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            logit = model(input0, input1)
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
            if criterion:
                loss = criterion(logit, label).item()
                total_loss += loss * input0.size(0)
    accuracy = correct_samples / total_samples
    avg_loss = total_loss / total_samples
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
    if criterion is not None:
        return accuracy, f1, matrix, avg_loss
    else:
        return accuracy, f1, matrix

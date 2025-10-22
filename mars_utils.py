import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pylab
import matplotlib.pyplot as plt
import os


# calculate the accuracy that the true label is in y_true's top k rank
# k: list of int. each <= num_cls
# y_pred: np array of probablities. (batch_size * cls_num) (output of softmax)
# y_true: batch_size * 1.
# return: list of acc given list of k
def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(get_X, log_interval, model, device, train_loader, optimizer, loss_func, metric_topk, epoch, para,l):
    # set model as training mode
    model.train()

    losses = []
    scores = [[]] * len(metric_topk)
    N_count = 0  # counting total trained sample in one epoch
    los_v_save = []
    los_a_save = []
    wei_v_save = []
    wei_a_save = []
    for batch_idx, sample in enumerate(train_loader):
        # distribute data to device
        X, n = get_X(device, sample)
        y = sample["emotion"].to(device).squeeze()
        output,loss, loss_v, loss_a, weight_v, weight_a = model(X, y, epoch, para, l)
        los_v_save.extend(loss_v)
        los_a_save.extend(loss_a)
        wei_v_save.extend(weight_v)
        wei_a_save.extend(weight_a)

        N_count += n
        optimizer.zero_grad()

        # loss = loss_func(output, y)
        losses.append(loss.item())

        step_score = accuracy_topk(output, y, topk=metric_topk)
        for i, ss in enumerate(step_score):
            scores[i].append(int(ss))

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
            for i, each_k in enumerate(metric_topk):
                print("Top {} accuracy: {:.2f}%".format(each_k, float(step_score[i])))

    return losses, scores, los_v_save, los_a_save, wei_v_save, wei_a_save


def validation(get_X, model, device, loss_func, val_loader, metric_topk, show_cm=False,epoch = 0):
    # set model as testing mode
    model.eval()

    test_loss = []
    all_y = []
    all_y_pred = []
    all_v_pred = []
    all_a_pred = []

    with torch.no_grad():
        for sample in val_loader:
            # distribute data to device
            X, _ = get_X(device, sample)
            y = sample["emotion"].to(device).squeeze()
            output_v,output_a,loss = model(X,y,epoch,0,0)

            # loss = loss_func(output, y)
            test_loss.append(loss.item())  # sum up batch loss

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_v_pred.extend(output_v)
            all_a_pred.extend(output_a)
            all_y_pred.extend(output_v+output_a)

    test_loss = np.mean(test_loss)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    all_v_pred = torch.stack(all_v_pred, dim=0)
    all_a_pred = torch.stack(all_a_pred, dim=0)
    test_score = [float(t_acc) for t_acc in accuracy_topk(all_y_pred, all_y, topk=metric_topk)]
    v_score = [float(t_acc) for t_acc in accuracy_topk(all_v_pred, all_y, topk=metric_topk)]
    a_score = [float(t_acc) for t_acc in accuracy_topk(all_a_pred, all_y, topk=metric_topk)]

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}'.format(len(all_y), test_loss))
    for i, each_k in enumerate(metric_topk):
        print("Top {} accuracy:, mean {}， v {}, a {}".format(each_k, test_score[i], v_score[i], a_score[i]))
    print("\n")
    all_y_pred = torch.squeeze(all_y_pred.data.max(1, keepdim=True)[1])
    all_v_pred = torch.squeeze(all_v_pred.data.max(1, keepdim=True)[1])
    all_a_pred = torch.squeeze(all_a_pred.data.max(1, keepdim=True)[1])
    # if show_cm:
    #     cm = confusion_matrix(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    #     print("Confusion matrix")
    #     print(cm)

    return test_loss, test_score, v_score, a_score, [all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy(), all_v_pred.cpu().data.squeeze().numpy(),all_a_pred.cpu().data.squeeze().numpy()]

def visualize_accuracy(acc, acc_v,acc_a,title, name):
    x = range(len(acc))

    plt.plot(x, acc, label = 'acc')
    plt.plot(x, acc_v, label='acc_v')
    plt.plot(x, acc_a, label='acc_a')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.grid(True)
    f = os.path.join(name, '{}.png'.format(title))
    plt.savefig(f, dpi=400)
    plt.legend()
    # pylab.show()
    plt.cla()

def visulize_loss(loss_v,loss_a, title, name):
    x = range(len(loss_v))
    loss_v = np.array(loss_v)
    loss_a = np.array(loss_a)
    l_v = np.mean(loss_v,axis=1)
    l_a = np.mean(loss_a, axis=1)

    plt.plot(x, l_v, label = 'loss_v')
    plt.plot(x, l_a, label='loss_a')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.grid(True)
    f = os.path.join(name, '{}.png'.format(title))
    plt.savefig(f, dpi=400)
    plt.legend()
    # pylab.show()
    plt.cla()

def visulize_testloss(loss, title, name):
    x = range(len(loss))

    plt.plot(x, loss, label = 'loss_test')

    plt.xlabel('Epoch')
    plt.ylabel('test_Loss')

    plt.grid(True)
    f = os.path.join(name, '{}.png'.format(title))
    plt.savefig(f, dpi=400)
    plt.legend()
    # pylab.show()

    plt.cla()

import torch
import time
# from advertorch import attacks
from utils import *


def train(prms, device, train_loader, optimizer, epoch):
    prms.model = prms.model.to(device)
    t1 = time.time()
    avg_train_acc = 0
    avg_ce_loss = 0
    ce_loss = torch.nn.CrossEntropyLoss()

    print('Epoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        x_batch, y_batch = data.to(device), target.to(device)

        # plt.figure()
        # plt.imshow(x_batch[31].permute([1,2,0]).cpu(), interpolation='nearest')
        # plt.show()
        # plt.pause(1000)

        x_outputs = prms.model(x_batch)
        x_correct = torch.mean(torch.argmax(x_outputs, dim=1).eq(y_batch).float())
        ce_batch_loss = ce_loss(x_outputs, y_batch)

        prms.model.train()
        optimizer.zero_grad()
        ce_batch_loss.backward()
        optimizer.step()

        avg_train_acc += x_correct.item()
        avg_ce_loss += ce_batch_loss.item()
        # print('[%d] Train acc: %.4f, CE loss: %.3lf' % (batch_idx, x_correct.item(), ce_batch_loss.item()))

    t2 = time.time()

    avg_train_acc /= len(train_loader.dataset)
    avg_ce_loss /= len(train_loader.dataset)
    t = t2 - t1

    return avg_train_acc, avg_ce_loss, t


def test(prms, device, test_loader):
    prms.model = prms.model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    prms.model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        x_batch, y_batch = data.to(device), target.to(device)
        output = prms.model(x_batch)
        test_loss += ce_loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y_batch.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Pred. Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset) , test_loss
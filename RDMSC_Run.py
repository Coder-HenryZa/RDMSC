import sys, os

sys.path.append(os.getcwd())
from utils.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from utils.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils.loadSplitData import *

import matplotlib.pyplot as plt
from datetime import datetime
from model.RDMSC import RDMSC
import argparse
import warnings

warnings.filterwarnings("ignore")


def run_model(treeDic, x_test, x_train, droprate, lr, weight_decay, patience, n_epochs, batchsize, in_feats, hid_feats,
              out_feats, atten_out_dim,
              dataname):
    model = RDMSC(in_feats, hid_feats, out_feats, atten_out_dim).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadUdData(dataname, treeDic, x_train, x_test, droprate)
    for epoch in range(n_epochs):

        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data, Batch_data2 in tqdm_train_loader:
            Batch_data.to(device)
            Batch_data2.to(device)
            out_labels = model(Batch_data, Batch_data2)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []

        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data, Batch_data2 in tqdm_test_loader:
            Batch_data.to(device)
            Batch_data2.to(device)
            val_out = model(Batch_data, Batch_data2)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            temp_val_accs.append(val_acc)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))

        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), model, 'RDMSC', dataname)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    show_val = list(val_accs)

    dt = datetime.now()
    save_time = dt.strftime('%Y_%m_%d_%H_%M_%S')

    fig = plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, color='b', label='train')
    plt.plot(range(1, len(show_val) + 1), show_val, color='r', label='dev')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_accs), step=4))
    fig.savefig('result/' + '{}_accuracy_{}.png'.format(dataname, save_time))

    fig = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='b', label='train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, color='r', label='dev')
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_losses) + 1, step=4))
    fig.savefig('result/' + '{}_loss_{}.png'.format(dataname, save_time))

    return train_losses, val_losses, train_accs, val_accs


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)  # 随机种子，神经网络初始化参数是固定的，而不是变化
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    print("seed:", seed)


parser = argparse.ArgumentParser(description='RDMSC')
parser.add_argument('--lr', default=0.0005, type=float, help='Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay coefficient')
parser.add_argument('--patience', default=10, type=int, help='Early Stopping')
parser.add_argument('--n_epochs', default=200, type=int, help='Training Epochs')
parser.add_argument('--batchsize', default=128, type=int, help='Batch Size')
parser.add_argument('--droprate', default=0.2, type=float, help='Randomly invalidate some edges')
parser.add_argument('--seed', default=11, type=int)
parser.add_argument('--in_feats', default=5000, type=int)
parser.add_argument('--hid_feats', default=64, type=int)
parser.add_argument('--out_feats', default=64, type=int)
parser.add_argument('--atten_out_dim', default=4, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    datasetname = "Twitter15"  # Twitter15 Twitter16
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_set, train_set = loadSplitData(datasetname)
    treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs = run_model(treeDic, test_set, train_set, args.droprate, args.lr,
                                                               args.weight_decay, args.patience, args.n_epochs,
                                                               args.batchsize, args.in_feats, args.hid_feats,
                                                               args.out_feats, args.atten_out_dim, datasetname)
    print("Total_Best_Accuracy:{}".format(max(val_accs)))

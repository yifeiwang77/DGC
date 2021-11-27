import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from utils import load_citation, set_seed

from dgc import dgc_precompute, DGC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use.')
parser.add_argument('--lr', type=float, default=0.156,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=3.544e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj, NormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')                    
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--trials', type=int, default=10, help='Run multiple trails for fair evaluation')
# DGC parameters
parser.add_argument('--T', type=float, default=5.27,
                    help='real-valued diffusion terminal time.')
parser.add_argument('--K', type=int, default=250,
                    help='number of propagation steps (larger K implies better numerical precision).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def main(args):
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

    # preprocessing with all DGC propagation steps
    features, precompute_time = dgc_precompute(features, adj, args.T, args.K)
    print("{:.4f}s".format(precompute_time))

    # initialize model (a linear head)
    model = DGC(features.size(1), labels.max().item()+1)
    model = model.cuda() if args.cuda else model

    # train logistic regression and collect test accuracy
    model, train_time = train(model, features[idx_train], labels[idx_train], args.epochs, args.weight_decay, args.lr)
    acc_test = test(model, features[idx_test], labels[idx_test])

    print("Test accuracy: {:.4f},  pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(acc_test, precompute_time, train_time, precompute_time+train_time))

    return acc_test


def train(model,
        train_features, train_labels,
        epochs=100, weight_decay=5e-6, lr=0.2):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t
    return model, train_time


def test(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


def accuracy(output, labels):
    preds = output.argmax(dim=1).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)

# perform n trials for a fair evaluation
accu_acc = []
for _ in range(args.trials):
    acc_test = main(args)
    accu_acc.append(acc_test)

accu_acc = np.array(accu_acc)
acc_mean, acc_std = accu_acc.mean(), accu_acc.std()

print('='*20)
print(f'Dataset: {args.dataset} Test accuracy of {args.trials} runs: mean {acc_mean:.5f}, std {acc_std:.5f}')
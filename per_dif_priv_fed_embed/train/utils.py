import time

import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from sklearn.manifold import TSNE
from tqdm import tqdm
# from tsne_torch import TorchTSNE as TSNE
from per_dif_priv_fed_embed.common.config import Config
import matplotlib.pyplot as plt
import numpy as np

WANDB_LOG = {}
LOG = {}


def get_loss_and_opt(net, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def save_model(model, path):
    torch.save(model.state_dict(), f'{path}_state_dict.pt')
    torch.save(model, f'{path}.pt')


def train_single_epoch(trainloader, net, criterion, optimizer, device):
    net.train()
    correct = 0
    total = 0
    max_prob_mean = torch.zeros(100).reshape(10, -1).to(device)
    labels_counter = torch.zeros(10).to(device)

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, centroids = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # sign sgd
        # for p in net.parameters():
        #     p.grad = torch.sign(p.grad)

        max_probs, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        max_prob_mean[labels] += outputs

        unique_labels, counts = torch.unique(labels, sorted=True, return_counts=True)
        for l_ind, l in enumerate(unique_labels):
            labels_counter[l_ind] += counts[l_ind]

        optimizer.step()

    assert labels_counter.sum() == total, f'labels_counter.sum()={labels_counter.sum()}  total={total}'
    WANDB_LOG['train_acc'] = 100 * correct / total
    # WANDB_LOG['train_max_prob'] = max_prob_mean / total
    for i in range(10):
        LOG[f'max_prob_mean_{i}'] = max_prob_mean[i] / labels_counter[i]


def eval_net(testloader, net, device):
    net.eval()
    correct = 0
    total = 0
    # max_prob_mean = 0
    max_prob_mean = torch.zeros(100).reshape(10, -1).to(device)
    labels_counter = torch.zeros(10).to(device)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs, centroids = net(inputs)
            # the class with the highest energy is what we choose as prediction
            max_probs, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # max_prob_mean += max_probs.sum().item()
            max_prob_mean[labels] += outputs
            unique_labels, counts = torch.unique(labels, sorted=True, return_counts=True)
            for l_ind, l in enumerate(unique_labels):
                labels_counter[l_ind] += counts[l_ind]

    WANDB_LOG['val_acc'] = 100 * correct / total
    # for i in range(10):
    #     max_prob_mean[i] /= labels_counter[i]
    #     WANDB_LOG[f'kl_div_label_{i}'] = kl_divergence(max_prob_mean[i].reshape(1, -1),
    #                                                    LOG[f'max_prob_mean_{i}'].reshape(1, -1))


def apply_tsne(trainloader, testloader, net, device):
    net.eval()

    all_labels_train = []
    all_outputs = np.zeros((1, 10))
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)

            all_labels_train.extend(labels.tolist())
            all_outputs = np.append(all_outputs, outputs.cpu().numpy(), axis=0)

    emb_train = TSNE(n_components=2, perplexity=30, n_iter=250, verbose=True).fit_transform(all_outputs[1:])
    # plt.scatter(emb[:, 0], emb[:, 1], c=all_labels)
    # plt.show()

    all_labels_test = []
    all_outputs = np.zeros((1, 10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)

            all_labels_test.extend(labels.tolist())
            all_outputs = np.append(all_outputs, outputs.cpu().numpy(), axis=0)

    print(all_outputs.shape)
    emb_test = TSNE(n_components=2, perplexity=30, n_iter=250, verbose=True).fit_transform(all_outputs[1:])

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 10))
    ax_train.scatter(emb_train[:, 0], emb_train[:, 1], c=all_labels_train)
    ax_train.set_title('Train')
    ax_test.scatter(emb_test[:, 0], emb_test[:, 1], c=all_labels_test)
    ax_test.set_title('Test')
    plt.show()


def train_method(trainloader, testloader, net, criterion, optimizer, device, epochs=50, save_model_every=100):
    for epoch in tqdm(range(epochs)):
        train_single_epoch(trainloader, net, criterion, optimizer, device=device)
        eval_net(testloader, net, device=device)

        wandb.log(WANDB_LOG)

        if save_model_every > 0 and epoch % save_model_every == (save_model_every - 1):
            filename_prefix = f'{Config.SAVED_MODELS_DIR}model_{time.asctime()}_epoch_{epoch}'
            save_model(model=net, path=filename_prefix)

    # apply_tsne(trainloader, testloader, net, device=device)

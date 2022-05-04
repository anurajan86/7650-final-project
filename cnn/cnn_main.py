import random

import torch.nn as nn
import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score

from cnn_model import CNNClassificationModel
from data_utils import *

genre_flag = True


def train(net, data, n_iter, lr=1e-5, batch_size=64, weight_decay=1e-2):
    print("Start Training!")
    loss_obj = nn.BCELoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)

    (X_train, y_train), (X_val, y_val), _ = data

    for epoch in range(n_iter):

        # if epoch <= 3:
        #     lr_ = lr
        # else:
        #     lr_ = lr * 0.1

        total_loss = 0.0
        net.train()   #Put the network into training mode
        idx_list = list(range(0, len(X_train)))
        random.shuffle(idx_list)

        for start_idx in tqdm.tqdm(range(0, len(X_train), batch_size), leave=False):
            idx_list_ = idx_list[start_idx: start_idx + batch_size]

            inp_sentence, _, inp_genre = pad_sequences([X_train[i] for i in idx_list_], genre_flag)

            pred = net(inp_sentence.cuda(), inp_genre.cuda()).squeeze()
            label = torch.tensor(y_train)[idx_list_].cuda().float()

            loss = loss_obj(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss

        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        eval(net, X_val, y_val)
        torch.save(cnn.state_dict(), f'cnn{epoch}.pt')


def eval(net, X, y, batch_size=64, outFile=None):
    y = torch.tensor(y).cuda()
    preds = []
    for start_idx in tqdm.tqdm(range(0, len(X), batch_size), leave=False):
        inp_sentence, _, inp_genre = pad_sequences([X[i] for i in range(start_idx, min(start_idx + batch_size, len(X)))], genre_flag)
        pred = net(inp_sentence.cuda(), inp_genre.cuda()).squeeze()
        preds.extend(pred.tolist())
    roc_auc = roc_auc_score(y.tolist(), preds)
    ap = average_precision_score(y.tolist(), preds)
    print(f"ROC AUC: {roc_auc}, AP: {ap}")
    if outFile:
        fOut = open(outFile, 'w')
        for i, pred in enumerate(preds):
            fOut.write(f"{X[i][1][0]}\t{pred}\n")


data, pretrained, vocab, word_to_id, id_to_word, genre_list = prepare_data(genre_flag=genre_flag)
cnn = CNNClassificationModel(pretrained_vector_weight=pretrained, num_genre=len(genre_list)).cuda()
train(cnn, data, n_iter=10)


print('Test result:')
eval(cnn, data[2][0], data[2][1], outFile='cnn_preds.txt')

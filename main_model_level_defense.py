# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 14:57
# @Author  : zhao
# @File    : main_fl_unsw15_oneshot.py
#### one-shot poisonning attack, test the accuracy of all the methods in this shot


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support,accuracy_score
import copy
from collections import Counter, Iterable
from itertools import chain,combinations, permutations
import random
from pyod.models.sos import SOS
from pyod.models.pca import PCA
from scipy.spatial.distance import cdist, euclidean
import argparse
from Net import CNN_UNSW,MLP_UNSW
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import math
import sklearn
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# convert a list of list to a list [[],[],[]]->[,,]
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def readdataset():
    normalized_X = np.load('X.npy')
    y = np.load('Y_attack.npy')
    print('y', sorted(Counter(y).items()))

    # downsampling = RandomUnderSampler(
    #     sampling_strategy={0: 100000, 1: 100000, 2: 44525, 3: 24246, 4: 16353, 5: 13987, 6: 0, 7: 0, 8: 0, 9: 0},
    #     random_state=0)
    downsampling = RandomUnderSampler(
        sampling_strategy={0: 400000, 1: 100000, 2: 44525, 3: 24246, 4: 16353, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        random_state=0)
    X_down, y_down = downsampling.fit_resample(normalized_X, y)
    # upsampling = RandomOverSampler(
    #     sampling_strategy={0: 100000, 1: 100000, 2: 100000, 3: 100000, 4: 100000, 5: 100000}, random_state=0)
    upsampling = RandomOverSampler(
        sampling_strategy={0: 400000, 1: 100000, 2: 100000, 3: 100000, 4: 100000}, random_state=0)
    Xt, yt = upsampling.fit_resample(X_down, y_down)
    print('transformed y', sorted(Counter(yt).items()))
    df = pd.DataFrame(Xt, index=yt)
    df.sort_index(ascending=True, inplace=True)
    train0 = df.iloc[0:280000]
    train1 = df.iloc[400000:400000 + 70000]
    train2 = df.iloc[500000:500000 + 70000]
    train3 = df.iloc[600000:600000 + 70000]
    train4 = df.iloc[700000:700000 + 70000]
    # train5 = df.iloc[500000:500000 + 70000]
    df_train = pd.concat([train0, train1, train2, train3, train4]) #, train5
    df_train = shuffle(df_train)
    np_features_train = df_train.values

    np_features_train = np_features_train[:, np.newaxis, :]
    np_label_train = df_train.index.values.ravel()
    print('train',sorted(Counter(np_label_train).items()))

    test0 = df.iloc[280000:400000]
    test1 = df.iloc[400000 + 70000:400000 + 100000]
    test2 = df.iloc[500000 + 70000:500000 + 100000]
    test3 = df.iloc[600000 + 70000:600000 + 100000]
    test4 = df.iloc[700000 + 70000:700000 + 100000]
    # test5 = df.iloc[500000 + 70000:500000 + 100000]
    df_test = pd.concat([test0, test1, test2, test3, test4]) #, test5
    df_test = shuffle(df_test)
    features_test = df_test.values
    np_features_test = np.array(features_test)

    np_features_test = np_features_test[:, np.newaxis, :]
    np_label_test = df_test.index.values.ravel()
    print('test',sorted(Counter(np_label_test).items()))
    return np_features_train, np_label_train,np_features_test,np_label_test


class ReadData(Dataset):
    def __init__(self, x_tra, y_tra):
        self.x_train = x_tra
        self.y_train = y_tra

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        image, label = self.x_train[item], self.y_train[item]
        image = torch.from_numpy(image)
        label = torch.from_numpy(np.asarray(label))
        return image, label

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.features, self.labels = self.dataset[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def iid(dataset, num_users,degree):
    num_normal = 280000//num_users
    num_attack = 280000//(num_users*degree)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(280000*2)
    labels = dataset.y_train
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) ###[[idxs 0,1,2,3],[labels 5,5,7,2]]
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:] ### idxs前224000为正常类样本的index，后面每116000为下一类
    dict_class_index = {} #{i: [] for i in range(7)}
    dict_class_index[0] = idxs[0:280000]
    for i in range(1,5):
        dict_class_index[i] = idxs[280000+(i-1)*70000:280000+i*70000]
    comb = list()
    for i in range(int(math.ceil(num_users/len(list(combinations([i for i in range(1, 5)], degree)))))):
        comb += list(combinations([i for i in range(1, 5)], degree))
    # comb_rand = random.sample(comb, 100)
    comb_rand = comb[0:100]
    print('comb',len(comb_rand))
    for i,classes in enumerate(comb_rand):
        # rand_set_normal = np.random.choice(dict_class_index[0], num_normal, replace=False)
        rand_set_normal = dict_class_index[0][0:num_normal]
        dict_users[i] = np.concatenate((dict_users[i], rand_set_normal), axis=0)
        dict_class_index[0] = list(set(dict_class_index[0]) - set(rand_set_normal))
        for cls in classes:
            if len(dict_class_index[cls])>= num_attack:
                # rand_set_attack = np.random.choice(dict_class_index[cls], num_attack, replace=False)
                rand_set_attack = dict_class_index[cls][0:num_attack]
                dict_users[i] = np.concatenate((dict_users[i], rand_set_attack), axis=0)
                dict_class_index[cls] = list(set(dict_class_index[cls]) - set(rand_set_attack))
            else:
                dict_users[i] = np.concatenate((dict_users[i], dict_class_index[cls]), axis=0)
    return dict_users



def test_img(net_g, datatest):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    x = datatest.x_train
    y = datatest.y_train
    anomaly_list = [i for i in range(len(y)) if y[i] != 0]
    y[anomaly_list] = 1
    dataset_test = ReadData(x,y)
    data_loader = DataLoader(dataset_test, batch_size=test_BatchSize)
    loss = torch.nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).type(torch.LongTensor).to(device)
        # data, target = Variable(data), Variable(target).type(torch.LongTensor)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss(log_probs, target).item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.detach().max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.detach().view_as(y_pred)).long().cpu().sum()
        data_pred.append(y_pred.cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    print('test_loss', test_loss)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_w(w, datatest):
    net_w = CNN_UNSW().double().to(device)
    net_w.load_state_dict(w)
    net_w.eval()
    # testing
    test_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    x = datatest.x_train
    y = datatest.y_train
    anomaly_list = [i for i in range(len(y)) if y[i] != 0]
    y[anomaly_list] = 1
    dataset_test = ReadData(x, y)
    data_loader = DataLoader(dataset_test, batch_size=test_BatchSize)
    loss = torch.nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device, dtype=torch.double), Variable(target).type(torch.LongTensor).to(device)
        # data, target = Variable(data), Variable(target).type(torch.LongTensor)
        log_probs = net_w(data)
        # sum up batch loss
        test_loss += loss(log_probs, target).item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.detach().max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.detach().view_as(y_pred)).long().cpu().sum()
        data_pred.append(y_pred.cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    # print('test_loss', test_loss)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def get_2_norm(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                params_b[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            b = copy.deepcopy(params_b[i].cpu().numpy())
            x = []
            y = []
            for j in a:
                x.append(copy.deepcopy(j.flatten()))
            for k in b:
                y.append(copy.deepcopy(k.flatten()))
            for m in range(len(x)):
                sum += pow(np.linalg.norm(x[m]-y[m], ord=2),2)
    norm = np.sqrt(sum)
    return norm

def defence_Krum(w, c):
    c = c+1
    euclid_dist_list = []
    euclid_dist_matrix = [[0 for i in range(len(w))] for j in range(len(w))]
    for i in range(len(w)):
        for j in range(i,len(w)):
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
        euclid_dist = euclid_dist_matrix[i][:]
        euclid_dist.sort()
        if len(w)>=c:
            euclid_dist_list.append(sum(euclid_dist[:c]))
        else:
            euclid_dist_list.append(sum(euclid_dist))
    s_w = euclid_dist_list.index(min(euclid_dist_list))
    w_avg = w[s_w]
    return  w_avg

def getGradVec(w):
    """Return the gradient flattened to a vector"""
    gradVec = []
    for k in w.keys():
        gradVec.append(w[k].view(-1).float())
    # concat into a single vector
    gradVec = torch.cat(gradVec).cpu().numpy()
    return gradVec

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def defence_GoeMed(w):
    w_avg = copy.deepcopy(w[0])
    ### Return the shapes and sizes of the weight matrices
    gradShapes = []
    gradSizes = []
    for k in w[0].keys():
        gradShapes.append(w[0][k].shape)
        gradSizes.append(np.prod(w[0][k].shape))
    ### Return the gradient flattened to a vector
    w_vec = []
    for i in range(len(w)):
       w_vec.append(getGradVec(w[i]))
    w_vec_array = np.array(w_vec).astype(float)
    # selected = torch.from_numpy(np.asarray(gmean(w_vec_array, axis=0))) ### 效果差，出现NAN
    selected = torch.from_numpy(np.asarray(geometric_median(w_vec_array)))
    ### 另一种计算geometric median方法
    # distance = euclidean
    # geometric_mediod = \
    # min(map(lambda p1: (p1, sum(map(lambda p2: distance(p1, p2), w_vec))), w_vec), key=lambda x: x[1])[0]
    # selected = torch.from_numpy(np.asarray(geometric_mediod))
    startPos = 0
    i = 0
    for k in w[0].keys():
        shape = gradShapes[i]
        size = gradSizes[i]
        i += 1
        # assert (size == np.prod(p.grad.data.size()))
        w_avg[k] = selected[startPos:startPos + size].reshape(shape).to(device)
        startPos += size
    return w_avg

def defence_det(w, d_out):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            ### 检测结果0为正常模型，1为异常模型
            if d_out[i]==0:
                w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], (len(d_out)-sum(d_out)))
    return w_avg

def defence_our(omega_locals,w_locals,w_local_pre):
    X_norm = []
    selected_index = {}
    for i in omega_locals[0].keys():
        aggregate_index = list()
        for j in range(0, len(omega_locals)):
            aggregate_index.append(omega_locals[j][i])
        selected_index[i] = Counter(list(chain(*aggregate_index)))
    print('interation', interation, 'client', client)
    # print('selected_index', selected_index)

    for i in range(0, len(w_locals)):
        selected_weights = []
        all_weights = []
        for n in w_locals[0].keys():
            # print(n)
            c = w_locals[i][n].cpu()
            c_pre = w_local_pre[n].cpu()
            # all_weights.append((c.view(-1).detach().numpy() - c_pre.view(-1).detach().numpy()))
            selected_index_dict = dict(selected_index[n])
            indice = []
            for a in range(0, len(selected_index_dict)):
                # if (list(selected_index_dict.values())[a] < 45)&(list(selected_index_dict.values())[a] >40):
                if (list(selected_index_dict.values())[a] > 90):
                    # if ((list(selected_index_dict.values())[a] > 30) & (list(selected_index_dict.values())[a] < 40)): # | (list(selected_index_dict.values())[a] > 95)
                    indice.append(list(selected_index_dict.keys())[a])
            if len(indice) > 0:
                indices = torch.tensor(indice)
                # print('indices', indices)
                d = torch.index_select(c.view(-1), 0, indices)
                d_pre = torch.index_select(c_pre.view(-1), 0, indices)
                selected_weights.append((d.view(-1).detach().numpy() - d_pre.view(-1).detach().numpy()))
            else:
                pass
        X_norm.append(list(chain(*selected_weights)))
        # X_all.append(list(chain(*all_weights)))

    X_norm = np.array(X_norm)
    # X_all = np.array(X_all)
    print('X', X_norm.shape)
    #### OUR
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(X_norm)
    X_norm = scaler.transform(X_norm)
    # outliers_fraction = float(num_poison_client/num_clients)
    # print('######outliers_fraction',outliers_fraction)
    random_state = 42
    clf = SOS(contamination=0.4, perplexity=90)

    clf.fit(X_norm)
    pre_out_label = clf.labels_
    print('prediction', pre_out_label)
    print(confusion_matrix(Y_norm.astype(int), pre_out_label))
    print(classification_report(Y_norm.astype(int), pre_out_label))
    # print("train AC", accuracy_score(Y_norm.astype(int), pre_out_label))
    ### Federated aggregation with defence
    w_glob = defence_det(w_locals, pre_out_label)
    return w_glob


def defence_pca2(w_locals,w_local_pre):
    X_norm = []
    for i in range(0, len(w_locals)):
        selected_weights = []
        for n in w_locals[0].keys():
            # print(n)
            c = w_locals[i][n].cpu()
            c_pre = w_local_pre[n].cpu()
            # selected_weights.append((c.view(-1).detach().numpy()))
            selected_weights.append((c.view(-1).detach().numpy() - c_pre.view(-1).detach().numpy()))
        X_norm.append(list(chain(*selected_weights)))
    X_norm = np.array(X_norm)
    pca = sklearn.decomposition.PCA()
    X_norm = pca.fit_transform(X_norm)
    print('X', X_norm.shape)
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(X_norm)
    X_norm = scaler.transform(X_norm)
    clf = SOS(contamination=0.4, perplexity=95)
    clf.fit(X_norm)
    pre_out_label = clf.labels_
    print('prediction', pre_out_label)
    print(confusion_matrix(Y_norm.astype(int), pre_out_label))
    print(classification_report(Y_norm.astype(int), pre_out_label))
    # print("train AC", accuracy_score(Y_norm.astype(int), pre_out_label))
    ### Federated aggregation with defence
    w_glob = defence_det(w_locals, pre_out_label)
    return w_glob

def defence_vae(w_locals):
    X_norm = []
    for i in range(0, len(w_locals)):
        selected_weights = []
        for n in w_locals[0].keys():
            # print(n)
            c = w_locals[i][n].cpu()
            selected_weights.append((c.view(-1).detach().numpy()))
        X_norm.append(list(chain(*selected_weights)))
    X_norm = np.array(X_norm)
    selected_index_x = joblib.load('vae_selected_index_unsw15_20210731.dpl')
    X_norm = X_norm[:, selected_index_x]
    print('X', X_norm.shape)
    model = torch.load('vae_128dim_unsw15_20210731.pkl')
    model.eval()
    running_loss = []
    pre_out_label = []
    for i in range(X_norm.shape[0]):
        single_x = torch.tensor(X_norm[i]).float()
        x_in = Variable(single_x).to(device)
        x_out, z_mu, z_logvar = model(x_in)
        # loss = self.criterion(x_out, x_in, z_mu, z_logvar)
        x_out = x_out.view(-1)
        x_in = x_in.view(-1)
        bce_loss = F.mse_loss(x_out, x_in, size_average=False)
        kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        # kld_loss = 0.5 * torch.sum(-1 - z_logvar + (z_mu ** 2) + torch.exp(z_logvar))
        loss = (bce_loss + kld_loss)
        running_loss.append(loss.item())
    score_avg = np.mean(running_loss)
    for score in running_loss:
        if score > score_avg:
            pre_out_label.append(1)
        else:
            pre_out_label.append(0)
    print('prediction', pre_out_label)
    print(confusion_matrix(Y_norm.astype(int), pre_out_label))
    print(classification_report(Y_norm.astype(int), pre_out_label))
    # print("train AC", accuracy_score(Y_norm.astype(int), pre_out_label))
    ### Federated aggregation with defence
    w_glob = defence_det(w_locals, pre_out_label)
    return w_glob


def consolidate(Model, Weight, MEAN_pre, epsilon):
    OMEGA_current = {n: p.data.clone().zero_() for n, p in Model.named_parameters()}
    for n, p in Model.named_parameters():
        p_current = p.detach().clone()
        p_change = p_current - MEAN_pre[n]
        # W[n].add_((p.grad**2) * torch.abs(p_change))
        # OMEGA_add = W[n]/ (p_change ** 2 + epsilon)
        # W[n].add_(-p.grad * p_change)
        OMEGA_add = torch.max(Weight[n], Weight[n].clone().zero_()) / (p_change ** 2 + epsilon)
        # OMEGA_add = Weight[n] / (p_change ** 2 + epsilon)
        # OMEGA_current[n] = OMEGA_pre[n] + OMEGA_add
        OMEGA_current[n] = OMEGA_add
    return OMEGA_current

class VAE(torch.nn.Module):
    def __init__(self, input_dim=128, latent_dim=20, hidden_dim=500): #input_dim=784,
        super(VAE, self).__init__()
        self.fc_e1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = torch.nn.Linear(hidden_dim, input_dim)

        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e1(x_in.view(-1, self.input_dim)))  ### input （-1，784） output (-1,500)
        x = F.relu(self.fc_e2(x))  ### (-1,500)
        mean = self.fc_mean(x)  #### mean (-1,20)
        logvar = F.softplus(self.fc_logvar(x))  ### logvar (-1,20)
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))  ### decoder input z (-1,20), output (-1,500)
        z = F.relu(self.fc_d2(z))  ### (-1,500)
        x_out = F.sigmoid(self.fc_d3(z))  ### (-1,768)
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        # sd = torch.exp(logvar * 0.5)
        # e = Variable(torch.randn(sd.size())).to(device)  # Sample from standard normal
        # z = e.mul(sd).add_(mean)
        std =logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

def test_adv(net_g, data_loader):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    loss = torch.nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).type(torch.LongTensor).to(device)
        log_probs = net_g(data)
        test_loss += loss(log_probs, target).item()
        y_pred = log_probs.data.detach().max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.detach().view_as(y_pred)).long().cpu().sum()
        data_pred.append(y_pred.cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest adv set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))

def cw_l2_attack(model, images, labels, targeted=False,c=1.0 , kappa=0, max_iter=1000, learning_rate=0.1):
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)
    # Define f-function
    def f(x):
        x = x.to(device)
        # print('x',x)
        outputs = model(x)
        # print('outputs', outputs)
        one_hot_labels = torch.eye(len(outputs[0]),device=device)[labels]#.to(device)
        # print('one_hot_labels', one_hot_labels)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1) ##除标签类外最大的概率
        # print('outputs',outputs)
        # print('one_hot_labels.bool()',one_hot_labels.bool())
        j = torch.masked_select(outputs, one_hot_labels.bool())#byte(), 标签类对应的概率
        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)
        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    changes_index = list(set([i for i in range(42)])-set([3,5,7,10,12,14,16,18,22,26,30])) ### 31 changed featurese
    images_change = images[:,:,changes_index]
    w = torch.zeros_like(images_change, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    prev = 1e10
    for step in range(max_iter):
        # print(torch.nn.Tanh()(w))
        # print( (1 / 2 * (torch.nn.Tanh()(w) + 1)))
        a0 = (1 / 2 * (torch.nn.Tanh()(w) + 1))
        a = images
        # print('a0',a0)
        a[:,:,changes_index]=a0[:,:,0:len(changes_index)]
        # print('a',a)
        loss1 = torch.nn.MSELoss(reduction='sum')(a0, images[:,:,0:len(changes_index)])
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward(retain_graph=True)
        optimizer.step()

        # if loss2.item() == 0.0:
        #     attack_images_change = 1 / 2 * (torch.nn.Tanh()(w) + 1)
        #     attack_images = images
        #     attack_images[:, :, changes_index] = attack_images_change[:, :, 0:len(changes_index)]
        #     print('Stop cost', cost.item(), 'loss1 MSE', loss1.item(), 'loss2 prediction', loss2.item())
        #     return attack_images
        # Early Stop when loss does not converge.
        # if step % (max_iter // 10) == 0:
        #     if cost > prev:
        #         print('Attack Stopped due to CONVERGENCE....')
        #         return a
        #     prev = cost
        # print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')
    print('cost',cost.item(),'loss1 MSE',loss1.item(),'loss2 prediction',loss2.item())
    attack_images_change = 1 / 2 * (torch.nn.Tanh()(w) + 1)
    attack_images = images
    attack_images[:,:,changes_index] = attack_images_change[:,:,0:len(changes_index)]
    return attack_images



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--defence', type=str, default="vae", choices=["fedavg", "our", "krum", "geomed","pca","vae"],
                        help="name of aggregation method")
    parser.add_argument('--scalar', type=float, nargs='?', default=1.0, help="sclar for poisoning model")
    parser.add_argument('--Tattack', type=int, nargs='?', default=3, help="attack round")
    parser.add_argument('--prate', type=float, nargs='?', default=1.0, help="poison instance ratio")
    args = parser.parse_args()

    scalar = args.scalar
    Ta = args.Tattack
    CWT = args.CWT
    prate = args.prate
    frac = 1.0
    num_clients = 100
    batch_size = 128
    test_BatchSize = 32
    x_train,y_train, x_test,y_test = readdataset()
    dataset_train = ReadData(x_train,y_train)
    dataset_test = ReadData(x_test,y_test)

    save_global_model = 'save_model.pkl'
    # # IID Data
    dict_clients = iid(dataset_train,num_clients,1)

    net_global = CNN_UNSW().double().to(device)
    # net_global = MLP_UNSW().double().to(device)
    w_glob = net_global.state_dict()
    crit = torch.nn.CrossEntropyLoss()
    net_global.train()

    for interation in range(Ta):
        w_locals, loss_locals = [], []
        w_local_pre = w_glob
        omega_locals = []
        ##### save the model weight as npy
        # X_norm = []
        # X_all = []
        Y_norm = np.empty(shape=[0, 1])

        num_poison_client = 0
        for client in range(num_clients):
            net = copy.deepcopy(net_global).to(device)
            net_pre = copy.deepcopy(net_global.state_dict())
            net.train()
            mean_pre = {n: p.clone().detach() for n, p in net.named_parameters()}
            w = {n: p.clone().detach().zero_() for n, p in net.named_parameters()}

            # opt_net = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.5) #0.05
            opt_net = torch.optim.Adam(net.parameters())
            print('interation', interation, 'client', client)
            idx_traindataset = DatasetSplit(dataset_train, dict_clients[client])
            x = idx_traindataset.features.detach().cpu().numpy()
            y = idx_traindataset.labels.detach().cpu().numpy()

            anomaly_list = [i for i in range(len(y)) if y[i] != 0]
            y[anomaly_list] = 1
            num_attack1 = np.sum(y == 1)
            num_poison = int(num_attack1 * prate)  # 0.8

            if (num_poison > 0) & (num_poison_client<40)& (interation == (Ta-1)): # & (interation > 0)
                num_poison_client += 1
                Y_norm = np.row_stack((Y_norm, [1]))  ### 异常为1
                print('##########poison client', num_poison_client)
                poison_client_flag = True
                res_list = [i for i in range(len(y)) if y[i] == 1]
                ###### Label flipping
                y[res_list[0:num_poison]] = 0
                ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                epochs_per_task = 5
            else:
                Y_norm = np.row_stack((Y_norm, [0]))
                poison_client_flag = False
                ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                epochs_per_task = 5
            # ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
            # test_adv(best_model, ldr_train)
            # epochs_per_task = 5

            dataset_size = len(ldr_train.dataset)

            for epoch in range(1, epochs_per_task + 1):
                correct = 0
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    old_par = {n: p.clone().detach() for n, p in net.named_parameters()}
                    images, labels = Variable(images).to(device), Variable(labels).type(torch.LongTensor).to(device)
                    net.zero_grad()
                    scores = net(images)
                    ce_loss = crit(scores, labels)
                    loss = ce_loss
                    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
                    pred = scores.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
                    loss.backward()
                    opt_net.step()
                    j = 0
                    for n, p in net.named_parameters():
                        # print(n,grad_params[j])
                        w[n] -= (grad_params[j].clone().detach()) * (p.detach() - old_par[n])###
                        j += 1
                Accuracy = 100. * correct.type(torch.FloatTensor) / dataset_size
                print('Train Epoch:{}\tLoss:{:.4f}\tCE_Loss:{:.4f}\tAccuracy: {:.4f}'.format(epoch,loss.item(),ce_loss.item(),Accuracy))
            # print(classification_report(labels.cpu().data.view_as(pred.cpu()), pred.cpu()))
            omega = consolidate(Model=net, Weight=w, MEAN_pre=mean_pre, epsilon=0.0001)
            omega_index = {}
            for k in omega.keys():
                if len(omega[k].view(-1))>1000:
                    Topk = 100
                else:
                    Topk = int(0.1 * len(omega[k].view(-1)))
                # Topk = int(0.1 * len(omega[k].view(-1)))
                Topk_value_index = torch.topk(omega[k].view(-1), Topk)
                omega_index[k] = Topk_value_index[1].tolist()
            omega_locals.append(omega_index)

            # w_locals.append(copy.deepcopy(net.state_dict()))
            if poison_client_flag:
                net_poison = copy.deepcopy(net.state_dict())
                for key in net_pre.keys():
                    difference = net_poison[key] - mean_pre[key]
                    scale_up = scalar # 5.0,3.0
                    net_poison[key] = scale_up*difference + mean_pre[key]
                w_locals.append(net_poison)
            else:
                w_locals.append(copy.deepcopy(net.state_dict()))

        if interation == (Ta-1):
            ####### no poisonning attack
            w_locals_normal = []
            for client in range(num_clients):
                net_normal = copy.deepcopy(net_global).to(device)
                net_normal.train()
                # opt_net = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.5) #0.05
                opt_net = torch.optim.Adam(net_normal.parameters())
                print('interation', interation, 'client', client)
                idx_traindataset = DatasetSplit(dataset_train, dict_clients[client])
                x = idx_traindataset.features.detach().cpu().numpy()
                y = idx_traindataset.labels.detach().cpu().numpy()
                anomaly_list = [i for i in range(len(y)) if y[i] != 0]
                y[anomaly_list] = 1
                ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                epochs_per_task = 5
                dataset_size = len(ldr_train.dataset)
                for epoch in range(1, epochs_per_task + 1):
                    correct = 0
                    for batch_idx, (images, labels) in enumerate(ldr_train):
                        images, labels = Variable(images).to(device), Variable(labels).type(torch.LongTensor).to(device)
                        net_normal.zero_grad()
                        scores = net_normal(images)
                        ce_loss = crit(scores, labels)
                        loss = ce_loss
                        pred = scores.max(1)[1]
                        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
                        loss.backward()
                        opt_net.step()
                    Accuracy = 100. * correct.type(torch.FloatTensor) / dataset_size
                    print('Train Epoch:{}\tLoss:{:.4f}\tCE_Loss:{:.4f}\tAccuracy: {:.4f}'.format(epoch, loss.item(),
                                                                                                 ce_loss.item(),
                                                                                                 Accuracy))
                    # print(classification_report(labels.cpu().data.view_as(pred.cpu()), pred.cpu()))
                w_locals_normal.append(copy.deepcopy(net_normal.state_dict()))
            w_glob_normal = FedAvg(w_locals_normal)
            test_acc_normal, test_loss_normal = test_w(w_glob_normal, dataset_test)
            print(
                'No poisoning attack Fedavg Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss_normal,
                                                                                                      test_acc_normal))
            #############
            t0 = time.clock()
            w_glob = FedAvg(w_locals)
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc, test_loss = test_w(w_glob, dataset_test)
            print('Fedavg Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss, test_acc))
            t0 = time.clock()
            w_glob_our = defence_our(omega_locals, w_locals, w_local_pre)
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc, test_loss = test_w(w_glob_our, dataset_test)
            print('OUR Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss, test_acc))
            t0 = time.clock()
            w_glob_pca = defence_pca2(w_locals, w_local_pre)
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc_pca, test_loss_pca = test_w(w_glob_pca, dataset_test)
            print('PCA Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss_pca, test_acc_pca))
            t0 = time.clock()
            w_glob_vae = defence_vae(w_locals)
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc_vae, test_loss_vae = test_w(w_glob_vae, dataset_test)
            print('VAE Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss_vae, test_acc_vae))
            t0 = time.clock()
            w_glob_krum = defence_Krum(w=w_locals,c=(num_clients-num_poison_client-2))
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc_krum, test_loss_krum = test_w(w_glob_pca, dataset_test)
            print('KRUM Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss_krum, test_acc_krum))
            t0 = time.clock()
            w_glob_geomed = defence_GoeMed(w=w_locals)
            t1 = time.clock()
            print('Time:\t', str(t1 - t0))
            test_acc_geomed, test_loss_geomed = test_w(w_glob_geomed, dataset_test)
            print('GEOMED Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss_geomed, test_acc_geomed))

        else:
            w_glob = FedAvg(w_locals)
            # pass

        # copy weight to net_glob
        net_global.load_state_dict(w_glob)
        # net_global.load_state_dict(w_glob)
        net_global.eval()
        acc_test, loss_test = test_img(net_global, dataset_test)
        print("Testing accuracy: {:.2f}".format(acc_test))


    model_dict = net_global.state_dict()  # 自己的模型参数变量
    test_dict = {k: w_glob[k] for k in w_glob.keys() if k in model_dict}  # 去除一些不需要的参数
    model_dict.update(test_dict)  # 参数更新
    net_global.load_state_dict(model_dict)  # 加载

    # net_global.load_state_dict(w_glob)
    net_global.eval()
    acc_test, loss_test = test_img(net_global, dataset_test)
    print("Testing accuracy: {:.2f}".format(acc_test))
    #### save the model trained with the norm dataset
    # torch.save(net_global,save_global_model)

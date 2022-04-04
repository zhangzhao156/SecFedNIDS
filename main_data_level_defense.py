# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 21:44
# @Author  : zhao
# @File    : main_fl_unsw15_oneshot_filter_new.py


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.modules import activation, dropout,batchnorm
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support,accuracy_score
import copy
from collections import Counter,Iterable, defaultdict, deque, OrderedDict
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
import inspect
import warnings
warnings.filterwarnings('ignore')
import traceback
from hprofile import Profile, jaccard_simple
from utils import TorchHook, DDPCounter, submatrix_generator, get_index

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

class TorchProfiler():

    def __init__(self, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super().__init__()
        self.activation_classes = [m[1] for m in inspect.getmembers(activation, inspect.isclass) if
                                   m[1].__module__ == 'torch.nn.modules.activation']  #### 激活函数类型
        self.dropout_classes = [m[1] for m in inspect.getmembers(dropout, inspect.isclass) if
                                m[1].__module__ == 'torch.nn.modules.dropout']  ##### dropout函数类型
        self.batchnorm_classes = [m[1] for m in inspect.getmembers(batchnorm, inspect.isclass) if
                                m[1].__module__ == 'torch.nn.modules.batchnorm']
        self.implemented_classes = [torch.nn.Linear,
                                    torch.nn.MaxPool1d,
                                    torch.nn.AdaptiveAvgPool1d,
                                    torch.nn.Conv1d]
        self.contrib_functions = ['_contrib_linear',
                                  '_contrib_max1d',
                                  '_contrib_adaptive_avg_pool1d',
                                  '_contrib_conv1d']

        self.model = TorchHook(model)
        self.hooks = self.model.available_modules()

    def create_layers(self, nlayers=0):
        hooks = self.hooks
        if nlayers == 0:
            # this will be greater than we need because activation layers will join with implemented layers.
            nlayers = len(hooks)
        namelist = set()

        layerdict = OrderedDict()

        revhooks = reversed(hooks)
        layeridx = DDPCounter(start=0)
        tmplyer = deque()
        for kdx in revhooks:
            if layeridx() == nlayers:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer), 0]
                break
            this_type = type(hooks[kdx])
            if this_type in self.dropout_classes:
                continue
            elif this_type in self.activation_classes:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                continue
            elif this_type in self.batchnorm_classes:
                # tmplyer.appendleft(kdx)
                # namelist.add(kdx)
                continue
            elif this_type in self.implemented_classes:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer),
                                         self.contrib_functions[self.implemented_classes.index(this_type)]]
                tmplyer = deque()
                continue
            else:
                print(f'profiler not implemented for layer of type: {type(hooks[kdx])}')
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer), 0]
                break
        else:
            layeridx.inc()
            layerdict[layeridx()] = [0, 0]
        namelist = list(namelist)
        if len(namelist) > 0:
            self.model.add_hooks(namelist)
        return layerdict

    def _single_profile(self, x_in, y_out, R, layers, layerdict, ldx, threshold):
        func = getattr(self.__class__, layerdict[ldx][1])
        return func(self, x_in, y_out, R, layers, threshold)

    def create_profile(self, x, layerdict, n_layers=0, threshold=0.5, show_progress=False, parallel=False):
        x = x.to(device)

        with torch.no_grad():
            y, actives = self.model.forward(x)
            # print('actives',actives.keys())

            neuron_counts = defaultdict(list)
            synapse_counts = defaultdict(Counter)
            synapse_weights = defaultdict(list)

            # initialize profile with index of maximal logit from last layer
            neuron = int((torch.argmax(y[0].cpu())).detach().numpy())
            neuron_counts[0].append(neuron)
            synapse_counts[0].update([(neuron, neuron, 0)])
            synapse_weights[0].append(torch.max(y[0].cpu()))
            mask = torch.zeros_like(y.cpu())
            mask[:, torch.argmax(y.cpu())] = 1
            R = y.cpu() * mask
            #             R = (torch.max(y[0])).detach().numpy()
            #             print('R0',R,R.shape)

            if n_layers == 0 or n_layers >= len(layerdict):
                n = len(layerdict)
            else:
                n = n_layers + 1
            for ldx in range(1, n):
                try:
                    if show_progress:
                        print(f'Layer #{ldx}')
                    inlayers, incontrib = layerdict[ldx + 1]
                    if incontrib == 0 and inlayers == 0:
                        x_in = x
                    else:
                        x_in = actives[inlayers[-1]]

                    # next retrieve y_out
                    layers, contrib = layerdict[ldx]
                    y_out = actives[layers[-1]]

                    nc, sc, sw, Rx = self._single_profile(x_in, y_out, R, layers, layerdict, ldx, threshold)
                    neuron_counts[ldx].append(nc)
                    synapse_counts[ldx].update(sc)
                    synapse_weights[ldx].append(sw)

                    #### 保存重要神经元激活值
                    # ords = [v for v in x_in.cpu().view(-1)]
                    # for i in selected_index[ldx]:
                    #     synapse_weights[ldx].append(ords[i].item())  # (i,ords[i])

                    R = Rx

                except Exception as ex:
                    traceback.print_exc()
                    break

            return Profile(neuron_counts=neuron_counts,
                           synapse_counts=synapse_counts,
                           synapse_weights=synapse_weights, num_inputs=1)


    def _contrib_max1d(self, x_in, y_out, R, layer, threshold=0.001):

        neuron_counts = list()
        synapse_counts = Counter()
        synapse_weights = list()
        maxpool = self.model.available_modules()[layer[0]]

        # Grab dimensions of maxpool from parameters
        stride = maxpool.stride
        kernel_size = maxpool.kernel_size
        #         Rx = torch.zeros_like(x_in)

        tmp_return_indices = bool(maxpool.return_indices)
        maxpool.return_indices = True
        _, indices = maxpool.forward(x_in)
        maxpool.return_indices = tmp_return_indices
        Rx = torch.nn.functional.max_unpool1d(input=R, indices=indices.cpu(), kernel_size=kernel_size, stride=stride,
                                              padding=maxpool.padding, output_size=x_in.shape)

        #### Topk neuron
        # # K = 100
        # K = int(threshold * len(Rx.view(-1)))
        # TOPK_value_index = torch.topk(Rx.view(-1), K)
        # neuron_counts.append(TOPK_value_index[1].tolist())

        ### Topk channel
        # print('Rx',Rx.shape)
        Rx_sum = torch.sum(Rx,dim=2)
        K = int(threshold * len(Rx_sum.view(-1)))
        TOPK_value_index = torch.topk(Rx_sum.view(-1), K)
        neuron_counts.append(TOPK_value_index[1].tolist())

        return neuron_counts, synapse_counts, synapse_weights, Rx

    def _contrib_adaptive_avg_pool1d(self, x_in, y_out, R, layer, threshold=0.001):

        neuron_counts = list()
        synapse_counts = Counter()
        synapse_weights = list()
        avgpool = self.model.available_modules()[layer[0]]

        '''Grab the dimensions used by an adaptive pooling layer'''
        output_size = avgpool.output_size[0]
        input_size = x_in.shape[-1]
        stride = (input_size // output_size)
        kernel_size = input_size - (output_size - 1) * stride
        Rx = torch.zeros_like(x_in).cpu()  ###, dtype=np.float

        for i in range(R.size(2)):
            for j in range(R.size(3)):
                Z = x_in[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].cpu()
                Zs = Z.sum(axis=(2, 3), keepdims=True)
                Zs += 1e-12 * ((Zs >= 0).float() * 2 - 1)
                Rx[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += (
                        (Z / Zs) * R[:, :, i:i + 1, j:j + 1])

        # # K = 100
        # K = int(threshold * len(Rx.view(-1)))
        # TOPK_value_index = torch.topk(Rx.view(-1), K)
        # neuron_counts.append(TOPK_value_index[1].tolist())

        ### Topk channel
        # print('Rx',Rx.shape)
        Rx_sum = torch.sum(Rx,dim=2)
        K = int(threshold * len(Rx_sum.view(-1)))
        TOPK_value_index = torch.topk(Rx_sum.view(-1), K)
        neuron_counts.append(TOPK_value_index[1].tolist())

        return neuron_counts, synapse_counts, synapse_weights, Rx

    def _contrib_conv1d(self, x_in, y_out, R, layers,
                        threshold=0.001):  ###(self, x_in, y_out, ydx, layer, threshold=0.1)
        threshold = 0.1
        neuron_counts = list()
        synapse_counts = Counter()
        synapse_weights = list()
        conv, actf = layers  ### conv卷积层，actf激活函数
        conv = self.model.available_modules()[conv]
        actf = self.model.available_modules()[actf]

        # assumption is that kernel size, stride are equal in both dimensions
        # and padding preserves input size
        kernel_size = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        #         print('kernel',kernel_size,'s',stride,'padding',padding)
        W = conv._parameters['weight']
        B = conv._parameters['bias']
        #         print('Wcov',W.shape,'Bcov',B.shape)

        #         Rx = torch.zeros_like(x_in)

        #         Z = y_out
        Z = torch.nn.functional.conv1d(x_in, weight=W, bias=None, stride=stride, padding=padding).cpu()
        S = R / (Z + 1e-16 * ((Z >= 0).float() * 2 - 1.))
        C = torch.nn.functional.conv_transpose1d(input=S, weight=W.cpu(), bias=None, stride=stride, padding=padding)
        Rx = C * x_in.cpu()

        # # K = 100
        # K = int(threshold * len(Rx.view(-1)))
        # TOPK_value_index = torch.topk(Rx.view(-1), K)
        # neuron_counts.append(TOPK_value_index[1].tolist())

        ### Topk channel
        # print('Rx',Rx.shape)
        Rx_sum = torch.sum(Rx,dim=2)
        K = int(threshold * len(Rx_sum.view(-1)))
        # print('K',K)
        TOPK_value_index = torch.topk(Rx_sum.view(-1), K)
        neuron_counts.append(TOPK_value_index[1].tolist())

        return neuron_counts, synapse_counts, synapse_weights, Rx

    def _contrib_linear(self, x_in, y_out, R, layers, threshold=0.0001):
        threshold = 0.1
        neuron_counts = list()
        synapse_counts = Counter()
        synapse_weights = list()
        Rx = torch.zeros_like(x_in).cpu()

        if len(layers) == 1:
            linear = layers[0]

            def actf(x):
                return x
        else:
            linear, actf = layers
            actf = self.model.available_modules()[actf]
        linear = self.model.available_modules()[linear]
        # print('linear',linear)

        xshape = x_in.shape
        xdims = x_in[0].shape
        if len(xdims) > 1:
            holdx = torch.Tensor(x_in.cpu())
            x_in = x_in[0].view(-1).unsqueeze(0)
        #         print('linear flatten xin',x_in.shape)

        W = linear._parameters['weight']
        B = linear._parameters['bias']

        #         Z = y_out
        Z = torch.nn.functional.linear(x_in, W, bias=None).cpu()
        #         print('Z',Z.shape)
        S = R / (Z + 1e-16 * ((Z >= 0).float() * 2 - 1.))
        #         print('S',S.shape,'W',W.shape)
        #         C = torch.matmul(S,W)
        #         Rx = (x_in*C).reshape(xshape)
        Rx = torch.nn.functional.linear(S, W.t().cpu(), bias=None)
        Rx *= x_in.cpu()
        Rx = Rx.reshape(xshape)
        #         print('C',C.shape)

        # print('Rx',Rx.shape)
        # K = 100
        K = int(threshold * len(Rx.view(-1)))
        TOPK_value_index = torch.topk(Rx.view(-1), K)
        neuron_counts.append(TOPK_value_index[1].tolist())

        return neuron_counts, synapse_counts, synapse_weights, Rx

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # Generates random samples from all_idexs,return a array with size of num_items
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, num_users,degree):
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


def getGradVec(w):
    """Return the gradient flattened to a vector"""
    gradVec = []
    for k in w.keys():
        gradVec.append(w[k].view(-1).float())
    # concat into a single vector
    gradVec = torch.cat(gradVec).cpu().numpy()
    return gradVec


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
    return w_glob, pre_out_label


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
    parser.add_argument('--defence', type=str, default="our", choices=["fedavg", "our", "krum", "geomed","pca","vae"],
                        help="name of aggregation method")
    parser.add_argument('--prate', type=float, nargs='?', default=0.1, help="poison instance ratio")
    parser.add_argument('--Tattack', type=int, nargs='?', default=10, help="attack round")
    args = parser.parse_args()

    prate = args.prate
    Ta = args.Tattack
    frac = 1.0
    num_clients = 100
    batch_size = 128
    test_BatchSize = 32
    x_train,y_train, x_test,y_test = readdataset()
    dataset_train = ReadData(x_train,y_train)
    dataset_test = ReadData(x_test,y_test)

    save_global_model = 'save_model.pkl'
    # # IID Data
    # dict_clients = iid(dataset_train, num_users=num_clients)
    dict_clients = noniid(dataset_train,num_clients,1)

    net_global = CNN_UNSW().double().to(device)
    # net_global = MLP_UNSW().double().to(device)
    w_glob = net_global.state_dict()
    crit = torch.nn.CrossEntropyLoss()
    net_global.train()

    x_client = {}
    y_client = {}
    normal_list_client = {}
    anomaly_list_client = {}
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
            num_poison = int(num_attack1 * 1.0)  # 0.8

            if (num_poison > 0) & (num_poison_client<40)& (interation == (Ta-1)): # & (interation > 0)
                num_poison_client += 1
                Y_norm = np.row_stack((Y_norm, [1]))  ### 异常为1
                print('##########poison client', num_poison_client)
                poison_client_flag = True
                res_list = [i for i in range(len(y)) if y[i] == 1] ###
                res_list1 = [i for i in range(len(y)) if y[i] != 1] ### normal
                ###### 标签翻转
                # x1 = x[res_list1, :, :] ### clean data
                # y1 = y[res_list1] ### labels of clean data
                # y[res_list[0:num_poison]] = 0
                # x2 = x[res_list, :, :] ### poison data
                # y2 = y[res_list[0:num_poison]]
                # x = np.concatenate((x1, x2[0:int(prate * len(x2)), :, :]), axis=0)
                # y = np.concatenate((y1, y2[0:int(prate * len(x2))]), axis=0)
                # ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                # epochs_per_task = 5
                # normal_list_client[client] = res_list1
                # anomaly_list_client[client] = [i for i in range(len(x1),(len(x1)+int(prate*len(x2))))]
                ###### CW攻击
                x1 = x[res_list1, :, :]  ### clean data
                y1 = y[res_list1]  ### labels of clean data
                x2 = x[res_list, :, :]
                y2 = y[res_list]
                best_model = torch.load('save_model_unsw15_noniid_20210802.pkl')
                x2_adv = cw_l2_attack(model=best_model, images=torch.tensor(x2), labels=torch.LongTensor(y2),
                                      targeted=False, c=1.0, kappa=0, max_iter=20, learning_rate=0.1)
                # x[res_list, :, :] = x2_adv[0:len(res_list), :, :].detach().cpu().numpy()
                # y[res_list[0:num_poison]] = 0
                x = np.concatenate((x1, x2_adv[0:int(prate*len(x2_adv)), :, :].detach().cpu().numpy()), axis=0)
                y = np.concatenate((y1, y2[0:int(prate*len(x2_adv))]), axis=0)
                ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                # test_adv(best_model, ldr_train)
                epochs_per_task = 20
                normal_list_client[client] = res_list1
                anomaly_list_client[client] = [i for i in range(len(x1), (len(x1) + int(prate * len(x2))))]
            else:
                Y_norm = np.row_stack((Y_norm, [0]))
                poison_client_flag = False
                ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
                epochs_per_task = 5
                if interation == (Ta-1):
                    normal_list_client[client] = [i for i in range(len(y)) if y[i] == 0]
                    anomaly_list_client[client] = [i for i in range(len(y)) if y[i] != 0]

            # ldr_train = DataLoader(ReadData(x, y), batch_size=1024, shuffle=True)
            # test_adv(best_model, ldr_train)
            # epochs_per_task = 5
            if interation == (Ta-1):
                x_client[client] = x
                y_client[client] = y

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
                    scale_up = 5.0 # 10.0
                    net_poison[key] = scale_up*difference + mean_pre[key]
                w_locals.append(net_poison)
            else:
                w_locals.append(copy.deepcopy(net.state_dict()))

        if interation == (Ta-1):
            w_glob,pre_out_label = defence_our(omega_locals, w_locals, w_local_pre)
            test_acc, test_loss = test_w(w_glob, dataset_test)
            print('OUR Test set: Average loss: {:.4f} \tAccuracy: {:.2f}'.format(test_loss, test_acc))
            print('###########Filter##################')
            normal_client_indexs = []
            poison_client_indexs = []
            for i in range(len(pre_out_label)):
                if pre_out_label[i] == 1:
                    poison_client_indexs.append(np.int(i))
                else:
                    normal_client_indexs.append(np.int(i))
            model = CNN_UNSW().to(device)  #
            model.load_state_dict(w_glob)
            model = model.eval()
            profiler = TorchProfiler(model)
            layerdict = profiler.create_layers(0)  #### all layers
            print(layerdict)
            tp = profiler.create_profile(torch.rand(1, 1, 42), layerdict, threshold=0.5, show_progress=False,
                                         parallel=False)
            class_profiles = dict()
            selected_index = dict()
            class_profiles_mal = dict()
            iou_normal = dict()
            iou_threshold = dict()
            neuron_count = {1: 12, 2: 28, 3: 3, 4: 1, 5: 1}
            # neuron_count = {1: 12, 2: 28, 3: 16, 4: 8, 5: 8}
            #####predefine class_profiles,class_profiles_mal,selected_index(class path),iou_normal(threshold)
            for cls in range(2):
                class_profiles[cls] = dict()
                selected_index[cls] = dict()
                class_profiles_mal[cls] = dict()
                iou_normal[cls] = list()
                for layer in tp.neuron_counts:
                    class_profiles[cls][layer] = list()
                    selected_index[cls][layer] = Counter({})
                    class_profiles_mal[cls][layer] = list()
                    # print(tp.neuron_counts[layer])
                    # print('Layer ', layer, 'the number of important neurons:', len(list(chain(*tp.neuron_counts[layer][0]))))

            for index in normal_client_indexs:
                # # print('normal client', index)
                # idx_traindataset = DatasetSplit(dataset_train, dict_clients[index])
                # images = idx_traindataset.features.detach().cpu().numpy()  # .numpy()
                # # print('images',images.shape)
                # labels = idx_traindataset.labels.detach().cpu().numpy()
                # anomaly_list = [i for i in range(len(labels)) if labels[i] != 0]
                # labels[anomaly_list] = 1
                # if Y_norm[index] == 1:
                #     for i in range(len(labels)):
                #         if labels[i] == 1:
                #             labels[i] = 0

                images = x_client[index]
                labels = y_client[index]

                normal_client_sampling_indexs = [i for i in range(len(labels))]
                normal_client_sampling_index = np.random.choice(normal_client_sampling_indexs,
                                                                int(len(labels) * 0.03),
                                                                replace=False)
                # for i in range(len(labels)):
                for i in normal_client_sampling_index:
                    tprofiles = profiler.create_profile(torch.Tensor(images[i]).resize_(1, 1, 42),
                                                        layerdict,
                                                        threshold=0.5,
                                                        show_progress=False,
                                                        parallel=False)
                    for layer in tprofiles.neuron_counts:
                        if layer == 0:
                            ###### 考虑预测错误样本加入神经元的聚合
                            # class_profiles[labels[i]][layer].append(tprofiles.neuron_counts[layer])
                            ###### 只聚合预测正确样本的神经元
                            if (tprofiles.neuron_counts[0])[0] == labels[i]:
                                class_profiles[labels[i]][layer].append(tprofiles.neuron_counts[layer])
                        else:
                            ###### 考虑预测错误样本加入神经元的聚合
                            # class_profiles[labels[i]][layer].append(list(chain(*tprofiles.neuron_counts[layer][0])))
                            ###### 只聚合预测正确样本的神经元
                            if (tprofiles.neuron_counts[0])[0] == labels[i]:
                                class_profiles[labels[i]][layer].append(
                                    list(chain(*tprofiles.neuron_counts[layer][0])))
                        selected_index[labels[i]][layer] += Counter(list(chain(*class_profiles[labels[i]][layer])))
            for cls in range(2):
                for i in range(len(class_profiles[cls][1])):
                    ious = list()
                    for layer in range(1, 6):
                        ious.append(jaccard_simple(set(class_profiles[cls][layer][i]),
                                                   set([val[0] for j, val in enumerate(
                                                       selected_index[cls][layer].most_common(
                                                           neuron_count[layer]))])))
                    avg_ious = np.mean(ious)
                    iou_normal[cls].append(avg_ious)
            for cls in range(2):
                # iou_threshold[cls] = np.median(np.array(iou_normal[cls]))
                iou_threshold[cls] = np.percentile(np.array(iou_normal[cls]), 5)
            print('iou_threshold', iou_threshold)
            print('###############normal client done')
            for client in poison_client_indexs:                

                images = x_client[client]
                labels = y_client[client]
                anomaly_list = anomaly_list_client[client]
                normal_list = normal_list_client[client]

                normal_list_predicted = []
                anomaly_list_predicted = []
                for i in range(len(labels)):
                    tprofiles_mal = profiler.create_profile(torch.Tensor(images[i]).resize_(1, 1, 42),
                                                            layerdict,
                                                            threshold=0.5,
                                                            show_progress=False,
                                                            parallel=False)
                    ious = list()
                    for layer in range(1, 6):
                        ious.append(jaccard_simple(set(list(chain(*tprofiles_mal.neuron_counts[layer][0]))),
                                                   set([val[0] for j, val in enumerate(
                                                       selected_index[labels[i]][layer].most_common(
                                                           neuron_count[layer]))])))
                    avg_ious = np.mean(ious)
                    if avg_ious < iou_threshold[labels[i]]:
                        anomaly_list_predicted.append(i)
                    
                # recall_normal = set(normal_list_predicted).intersection(set(normal_list))
                recall_anomaly = set(anomaly_list_predicted).intersection(set(anomaly_list))
                # print('normal 0', 'recall of predicted: ', len(recall_normal) / len(normal_list_predicted),
                #       'recall of all: ', len(recall_normal) / len(normal_list))
                print('anomaly 1', 'recall of predicted: ', len(recall_anomaly) / len(anomaly_list_predicted),
                      'recall of all: ', len(recall_anomaly) / len(anomaly_list),'clean removed: ',(len(anomaly_list_predicted)-len(recall_anomaly))/len(normal_list),'recall_anomaly: ',len(recall_anomaly),len(anomaly_list),len(anomaly_list_predicted),len(normal_list))
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

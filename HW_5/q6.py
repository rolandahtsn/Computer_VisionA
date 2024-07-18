import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data.dataset import TensorDataset


max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 3e-3 #7e-3; 1e-3
hidden_size = 64

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')
test_data = scipy.io.loadmat('data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_xt = torch.from_numpy(train_x).type(torch.float32)
train_yt = torch.from_numpy(train_y).type(torch.float32)
valid_xt = torch.from_numpy(valid_x).type(torch.float32)
valid_yt = torch.from_numpy(valid_y).type(torch.float32)

load_train_data = torch.utils.data.DataLoader(TensorDataset(train_xt,train_yt),batch_size,shuffle = True)
load_valid_data = torch.utils.data.DataLoader(TensorDataset(valid_xt,valid_yt),batch_size,shuffle = True)

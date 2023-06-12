import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
import random
import numpy as np
import math
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in semi-supervised learning')
# Data from http://dreamchallenges.org/project/dream-5-network-inference-challenge/
# data1: In silico
# data3: E. coli
# data4: Yeast
# general settings
# parser.add_argument('--traindata-name', default='data3', help='train network name')
# parser.add_argument('--traindata-name2', default=None, help='also train another network')
# parser.add_argument('--testdata-name', default='data4', help='test network name')
parser.add_argument('--data-name', default='data3', help='data name')
parser.add_argument('--training-ratio', type=float, default=0.67,
                    help='ratio of used training set')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# semi
parser.add_argument('--semi-pool-fold', type=int, default=5,
                    help='semi pool fold ')
parser.add_argument('--semi-iter', type=int, default=5,
                    help='semi iter')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
parser.add_argument('--bio-mean', type=float, default=0.0,
                    help='biological noise mean')
parser.add_argument('--bio-std-dev', type=float, default=0.1,
                    help='biological noise standard deviation')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# data1: top 195 are TF
# data3: top 334 are TF
# data4: top 333 are TF
dreamTFdict={}
dreamTFdict['data1']=195
dreamTFdict['data3']=334
dreamTFdict['data4']=333

def add_biological_noise(data, mean, std_dev):
    noise = np.random.normal(mean, std_dev, data.shape)
    return data + noise

bio_mean = args.bio_mean
bio_std_dev = args.bio_std_dev

# GeneNet Training / Testing Data
if args.data_name is not None:
    # Load dataset
    dataNet_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.data_name)), allow_pickle=True)
    dataGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.data_name)), allow_pickle=True)
    dataNet_agent0 = np.load(args.file_dir+'/data/dream/'+args.data_name+'_pmatrix_'+str(args.pearson_net)+'.npy', allow_pickle=True).tolist()
    dataNet_agent1 = np.load(args.file_dir+'/data/dream/'+args.data_name+'_mmatrix_'+str(args.mutual_net)+'.npy', allow_pickle=True).tolist()
    allx = dataGroup.toarray().astype('float32')

    # Process features
    dataAttributes = genenet_attribute(allx, dreamTFdict[args.data_name])

    # Split dataset into training and testing
    total_pos, total_neg, _, _ = sample_neg_semi_TF(dataNet_ori, 0.0, TF_num=dreamTFdict[args.data_name], max_train_num=args.max_train_num, semi_pool_fold=args.semi_pool_fold)

    train_pos_0, test_pos_0 = train_test_split(total_pos[0], train_size=args.training_ratio)
    train_pos_1, test_pos_1 = train_test_split(total_pos[1], train_size=args.training_ratio)
    train_neg_0, test_neg_0 = train_test_split(total_neg[0], train_size=args.training_ratio)
    train_neg_1, test_neg_1 = train_test_split(total_neg[1], train_size=args.training_ratio)

    train_pos = (train_pos_0, train_pos_1)
    test_pos = (test_pos_0, test_pos_1)
    train_neg = (train_neg_0, train_neg_1)
    test_neg = (test_neg_0, test_neg_1)

    # Apply biological noise to training dataset
    testAttributes = dataAttributes
    trainAttributes = add_biological_noise(dataAttributes, bio_mean, bio_std_dev)

# train_pos, test_pos, train_neg, test_neg is the relevant data for you
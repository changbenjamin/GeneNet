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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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



# Train and apply classifier
Atrain_agent0 = dataNet_agent0.copy()  # the observed network
Atrain_agent1 = dataNet_agent1.copy()

num_nodes = Atrain_agent0.shape[0]

assert all(i < num_nodes for pos_neg in [train_pos, train_neg, test_pos, test_neg] for tup in pos_neg for i in tup), "Node index out of range"


train_node_information = None
test_node_information = None
if args.use_embedding:
    train_embeddings_agent0 = generate_node2vec_embeddings(Atrain_agent0, args.embedding_dim, True, train_neg)
    train_node_information_agent0 = train_embeddings_agent0

    train_embeddings_agent1 = generate_node2vec_embeddings(Atrain_agent1, args.embedding_dim, True, train_neg)
    train_node_information_agent1 = train_embeddings_agent1

if args.use_attribute and trainAttributes is not None:
    if train_node_information is not None:
        train_node_information_agent0 = np.concatenate([train_node_information_agent0, trainAttributes], axis=1)
        train_node_information_agent1 = np.concatenate([train_node_information_agent1, trainAttributes], axis=1)
    else:
        train_node_information_agent0 = trainAttributes
        train_node_information_agent1 = trainAttributes

# The test dataset should use the same observed network as the training dataset
Atest_agent0 = Atrain_agent0.copy()
Atest_agent1 = Atrain_agent1.copy()

# Mask test links
Atest_agent0[test_pos[0], test_pos[1]] = 0
Atest_agent0[test_pos[1], test_pos[0]] = 0
Atest_agent1[test_pos[0], test_pos[1]] = 0
Atest_agent1[test_pos[1], test_pos[0]] = 0

# Extract test attributes
if args.use_attribute and testAttributes is not None:
    if test_node_information is not None:
        test_node_information_agent0 = np.concatenate([train_node_information_agent0[:len(testAttributes)], testAttributes], axis=1)
        test_node_information_agent1 = np.concatenate([train_node_information_agent1[:len(testAttributes)], testAttributes], axis=1)
    else:
        test_node_information_agent0 = testAttributes
        test_node_information_agent1 = testAttributes

train_graphs_agent0, test_graphs_agent0, max_n_label_agent0 = extractLinks2subgraphs(Atrain_agent0, Atest_agent0, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent0, test_node_information_agent0)
train_graphs_agent1, test_graphs_agent1, max_n_label_agent1 = extractLinks2subgraphs(Atrain_agent1, Atest_agent1, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent1, test_node_information_agent1)

print('# train: %d, # test: %d' % (len(train_graphs_agent0), len(test_graphs_agent0)))

def compute_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

# Function to save ROC curve plot
def save_roc_curve(y_true, y_score, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(filename)

# Generate 
trueList=[]
for i in range(len(test_pos[0])):
    trueList.append(1)
for i in range(len(test_neg[0])):
    trueList.append(0)

ensembleProb=[]

#DGCNN as the graph classifier
def DGCNN_classifer(train_graphs, test_graphs, train_node_information, max_n_label, set_epoch=50, eval_flag=True):
    # DGCNN configurations
    cmd_args.gm = 'DGCNN'
    cmd_args.sortpooling_k = 0.6
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu'
    cmd_args.num_epochs = set_epoch
    cmd_args.learning_rate = 1e-4
    cmd_args.batch_size = 50
    cmd_args.printAUC = True
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.attr_dim = 0
    if train_node_information is not None:
        cmd_args.attr_dim = train_node_information.shape[1]
    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss, train_neg_idx, train_prob_results = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        test_loss=[]
        test_neg_idx=[]
        test_prob_results=[]
        if eval_flag:
            classifier.eval()
            test_loss, test_neg_idx, test_prob_results = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))
    
    return test_loss, train_neg_idx, test_neg_idx, train_prob_results, test_prob_results


def evaluateResults(control_prob, test_pos, trueList):
    tp = len(test_pos[0])
    tn = 0
    fn = 0
    fp = len(control_prob) - len(test_pos[0])
    accuracy = (tp + tn) / len(control_prob)
    
    print("Control Classifier: Accuracy tp fn tn fp")
    print(f"{accuracy}\t{tp}\t{fn}\t{tn}\t{fp}\n")

    # Calculate AUC for the control classifier
    control_prob_zero = [0] * len(control_prob)
    control_auc = roc_auc_score(trueList, control_prob_zero)
    print(f"Control Classifier AUC: {control_auc}")

    # Save the ROC curve plot for the ensemble results
    filename = f'{bio_mean:.2f}_{bio_std_dev:.2f}_on_roc_curve.png'
    save_roc_curve(trueList, control_prob_zero, filename)

    return str(accuracy)

def all_link_classifier(test_graphs):
    num_test_samples = len(test_graphs)
    # Generate an array of all zeros (no link) for the given number of test samples
    predictions = [1] * num_test_samples
    return predictions

# Generate control predictions
control_predictions = all_link_classifier(test_graphs_agent0)
control_prob = [0 if pred == 1 else 1 for pred in control_predictions]

# Call the DGCNN classifier
test_loss, train_neg_idx, test_neg_idx, train_prob_results, test_prob_results = DGCNN_classifer(train_graphs_agent0, test_graphs_agent0, train_node_information_agent0, max_n_label_agent0)

# Get the test_prob_agent0 and test_prob_agent1 as required by the evaluateResults function
test_prob_agent0 = test_prob_results
test_prob_agent1 = control_prob

# Evaluate the ensemble performance
result = evaluateResults(control_prob, test_pos, trueList)
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys, importlib
from time import time
import argparse
import itertools
from dataset import *
import numpy as np
import copy
import torch.optim as optim
from wasserstein import calculate_2_wasserstein_dist
from torchvision.utils import save_image
import torch.utils.data as data_utils
import random
from util import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from  torch.distributions import multivariate_normal as db
def cov(X):
    N,M = X.shape
    cova = torch.zeros((M, M))
    for i in range(M):
        mean_i = torch.sum(X[:, i]) / N
        for j in range(M):
            mean_j = torch.sum(X[:, j]) / N
            cova[i, j] = torch.sum((X[:, i] - mean_i) * (X[:, j] - mean_j)) / (N - 1)
    return cova

def mscatter(x,y, ax=None, m=None, c=None,label=None):
        import matplotlib.markers as mmarkers
        fig, ax = plt.subplots()
        for i in range(len(x)):
            sc = ax.scatter(x[i],y[i],color=c[i])
            if (m[i] is not None):
                paths = []
                for marker in m[i]:
                    if isinstance(marker, mmarkers.MarkerStyle):
                        marker_obj = marker
                    else:
                        marker_obj = mmarkers.MarkerStyle(marker)
                    path = marker_obj.get_path().transformed(
                                marker_obj.get_transform())
                    paths.append(path)
                sc.set_paths(paths)
        return sc, ax

parser = argparse.ArgumentParser(description='DG')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--n_classes', type=int, default=7)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--n_domains', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=8)
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--gan_path', type=str, default='saved/stargan_model/')
parser.add_argument('--target_domain', type=int, default=0)
parser.add_argument('--model', type=str, default='dirt6')
parser.add_argument('--nsamples', type=int, default=10)
parser.add_argument('--base', type=str, default='resnet18')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--outpath', type=str, default='./saved/',  help='where to save')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

best_model = importlib.import_module('models2.'+args.model).Model(args,args.hidden_dim, args.base).to(device)
best_model.load_state_dict(torch.load('saved/{}_{}_target{}_seed{}_nsample{}_alp{}.pt'.format(args.dataset,args.model,args.target_domain,args.seed,args.nsamples,args.alpha)))
#best_model.load_state_dict(torch.load('saved/{}_{}_target{}_seed{}_nsample{}.pt'.format(args.dataset,args.model,args.target_domain,args.seed,args.nsamples)))
#best_model.load_state_dict(torch.load('saved/{}_{}_target{}_seed{}.pt'.format(args.dataset,args.model,args.target_domain,args.seed)))

best_model.eval()

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

train_dataset, val_dataset, test_dataset = get_datasets(args.data_dir,args.dataset,[2],val=0.1)
entire = data_utils.DataLoader(train_dataset, num_workers=1, batch_size=len(train_dataset), shuffle=False)
t = time()
for i,batch in enumerate(entire):
    images_e, labels_e, domains_e = batch
print(time()-t)
label_dictionary = dict()
for i in range(args.num_classes):
    idx = labels_e == i
    label_dictionary[i] = np.where(idx)
domain_dictionary = dict()
for i in range(args.n_domains):
    idx = domains_e == i
    domain_dictionary[i] = np.where(idx)
ld_dictionary = {('{}{}'.format(k1,k2)):np.intersect1d(label_dictionary[k1][0],domain_dictionary[k2][0]) for k1,v1 in label_dictionary.items() for k2,v2 in domain_dictionary.items()}


labels0 = np.random.choice(np.arange(0,args.num_classes),100)
domains0 = np.zeros((100,1), dtype=np.int8)
domains1 = np.ones((100,1), dtype=np.int8)
domains2 = np.ones((100,1), dtype=np.int8)*2

inds0 = []
inds1 = []
inds2 = []

for i in range(len(labels0)):
    inds0.extend(np.random.choice(ld_dictionary['{}{}'.format(labels0[i],domains0[i][0])],1,replace=False))
    inds1.extend(np.random.choice(ld_dictionary['{}{}'.format(labels0[i],domains1[i][0])],1,replace=False))
    inds2.extend(np.random.choice(ld_dictionary['{}{}'.format(labels0[i],domains2[i][0])],1,replace=False))

input_d0 = DatasetDomain(images_e[inds0], labels_e[inds0], domains_e[inds0])
input_d0_loader = torch.utils.data.DataLoader(input_d0, batch_size=len(input_d0),shuffle=False)

input_d1 = DatasetDomain(images_e[inds1], labels_e[inds1], domains_e[inds1])
input_d1_loader = torch.utils.data.DataLoader(input_d1, batch_size=len(input_d1),shuffle=False)

input_d2 = DatasetDomain(images_e[inds2], labels_e[inds2], domains_e[inds2])
input_d2_loader = torch.utils.data.DataLoader(input_d2, batch_size=len(input_d2),shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(input_d2),shuffle=False)

total = [input_d0_loader, input_d1_loader, input_d2_loader]

domain_pair_list = list(itertools.combinations(total,2))

for pairs in domain_pair_list:
    input_d_loader, target_d_loader = pairs

    for ir, batch in enumerate(input_d_loader):
        x1,lab1,dom1 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    for ir, batch in enumerate(target_d_loader):
        x2,lab2,dom2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    z1 = F.relu(best_model.base(x1))
    z2 = F.relu(best_model.base(x2))
    pred1 = best_model.out_layer(z1).argmax(1)
    pred2 = best_model.out_layer(z2).argmax(1)
    result = calculate_2_wasserstein_dist(z1, z2)
    print("%.2f" % result)

for source in total:

    for ir, batch in enumerate(source):
        x1,lab1,dom1 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    for ir, batch in enumerate(test_loader):
        x2,lab2 = batch[0].to(device), batch[1].to(device)
        break
    
    z1 = F.relu(best_model.base(x1))
    z2 = F.relu(best_model.base(x2))
    result = calculate_2_wasserstein_dist(z1, z2)
    print("%.2f" % result)
    



    '''cov1 = cov(z1)
    cov2 = cov(z2)
    mean1 = torch.mean(z1,dim=0)
    mean2 = torch.mean(z2,dim=0)

    cov1.add_(torch.eye(256)*1e-5)
    cov2.add_(torch.eye(256)*1e-5)

    cov1 = cov1.to(device)
    cov2 = cov2.to(device)

    p = db.MultivariateNormal(mean1, cov1)
    q = db.MultivariateNormal(mean2, cov2)

    print(torch.distributions.kl.kl_divergence(p, q))

    L1, V1 = torch.linalg.eigh(cov1)
    L2, V2 = torch.linalg.eigh(cov2)
    L1 = torch.where(L1 < 0, -L1, L1)
    L2 = torch.where(L2 < 0, -L2, L2)

    cov1 = V1 @ torch.diag(L1) @ V1.T
    cov2 = V2 @ torch.diag(L2) @ V2.T
    '''


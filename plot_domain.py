import sys, importlib
from time import time
import argparse
from dataset import *
import numpy as np
import copy
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils
import random
from util import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

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
parser.add_argument('--gan_path', type=str, default='saved/stargan_model20/')
parser.add_argument('--target_domain', type=int, default=0)
parser.add_argument('--model', type=str, default='dirt2')
parser.add_argument('--nsamples', type=int, default=1)
parser.add_argument('--alp', type=float, default=1)
parser.add_argument('--val_stop', type=int, default=0)
parser.add_argument('--base', type=str, default='resnet18')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--outpath', type=str, default='./saved/',
                    help='where to save')
#parser.add_argument('--load_path', type=str, default='inds_500')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

best_model = importlib.import_module('models2.'+args.model).Model(args,args.hidden_dim).to(device)
best_model.load_state_dict(torch.load('saved/{}_{}_target{}_seed{}_nsample{}_alp{}.0.pt'.format(args.dataset,args.model,args.target_domain,args.seed,args.nsamples,args.alp)))
best_model.eval()

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

train_dataset, val_dataset, test_dataset = get_datasets(args.data_dir,args.dataset,[args.target_domain],val=0.1)
entire = data_utils.DataLoader(train_dataset, num_workers=1, batch_size=len(train_dataset), shuffle=False)
t = time()
for i,batch in enumerate(entire):
    images_e, labels_e, domains_e = batch
print(time()-t)

label_dictionary = dict()
for i in range(args.num_classes):
    idx = labels_e.detach().cpu() == i
    label_dictionary[i] = np.where(idx)
domain_dictionary = dict()
for i in range(args.n_domains):
    idx = domains_e.detach().cpu() == i
    domain_dictionary[i] = np.where(idx)
ld_dictionary = {('{}{}'.format(k1,k2)):np.intersect1d(label_dictionary[k1][0],domain_dictionary[k2][0]) for k1,v1 in label_dictionary.items() for k2,v2 in domain_dictionary.items()}
  
markers = np.array(["o" , "v" , "*", "+"])
colors = np.array([ '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2','#ff7f0e'])
#'#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
for l in range(6):
    labels = np.ones((498,1),dtype='int32')*l
    dom = np.array([0, 1, 2])
    domains = np.tile(dom, 166)

    inds0 = []
    for i in range(len(labels)):
        inds0.extend(np.random.choice(ld_dictionary['{}{}'.format(labels[i][0],domains[i])],1))

    plot_z0i_dataset = DatasetDomain(images_e[inds0], labels_e[inds0], domains_e[inds0])
    plot_z0i_loader = torch.utils.data.DataLoader(plot_z0i_dataset, batch_size=len(plot_z0i_dataset),shuffle=False)


    for ir, batch in enumerate(plot_z0i_loader):
        xi,labi,domi = batch[0].to(device), batch[1].to(device), batch[2].to(device) 

    zi = F.relu(best_model.base(xi))
    pred = best_model.out_layer(zi).argmax(1)
    enc_1 = zi.reshape(-1,256)

    pca = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    results = pca.fit_transform(enc_1.detach().cpu().numpy())
    np.savetxt('y_act.txt',labi.detach().cpu().numpy())
    np.savetxt('y_pred.txt',pred.detach().cpu().numpy())
    print(np.mean( labi.detach().cpu().numpy() != pred.detach().cpu().numpy() ))
    pred = pred.to(torch.int32)
    ind = pred.detach().cpu().numpy()
    labi = labi.detach().cpu().numpy()
    scatter, ax =mscatter(results[:,0],results[:,1], ax=None, c=colors[ind], m=markers[domi.detach().cpu().numpy()])
    all_labels = ['0','1','2','3','4','5','6']
    label_row = [all_labels[l]]
    label_column = ['C','P','S']#,'Interp']
    rows = [mpatches.Patch(color=colors[l]) for i in range(1)]
    columns = [plt.plot([], [], markers[i], markerfacecolor='w',
                        markeredgecolor='k')[0] for i in range(args.n_domains+1)]

    plt.legend(rows + columns, label_row + label_column, loc="best", ncol=6, fontsize='xx-small')
    plt.savefig('pacs_500_{}_{}_{}_{}_{}_P.png'.format('mixup',args.target_domain, args.nsamples,args.seed, l),bbox_inches='tight')

    plt.figure()
    scatter, ax =mscatter(results[:,0],results[:,1], ax=None, c=colors[labi], m=markers[domi.detach().cpu().numpy()])
    plt.legend(rows + columns, label_row + label_column, loc="best", ncol=6, fontsize='xx-small')
    plt.savefig('pacs_500_{}_{}_{}_{}_{}_NP.png'.format('mixup',args.target_domain, args.nsamples,args.seed, l),bbox_inches='tight')
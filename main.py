import numpy as np
import wandb
from torch.utils import data
from torchvision import datasets, transforms
import torch
import random
from torch import nn, optim
import os, pdb, importlib
from tqdm import tqdm
from time import time
from util import AverageMeter
from torchvision.utils import save_image
from dataset import *
import argparse
parser = argparse.ArgumentParser(description='DG')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--n_classes', type=int, default=7)
parser.add_argument('--n_domains', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--alp', type=float, default=1.0)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=8)
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--gan_path', type=str, default='saved/stargan_model/')
parser.add_argument('--target_domain', type=int, default=3)
parser.add_argument('--model', type=str, default='dirt0')
parser.add_argument('--nsamples', type=int, default=1)
parser.add_argument('--val_stop', type=int, default=0)
parser.add_argument('--base', type=str, default='resnet18')
flags = parser.parse_args()

'''wandb.config = {
  "learning_rate":flags.lr,
  "epochs": flags.epochs,
  "batch_size": flags.batch_size
}'''

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

device = 'cuda'
# set seed
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.cuda.manual_seed(flags.seed)
torch.cuda.manual_seed_all(flags.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# load models
model = importlib.import_module('models2.'+flags.model).Model(flags,flags.hidden_dim,flags.base).to(device)
#wandb.watch(model, log_freq=100)
optim = torch.optim.SGD(model.parameters(),lr=flags.lr,weight_decay=flags.weight_decay,momentum=0.9)

#load data
train_dataset, val_dataset, test_dataset = get_datasets(flags.data_dir,flags.dataset,[flags.target_domain],val=0.1)

train_loader = data.DataLoader(train_dataset, 
                                num_workers=1, batch_size=flags.batch_size, 
                                shuffle=True, drop_last=True)
val_loader = data.DataLoader(val_dataset, 
                                num_workers=1, batch_size=flags.batch_size, 
                                shuffle=False)
test_loader = data.DataLoader(test_dataset, 
                                num_workers=1, batch_size=flags.batch_size, 
                                shuffle=False)

entire = data.DataLoader(train_dataset, num_workers=1, batch_size=len(train_dataset), shuffle=False)
t = time()
for i,batch in enumerate(entire):
    images_e, labels_e, domains_e = batch
print(time()-t)
label_dictionary = dict()
for i in range(flags.num_classes):
    idx = labels_e == i
    label_dictionary[i] = np.where(idx)
domain_dictionary = dict()
for i in range(flags.n_domains):
    idx = domains_e == i
    domain_dictionary[i] = np.where(idx)
ld_dictionary = {('{}{}'.format(k1,k2)):np.intersect1d(label_dictionary[k1][0],domain_dictionary[k2][0]) for k1,v1 in label_dictionary.items() for k2,v2 in domain_dictionary.items()}
def to_device(data):
    for i in range(len(data)):
        data[i] = data[i].to(device)
    return data
best_by_val = 0
best_val_acc = 0.0
best_val_loss = float('inf')
for epoch in range(flags.epochs):
    print('Epoch {}: Best by val {}'.format(epoch,best_by_val))
    lossMeter = AverageMeter()
    regMeter = AverageMeter()
    correctMeter = AverageMeter()
    model.train()
    itere = 0
    for data in tqdm(train_loader,ncols=75,leave=False):
        images,labels,domains = data[0], data[1], data[2]
        domain_list = np.arange(flags.n_domains)
        domain_list_ = np.tile(domain_list,(len(domains),1))
        m,n = domain_list_.shape
        domain_removed = domain_list_[np.arange(n) != domains.numpy()[:,None]].reshape(m,-1)
        new_domains1 = np.take_along_axis(domain_removed, np.random.randint(0,len(domain_list)-1,len(domains)).reshape(-1,1), axis=1)
        dr_2 = domain_list_[np.logical_and(np.arange(n) != new_domains1.reshape(-1)[:,None], np.arange(n) != domains.numpy()[:,None])].reshape(m,-1)
        new_domains2 = np.take_along_axis(dr_2, np.random.randint(0,len(domain_list)-2,len(domains)).reshape(-1,1), axis=1)
        inds = []
        for i in range(len(labels)):
            #print(labels[i],new_domains1[i])
            inds.extend(np.random.choice(ld_dictionary['{}{}'.format(labels[i],new_domains1[i][0])],1))
        inds2 = []
        for i in range(len(labels)):
            #print(labels[i],new_domains2[i])
            inds2.extend(np.random.choice(ld_dictionary['{}{}'.format(labels[i],new_domains2[i][0])],1))

        inv_dataset = DatasetDomain(images_e[inds], labels_e[inds], domains_e[inds])
        invar_loader = torch.utils.data.DataLoader(inv_dataset, batch_size=len(inv_dataset),shuffle=False)

        inv_dataset2 = DatasetDomain(images_e[inds2], labels_e[inds2], domains_e[inds2])
        invar_loader2 = torch.utils.data.DataLoader(inv_dataset2, batch_size=len(inv_dataset2),shuffle=False)

        images,labels,domains = images.to(device),labels.to(device),domains.to(device)
    
        for i2, batch2 in enumerate(invar_loader):
            images2,labels2,domains2 = batch2[0].to(device), batch2[1].to(device), batch2[2].to(device)
        
        for i3, batch3 in enumerate(invar_loader2):
            images3,labels3,domains3 = batch3[0].to(device), batch3[1].to(device), batch3[2].to(device)
        #print(labels,labels2)
        #print(domains,domains2)
        data = to_device(data)
        itere = itere + 1
        loss, reg, correct = model(images,labels,images2,images3,domains,domains2,domains3,epoch)
        obj = loss + reg
        #wandb.log({"loss": obj})
        optim.zero_grad()
        obj.backward()
        optim.step()

        lossMeter.update(loss.item(),data[0].shape[0])
        regMeter.update(reg.item(),data[0].shape[0])
        correctMeter.update(correct.item(),data[0].shape[0])
        del loss, reg, correct
    print('>>> Training: Loss ', lossMeter,', Reg ', regMeter,', Acc ', correctMeter)

    vallossMeter = AverageMeter()
    valregMeter = AverageMeter()
    valcorrectMeter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(val_loader,ncols=75,leave=False):
            x,y = x.to(device), y.to(device)
            loss, reg, correct = model(x,y)

            vallossMeter.update(loss.item(),x.shape[0])
            valregMeter.update(reg.item(),x.shape[0])
            valcorrectMeter.update(correct.item(),x.shape[0])
            del loss, reg, correct
    print('>>> Val: Loss ', vallossMeter,', Reg ', valregMeter,', Acc ', valcorrectMeter)

    testlossMeter = AverageMeter()
    testregMeter = AverageMeter()
    testcorrectMeter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(test_loader,ncols=75,leave=False):
            x,y = x.to(device), y.to(device)
            loss, reg, correct = model(x,y)

            testlossMeter.update(loss.item(),x.shape[0])
            testregMeter.update(reg.item(),x.shape[0])
            testcorrectMeter.update(correct.item(),x.shape[0])
            del loss, reg, correct
    print('>>> Test: Loss ', testlossMeter,', Reg ', testregMeter,', Acc ', testcorrectMeter)

    if vallossMeter.float()<best_val_loss and valcorrectMeter.float()>best_val_acc and epoch>flags.val_stop:
        best_by_val = testcorrectMeter.float()
        torch.save(model.state_dict(),'saved/{}_{}_target{}_seed{}_nsample{}_alp{}.pt'.format(flags.dataset,flags.model,flags.target_domain,flags.seed,flags.nsamples,flags.alp))
    if vallossMeter.float()<best_val_loss and epoch>flags.val_stop:
        best_val_loss = vallossMeter.float()
    if valcorrectMeter.float()>best_val_acc and epoch>flags.val_stop:
        best_val_acc = valcorrectMeter.float()



    

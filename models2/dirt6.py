import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
import pdb
from torchvision.utils import save_image
from models.stargan import load_stargan
import argparse
device = 'cuda'
def interpolate(z_1, z_2, start, end, mid):
    z_list = []
    for weight in torch.arange(start,end,mid).to(device):
        z_list.append(torch.lerp(z_1, z_2, weight))
    my_tensor = torch.vstack([ten for ten in z_list])
    return my_tensor
class _interp_branch(nn.Module):
    '''
    one branch of the interpolator network
    '''

    def __init__(self, in_channels, out_channels):
        super(_interp_branch, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
                                   #nn.ReLU(True),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels*4, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)
 
class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='resnet18'):
        super(Model, self).__init__(hidden_dim,base)

        #self.w1 = nn.Parameter(torch.ones([hidden_dim,7]))
        #torch.nn.init.kaiming_normal_(self.w1)
        self.hidden_dim = hidden_dim
        self.out_layer = nn.Linear(hidden_dim,config.num_classes)
        self.trans = load_stargan(config.gan_path + '{}_domain{}_last-G.ckpt'.format(config.dataset,config.target_domain))
        self.trans.eval()
        self.alpha = config.alpha
        self.intermod = _interp_branch(self.hidden_dim,self.hidden_dim) 
        self.batch_size = config.batch_size 
    
    def forward(self,x,y,x2=None,x3=None,d=None,d2=None, d3=None,epoch=None):
        z = F.relu(self.base(x))
        logits = self.out_layer(z)
        loss = F.cross_entropy(logits,y)
        correct = (torch.argmax(logits,1)==y).sum().float()/x.shape[0]
        reg =loss.new_zeros([1])
        if self.training:
            with torch.no_grad():
                rand_idx = torch.randperm(d.size(0))
                d_new = d[rand_idx]
                d_onehot = d.new_zeros([d.shape[0],3])
                d_onehot.scatter_(1, d[:,None], 1)
                d_new_onehot = d.new_zeros([d.shape[0],3])
                d_new_onehot.scatter_(1, d_new[:,None], 1)
                x_new = self.trans(x,d_onehot,d_new_onehot)
            z_new = F.relu(self.base(x_new))
            z1 = F.relu(self.base(x))
            z2 = F.relu(self.base(x2))
            start = 0
            mid = 0.01
            end = 1.01
            z_list = []
            samples = torch.arange(start,end,mid)
            iterator = samples
            #iterator = torch.where(samples < 0.5, samples * 0.5, 1 - (1 - samples) * 0.5)
            #iterator = samples * 0.2 + 0.4
            for v in iterator.to(device):
                z_list.append(z1+v*self.intermod((z2-z1).reshape(len(x),-1, 1, 1)).reshape(len(x),-1))
            zi = torch.vstack([ten for ten in z_list])
            perm = torch.randperm(zi.size(0))
            idx1 = perm[:len(x)]
            #idx2 = perm[-self.batch_size:]
            zi1 = zi[idx1]
            #zi2 = zi[idx2]
            zj = zi[-len(x):]
            reg = self.alpha * F.cross_entropy(self.out_layer(zi),torch.cat(101*[y],dim=0))+ self.alpha * F.mse_loss(zj,z2) + F.mse_loss(z_new, z)
            #+ F.mse_loss(sum(((z1 - z2)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim),(sum(((zi1 - z1)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim) + sum(((zi1 - z2)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim)))
            #print(F.mse_loss(sum(((z1 - z2)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim),(sum(((zi1 - z1)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim) + sum(((zi1 - z2)**2).reshape(self.batch_size*self.hidden_dim))/(self.batch_size*self.hidden_dim))))
        return loss,reg,correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--dataset', type=str, default='VLCS')
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--n_domains', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--gan_path', type=str, default='saved/stargan_model/')
    parser.add_argument('--target_domain', type=int, default=3)
    parser.add_argument('--model', type=str, default='dirt2')
    parser.add_argument('--nsamples', type=int, default=100)
    parser.add_argument('--base', type=str, default='alexnet')
    flags = parser.parse_args()
    N, in_channels, H, W = 16, 3, 224, 224
    z_dim = 256
    model = Model(flags, 256, 'resnet18')
    model.train()
    x = torch.randn((N, in_channels, H, W))
    y = torch.tensor([1, 2, 3, 4, 3, 2, 0, 0, 4, 2, 1, 1, 0, 3, 1, 4], dtype=torch.int64)
    output, loss, reg = model(x,y,torch.randn((N, in_channels, H, W)))
    

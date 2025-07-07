import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)


def compute_dicard_rate(Perms):
    discard_rates = []
    for perm in Perms[1:]:
        rate = (perm[:, 1:] > 0).type(dtype).sum(1) / perm.size()[1]
        discard_rates.append(rate.mean().data.cpu().numpy())
    return discard_rates


def compute_accuracy(output, target):
    # convert to numpy arrays
    tar = target.data.cpu().numpy()
    out = output.data.cpu().numpy()
    return np.mean(np.all(np.equal(tar, out[:, 1:]), axis=1))


def compute_miss_rate(output, target):
    tar = target.data.cpu().numpy()
    out = output.data.cpu().numpy()
    
    return np.mean(np.equal(tar, out[:, 1:]))

# Space efficient way of counting size of different groups of identical
# scalars in batched tensors
def unique_scalars_batch(t_in):
    batches, seq_len = t_in.shape

    # number of unique groups
    group_no = int(torch.max(t_in).item())+1

    offsetted_input = t_in + group_no * torch.arange(batches)[:,None]

    flat_counts = torch.bincount(offsetted_input.flatten(), minlength = group_no*batches)

    counts = flat_counts.reshape(t_in)

    return counts

# Gets N from Phi, uses summing so can't be used with sparse
def phi_to_N(Phi):
    N = torch.mul(Phi, torch.sum(Phi,dim=1,keepdim = True))
    #print(N.shape)

    return N


# Masks input by phi
def mask_input_phi(phi,input):
        # Phi is B len cats
        # Inp is B len dimensionality
        # Make phis broadcast over inp B len cats d
        inp_masked = input[:,:,None,:].expand(-1,-1,phi.shape[2],-1) 

        # For sparse phi, stacking needed bc unsqueeze doesn't work
        phi = phi.unsqueeze(3)        
        inp_masked = inp_masked * phi
        return inp_masked

# Gets means within categories
def cat_means(phi,input,N):
    
    inp_masked = mask_input_phi(phi,input)


    
    # Size B, length, cats, 2 or 2+hiddenlayers with means of each category
    # Broadcasting with N
    N = N[:,:,:,None]


    means = (inp_masked.sum(1,keepdim=True) / N)

    #print("Means",means.shape)
    return means

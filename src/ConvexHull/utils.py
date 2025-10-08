import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

def acc_std_dev(output, target):
    # convert to numpy arrays
    if isinstance(output,torch.Tensor):
        target = target.data.cpu().numpy()
        output = output.data.cpu().numpy()[:, 1:]

def compute_miss_rate(output, target, lengths): 
    output, target = input_pre(output, target, lengths)

    sample_acc = []
    
    for i, (ti, oi) in enumerate(zip(target, output)):
        sample_acc.append(np.sum(np.equal(oi, ti))/lengths[i])

    return np.mean(sample_acc)

def input_pre(output, target, lengths):
    """Converts to numpy, trims the output pad, then trims the padding to the sequence length"""
    if isinstance(output,torch.Tensor):
        target = target.data.cpu().numpy()
        output = output.data.cpu().numpy()[:, 1:]
    
    def strip_pad(input_arr):
        dest_list = []
        for perm,length in zip(input_arr,lengths):
            dest_list.append(perm[:length])

        return dest_list

    output_list = strip_pad(output)
    target_list = strip_pad(target)

    return output_list, target_list

def compute_inclusion_exclusion_acc(output, target, lengths):
    output, target = input_pre(output, target, lengths)

    inclusion_acc = compute_inclusion_acc(output, target)
    exclusion_acc = compute_exclusion_acc(output, target, lengths)

    no_included = np.mean([np.sum(t != 0) for t in target])
    no_excluded = np.mean([np.sum(t == 0) for t in target])

    inclusion_exclusion_acc = (inclusion_acc * no_included + exclusion_acc * no_excluded) / (no_included + no_excluded)
    #inclusion_exclusion_acc = inclusion_acc+exclusion_acc

    return inclusion_exclusion_acc

def compute_inclusion_acc(output, target, lengths=None):
    """Calculates the proportion of vertices that are supposed to be included that are"""
    if lengths:
        output, target = input_pre(output, target, lengths)

    batch_size = len(target)

    # Finds the number of correctly included vertices for each hull in the batch
    correct_inclusions = np.zeros(batch_size)
    target_inclusions = np.zeros(batch_size)

    for i, (ti, oi) in enumerate(zip(target, output)):
        correct_inclusions[i] = sum(np.in1d(ti[ti != 0],oi))
        target_inclusions[i]  = max(np.sum(ti != 0),1e-6)

    return np.mean(correct_inclusions / target_inclusions)


def compute_exclusion_acc(output, target, lengths):
    """Calculates the proportion of vertices that are supposed to be excluded that are"""
    output, target = input_pre(output, target, lengths)

    # Inverts the target sequence and output sequence to only capture those that were at zeros
    def invert_inclusions(input_perm):
        out_list = []
        
        # Makes a permutation of the same length of the input and removes elements corresponding to the input
        for i, perm in enumerate(input_perm):
            out_list.append(np.arange(lengths[i])+1)
            
            included_inds = perm[perm != 0]-1

            # Only used for comparing inclusions so doesn't matter if there are gaps of zeros in between active indices
            out_list[-1][included_inds.astype(np.int64)] = 0

        return out_list

    output = invert_inclusions(output)
    target = invert_inclusions(target)

    return compute_inclusion_acc(output, target)

def compute_order_acc(output, target, lengths=None):
    if lengths:
        output, target = input_pre(output, target, lengths)

    correct_orders = np.zeros(len(target))
    for i, (ti, oi) in enumerate(zip(target, output)):
        included = np.in1d(ti,oi)
        correct_orders[i] = sum(np.equal(ti,oi)) / sum(included)

    return np.mean(correct_orders)
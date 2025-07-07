import numpy as np
from scipy import sparse
import csv
from scipy.spatial import ConvexHull

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Pytorch requirements
import unicodedata
import string
import re
import random
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

dtype = torch.float32

device = torch.device('cuda')

torch.cuda.manual_seed(0)


class SplitLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(SplitLayer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        W1 = nn.Parameter(torch.randn((hidden_size + input_size,
                          hidden_size),dtype=dtype))
        self.register_parameter('W1', W1)
        W2 = nn.Parameter(torch.randn((hidden_size + input_size,
                          hidden_size),dtype=dtype))
        self.register_parameter('W2', W2)
        b = nn.Parameter(torch.randn(hidden_size,dtype=dtype))
        self.register_parameter('b', b)

    
    def forward(self, input_n, hidden, phi, nh):
        self.batch_size = input_n.size()[0]

        # Concatenates the normalized input and the last layer
        hidden = torch.cat((hidden, input_n), 2).to(dtype)

        # Group wise sums
        h_conv = torch.div(torch.bmm(phi.to(dtype), hidden), nh)

        #print(hidden.type())

        # Input to the hidden layers
        hidden = hidden.view(-1, self.hidden_size + self.input_size)
        h_conv = h_conv.view(-1, self.hidden_size + self.input_size)

        # h_conv has shape (batch_size, n, hidden_size + input_size)

        #print(hidden.type(),self.W1.type())
        m1 = (torch.mm(hidden, self.W1)
              .view(self.batch_size, -1, self.hidden_size))
        m2 = (torch.mm(h_conv, self.W2)
              .view(self.batch_size, -1, self.hidden_size))
        m3 = self.b.unsqueeze(0).unsqueeze(1).expand_as(m2)

        hidden = torch.sigmoid(m1 + m2 + m3)
        return hidden


class Split(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, n_layers=1):
        super(Split, self).__init__()
        print('Initializing Parameters Split')
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.n = 32
        layers = [SplitLayer(input_size, hidden_size, batch_size)
                  for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.linear_b = nn.Linear(hidden_size, 1, bias=True).type(dtype)

    def forward(self, e, input, mask, scale=0):
        hidden = torch.randn(self.batch_size, self.n, self.hidden_size,device=device,dtype=dtype)
        
        if scale == 0:
            e = torch.zeros(self.batch_size, self.n,device=device,dtype=torch.int64)

        Phi = self.build_Phi(e, mask)
        
        uniques = torch.unique_consecutive(e,dim=1,return_counts=True)[-1].to(dtype)
        N = Phi @ uniques[:,None,:]

        #N = Phi.sum(2)

        N += (N == 0)  # avoid division by zero
        Nh = N.unsqueeze(2).expand(self.batch_size, self.n,
                                   self.hidden_size + self.input_size).type(dtype)
        
        # Normalize inputs, important part!
        mask_inp = mask.unsqueeze(2).expand_as(input)
        input_n = self.Normalize_inputs(Phi, N, input) * mask_inp

        # input_n = input * mask_inp
        for i, layer in enumerate(self.layers):
            hidden = layer(input_n, hidden, Phi, Nh)
        hidden_p = hidden.view(self.batch_size * self.n, self.hidden_size)
        scores = self.linear_b(hidden_p)
        probs = torch.sigmoid(scores).view(self.batch_size, self.n) * mask
        # probs has shape (batch_size, n)
        return scores, probs, input_n, Phi

    def build_Phi(self, e, mask):
        # number of groups
        group_no = int(torch.max(e).item())+1
        # Shape of batch, length, number of groups
        s = (e.shape[0],e.shape[1],group_no)
        Phi = torch.zeros(s,device=device,dtype=dtype)
        
        # Prep for broadcasting
        et = e.unsqueeze(2)
        
        Phi.scatter_(2,et,1)

        """# Make on the CPU expanded, then compress and send to gpu
        e, mask = e.to(torch.device('cpu')), mask.to(torch.device('cpu'))

        e_rows = e.unsqueeze(1).expand(self.batch_size, self.n, self.n)
        e_cols = e.unsqueeze(2).expand(self.batch_size, self.n, self.n)
        Phi = (e_rows == e_cols)

        # mask attention matrix
        mask_rows = mask.unsqueeze(2)
        mask_cols = mask.unsqueeze(1)

        Phi = Phi * mask_rows * mask_cols"""

        # For storage, make Phi sparse
        # Ideal block size for bsr is length/number of groups
        """ group_no = torch.max(e).item()+1
        
        if group_no > 0:
            size = mask.shape[1] / group_no
            print(size)
            Phi = Phi.to_sparse_bsr(int(size))
         """
        return Phi.to(device)

    def Normalize_inputs(self, phis, N, input):
        # phis defines the clusters
        length = phis.size()[1]

        # Phi is B len cats
        # Inp is B len 2
        # Make phis broadcast over inp B len cats 2

        inp_masked = input[:,:,None,:].expand(-1,-1,phis.shape[2],-1) * phis

        # Size B cats 2 with means of each category
        means = inp_masked.sum(1,keepdim=True) / N


        dif = inp_masked - means
        var = (dif * dif).sum(1,keepdim=True) / N
        var += (var == 0).type(dtype)

        masked_norm = (inp_masked - means) / (3 * var.sqrt()) + 0.5

        inp_norm = torch.amax(masked_norm,dim=2)

        return inp_norm

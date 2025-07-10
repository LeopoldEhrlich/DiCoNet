"""Runs various experiments on DCN code to identify properties of the network."""


import matplotlib.pyplot as plt
import os
import argparse

from main import test
from DCN import DivideAndConquerNetwork
from data_generator import Generator
from experiments.settings import DCN_Settings
from experiments.display_splits import SplitVisualizer
from experiments.experiment_utils import create_DCN, create_gen, default_settings, graph_acc_miss

import torch
import numpy as np

def test_overall_scale_invariancy():
    """Shows accuracies of whole forward passes over scales"""
    DCN = create_DCN()
    gen = create_gen(range(1,9))

    accuracies_test, miss_rates = test(DCN, gen, default_settings)
    graph_acc_miss('scale_invar', accuracies_test, miss_rates)


def test_intra_scale_invariancy():
    """Shows accuracies at each level of a network's forward pass"""
    from scipy.spatial import ConvexHull

    def find_target(points):
        ch = ConvexHull(points).vertices
    
        argmin = np.argsort(ch)[0]
        # Moves zeros to the end of the list
        ch = list(ch[argmin:]) + list(ch[:argmin])
        target[:len(ch)] = np.array(ch)
    
        target += 1  


def test_base_case():
    """Tests the accuracy of the base cases of the network"""

    settings = DCN_Settings(num_examples_test = 32_768, batch_size=256)

    DCN = create_DCN(settings)

    gen = Generator(0, settings.num_examples_test,
                    settings.path_dataset, settings.batch_size, scales_test=[1,2])
    
    gen.scales['train'] = []
    
    pair = lambda x: (x,x)

    gen.compute_length = lambda scale, mode: pair(3 if scale == 1 else 6)
    gen.load_dataset('base')

    accuracies_test, miss_rates = test(DCN, gen, settings)
    graph_acc_miss('base_case', accuracies_test, miss_rates)


def graph_split():
    """Creates a figure of various split results"""
    settings = DCN_Settings(num_examples_test=18,batch_size=18)

    DCN = create_DCN(settings)
    gen = create_gen([6],settings=settings)

    out_arr = run_fwd_test(DCN,gen)

    for (Phis, Inputs_N, target, Perms, e, loss, pg_loss, var), input, s in out_arr:
        visualizer = SplitVisualizer(input, e, s)
        visualizer.show()
    

def run_fwd_test(DCN,gen):
    out_arr = []

    with torch.no_grad():
        for s in gen.scales['test']:
            input, tar = gen.get_batch(batch=0, scales=s, mode='test')

            _, length = gen.compute_length(s, mode='test')

            out = DCN(input, tar, length, s, it=0,
                        random_split=False,
                        mode='test', dynamic=True)
            
            Phis, Inputs_N, target, Perms, e, loss, pg_loss, var = out

            out_arr.append((out,input,s))

    return out_arr

if __name__ == "__main__":
    #test_scale_invariancy()
    #test_base_case()
    graph_split()

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

def test_scale_invariancy():
    """Shows accuracies over scales"""
    DCN = create_DCN()
    gen = create_gen(range(1,9))

    accuracies_test, miss_rates = test(DCN, gen, default_settings)
    graph_acc_miss('scale_invar', accuracies_test, miss_rates)

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

    DCN = create_DCN()
    gen = create_gen(range(6,7))

    with torch.no_grad():
        for s in gen.scales['test']:
            input, tar = gen.get_batch(batch=0, scales=s, mode='test')

            _, length = gen.compute_length(s, mode='test')

            out = DCN(input, tar, length, s, it=0,
                        random_split=False,
                        mode='test', dynamic=True)
            
            Phis, Inputs_N, target, Perms, e, loss, pg_loss, var = out

            visualizer = SplitVisualizer(Inputs_N, e, Perms, s)
            visualizer.show()

if __name__ == "__main__":
    #test_scale_invariancy()
    #test_base_case()
    graph_split()

"""Runs various experiments on DCN code to identify scale invariant properties of the network."""
import matplotlib.pyplot as plt

from main import test
from DCN import DivideAndConquerNetwork
from data_generator import Generator
from experiments.settings import DCN_Settings
from experiments.display_splits import SplitVisualizer
from experiments.experiment_utils import *

import numpy as np
from scipy.stats import linregress


default_settings = DCN_Settings(num_examples_test=64*16,batch_size=64)

def test_overall_scale_invariancy():
    """Shows accuracies of whole forward passes over scales"""
    DCN = create_DCN(default_settings)
    gen = create_gen(range(1,7),default_settings)

    accuracies_test, miss_rates = test(DCN, gen, default_settings)

    print(linregress(list(range(1,7)),miss_rates).pvalue)

    graph_acc_miss('scale_invar', accuracies_test, miss_rates,default_settings.path)



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
    graph_acc_miss('base_case', accuracies_test, miss_rates,default_settings.path)


if __name__ == "__main__":
    test_overall_scale_invariancy()
    #test_base_case()

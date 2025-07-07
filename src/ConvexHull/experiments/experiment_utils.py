"""Utility functions for use in experiments"""

import os
import matplotlib.pyplot as plt
from experiments.settings import DCN_Settings
from data_generator import Generator
from DCN import DivideAndConquerNetwork

default_settings = DCN_Settings(num_examples_test=8,batch_size=8)
hull_dims = 2

def graph_acc_miss(testName, accuracies_test, hit_rates, directory=default_settings.path):
    """Given two accuracy metrics, 
    
    testName - The name of the test this function is used for, for creating output folder

    accuracies_test - list of proportions of convex hulls that were exact matches for the target
    hit_rates - list of average proportion of correctly identified vertecies for over all convex hulls

    directory - where to find the experiment results folder
    """
    out_dir = os.path.join(directory, "experiment_results", testName)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(10, 6))

    # Plot both metrics on the same axes
    xscale = range(1,len(accuracies_test)+1)

    line1, = plt.plot(xscale, accuracies_test, 'b-o', label='Whole Hull Accuracy', linewidth=2, markersize=8)
    line2, = plt.plot(xscale, hit_rates, 'r--s', label='Per-Vertex Accuracy', linewidth=2, markersize=8)

    plt.title('Model Performance vs. Max Depth', fontsize=14, pad=20)    
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(xscale, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(handles=[line1, line2], fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(out_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    
def create_DCN(settings=default_settings):
    """Helper for creating a DCN with a settings object and loading in weights"""

    DCN = DivideAndConquerNetwork(hull_dims, settings.batch_size,
        settings.num_units_merge, settings.rnn_layers,
        settings.grad_clip_merge,
        settings.num_units_split, settings.split_layers,
        settings.grad_clip_split, beta=settings.beta)

    DCN.load_split(settings.load_split)
    DCN.load_merge(settings.load_merge)

    DCN.cuda()
    DCN.batch_size = settings.batch_size
    DCN.merge.batch_size = settings.batch_size
    DCN.split.batch_size = settings.batch_size

    return DCN


def create_gen(scales, settings=default_settings):
    """Creates a dataset for testing DCN on convex hull"""
    gen = Generator(0, settings.num_examples_test,
                    settings.path_dataset, settings.batch_size,scales_test=scales)
    gen.load_dataset()

    return gen
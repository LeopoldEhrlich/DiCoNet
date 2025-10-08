"""Utility functions for use in experiments"""

import os
import matplotlib.pyplot as plt
from experiments.settings import DCN_Settings
from data_generator import Generator
from DCN import DivideAndConquerNetwork
import utils

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

from scipy.spatial import ConvexHull

hull_dims = 2

def graph_acc_miss(testName, accuracies_test, hit_rates, directory):
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

    
def create_DCN(settings):
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


def create_gen(scales, settings):
    """Creates a dataset for testing DCN on convex hull"""
    gen = Generator(0, settings.num_examples_test,
                    settings.path_dataset, settings.batch_size,scales_test=scales)
    gen.load_dataset()

    return gen

def run_fwd_test(DCN,gen):
    out_arr = []

    with torch.no_grad():
        no_batches = int(gen.num_examples_test / gen.batch_size)

        for s in tqdm(gen.scales['test'],leave=False):
            scale_arr = []
            for batch in tqdm(range(no_batches),leave=False):
                input, tar = gen.get_batch(batch=batch, scales=s, mode='test')

                _, length = gen.compute_length(s, mode='test')

                out = DCN(input, tar, length, s, it=0,
                            random_split=False,
                            mode='test', dynamic=True)

                scale_arr.append((out,input,s))
            out_arr.append(scale_arr)
    return out_arr


def get_prediction_metrics(DCN,gen,fwd_test_results=None):
    if fwd_test_results is None:
        fwd_test_results = run_fwd_test(DCN,gen)

    no_batches = int(gen.num_examples_test / gen.batch_size)
    no_scales  = len(gen.scales['test'])

    cat_names = ["Vertex Match Accuracy", "Whole Hull Accuracy", "Inclusion-Exclusion Accuracy", "Order Accuracy", "Exclusion Accuracy", "Inclusion Accuracy"]

    accuracies = np.zeros((no_scales,len(cat_names)))

    df = pd.DataFrame(accuracies,columns=cat_names)

    for i, scale_test_result in enumerate(fwd_test_results):
        for batch_test_result in scale_test_result:
            lengths = np.sum(batch_test_result[1][batch_test_result[1] != -1], axis=1)

            Perms, target = batch_test_result[0][3],batch_test_result[0][2]

            df["Vertex Match Accuracy"][i] += utils.compute_miss_rate(Perms[-1], target) / no_batches
            df["Whole Hull Accuracy"][i] += utils.compute_accuracy(Perms[-1], target) / no_batches
            df["Inclusion-Exclusion Accuracy"][i] += utils.compute_inclusion_exclusione_acc(Perms[-1], target, lengths) / no_batches
            df["Order Accuracy"][i] += utils.compute_order_acc(Perms[-1], target) / no_batches
            df["Exclusion Accuracy"][i] += utils.compute_exclusion_acc(Perms[-1], target, no_batches, lengths) / no_batches
            df["Inclusion Accuracy"][i] += utils.compute_inclusion_acc(Perms[-1], target, no_batches, lengths) / no_batches

    return df

def convexhull_example(points):
        if len(points) <= 3:
            return points, np.arange(len(points))
        
        target = -1 * np.ones([len(points)])
        ch = ConvexHull(points).vertices

        argmin = np.argsort(ch)[0]

        # Moves zeros to the end of the list
        ch = list(ch[argmin:]) + list(ch[:argmin])
        target[:len(ch)] = np.array(ch)

        target += 1
        return points, target

def make_target(points):
    length = len(points)

    x = -1 * np.ones((length, 2))
    y = np.zeros(length)
        
    x_ex, y_ex = convexhull_example(input)
    x[:length], y[:length] = x_ex, y_ex

def get_global_acc(DCN,gen,fwd_test_results=None):
    if fwd_test_results is None:
        fwd_test_results = run_fwd_test(DCN,gen)
    
    input_grouped = fwd_test_results[1].select(e)

import matplotlib.pyplot as plt

from main import test
from DCN import DivideAndConquerNetwork
from data_generator import Generator
from experiments.settings import DCN_Settings
from experiments.display_splits import SplitVisualizer
from experiments.experiment_utils import *
from utils import compute_exclusion_acc, compute_miss_rate

from matplotlib.patches import Polygon
import matplotlib
#matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

from itertools import pairwise

from scipy.stats import linregress

import numpy as np
import pandas as pd

default_settings = DCN_Settings(num_examples_test=64*1,batch_size=64)

def test_order_accuracy():
    """Compares the vertex accuracy rate to the accuracy caring only about inclusions and the accuracy caring only about order"""
    coop_DCN = create_DCN(default_settings)

    gen = create_gen(range(1,6),settings=default_settings)

    df = get_prediction_metrics(coop_DCN,gen)

    print(df)

    print([linregress(df.index,df[attribute]).pvalue for attribute in df.columns])

def get_group_perm(perm, pts):
    """Helper to map from local net permutation to the corresponding permutation of the group"""
    perm_out = np.zeros(sum(pts[pts != -1]))

    active_pts = pts[perm]
    active_pts = active_pts[active_pts != 0]

    perm_out[:len(active_pts)] = active_pts

    return perm_out

def test_global_local_accumulation():
    DCN = create_DCN(default_settings)
    gen = create_gen(range(5,6), settings=default_settings)
    out = run_fwd_test(DCN, gen)[0]
    
    num_levels = len(out[0][0][3])-1  # Number of permutation levels
    global_acc = np.zeros(num_levels)
    global_acc_ref = np.zeros(num_levels)
    local_exclusion_scores = np.zeros(num_levels)
    local_preservation = np.zeros(num_levels)
    
    for d, scale_test_result in enumerate(out):
        (Phis, Inputs_N, target, Perms, e, loss, pg_loss, var), input, scales = scale_test_result
        Perms = list(reversed(Perms))
        
        # Track the chain of indexes through levels
        current_indexes = None  
        input_bat = Inputs_N[0].cpu().numpy()  # Original input points
        
        for i, (Phi, Perm) in enumerate(zip(Phis, Perms)):
            perms = Perm.data.cpu().numpy()[:,1:]  # Get permutation indexes
            phi = Phi.data.cpu().numpy()
            
            level_acc = 0
            level_acc_ref = 0
            local_exclusion = 0
            total_groups = 0
            pres = 0
            
            for j, hull in enumerate(phi):
                groups = np.unique(hull, axis=1).T.astype(bool)
                groups = groups[np.any(groups, axis=1)]

                previous_perm = None
                
                for k, grp in enumerate(groups):
                    pts = np.where(grp[:,None], input_bat[j], np.ones_like(input_bat[j]) * -1)
                    
                    pred_indexes = get_group_perm(perms[j], pts)

                    # Get the global solution over all the points included in the current scope      
                    pts = pts[pts[:,0] != 0,:]
                    lengths = [len(pts)]
                    global_gt = convexhull_example(pts)[1]

                    if j == 0:
                        breakpoint()

                    level_acc += compute_exclusion_acc(pred_indexes[None,:], global_gt[None,:],lengths)
                    level_acc_ref += compute_miss_rate(pred_indexes[None,:], global_gt[None,:],lengths)
                    

                    # Calculating the proportion of points preserved from one scope to the next
                    if i == 0:
                        pres += 1
                    
                    else:
                        pres += sum(pred_indexes != 0)/sum(previous_perm != 0)

                    previous_perm = pred_indexes

                    # Finding the points in use by the local scope (what has been passed into the input of the merge)
                    active_indices = pred_indexes[pred_indexes != 0]-1
                    current_points = pts[active_indices]                  
                    
                    # Get the ground truth over these points only
                    local_gt = convexhull_example(current_points)[1]
                    local_exclusion += compute_exclusion_acc(
                        pred_indexes[None,:], 
                        local_gt[None,:],
                        lengths
                    )
                    
                    total_groups += 1

                

            
            if total_groups > 0:
                global_acc[i] = level_acc / total_groups
                global_acc_ref[i] = level_acc_ref / total_groups
                local_exclusion_scores[i] = local_exclusion / total_groups
                local_preservation[i] = pres / total_groups


    #global_acc, local_exclusion_scores = global_acc[::-1], local_exclusion_scores[::-1]

    print("Global Exclusion Error:", global_acc)
    print("Global Accuraccy:", global_acc_ref)
    print("Local Exclusion Scores:", local_exclusion_scores)
    print("Local Pres Scores:", local_preservation)

    ed = np.zeros_like(global_acc)
    ed[0] = local_exclusion_scores[0]

    for i in range(1,len(ed)):
        ed[i] = local_preservation[i] *  ed[i-1] + local_exclusion_scores[i]

    print("Predicted exclusion error", ed)

    print(" & ".join([f"{x*100:.2f}\\%" for x in global_acc]) + " \\\\")
    print(" & ".join([f"{x*100:.2f}\\%" for x in ed]) + " \\\\")


    return {
        'global_exclusion': global_acc,
        'global_miss_rate': global_acc_ref,
        'local_exclusion': local_exclusion_scores
    }


def draw_errs():
    # Base points
    points = np.array([
        [-0.8,  0.9],  # v1 (top left)
        [-.3, -0.3],
        [-1.4, 0.5],
        [-1.4, -0.9],  # v3 (bottom left)
        [ 0.7,  1.5],  # v2 (top right)
        [ 0.9, -1.1],  # v4 (bottom right)
        [ 0,  0]   # Center
    ])

    points = points * 0.7 # scale down

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs = axs.flatten()

    plt.subplots_adjust(
        wspace=0.1,
        hspace=0.2,
    )

    # Helper: draw graph
    def draw_graph(ax, title=None, included=[], points=points, color='k'):
        if title: ax.set_title(title,verticalalignment='bottom',y=0)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Plot all corners
        for i, (x, y) in enumerate(points):
            ax.plot(x, y, marker='o', color=color)

        # Draw edges to included points
        #for i1, i2 in pairwise(points[included,:]):
        #    ax.plot([i1[0], i2[0]], [i1[1], i2[1]], 'gray', linestyle='--')

        facecolor =  'lightgray'if color == 'k' else 'light'+color

        # Draw convex hull if at least 3 points
        if len(included) >= 3:
            hull_points = points[included]
            polygon = Polygon(hull_points, closed=True, edgecolor=color, facecolor=facecolor, alpha=0.3)
            ax.add_patch(polygon)


    """ # Plot 1: Correct
    draw_graph(axs[0], included=[0, 1, 2, 3], title="Correct convex hull")

    # Plot 2: Inclusion Error (missing top right = v2)
    draw_graph(axs[1], included=[0, 2, 3], title="Exclusion Error: Incorrectly excludes top right")
    # Plot 3: Exclusion Error (includes extra point)
    draw_graph(axs[2], included=[0, 1, 2, 3, 4], title="Inclusion Error: Incorrectly includes center")

    # Plot 4: Order Error (same points, wrong sequence)
    draw_graph(axs[3], included=[0, 2, 1, 3], title="Order Error: Sequence order creates invalid hull")"""

    draw_graph(axs[0], points=points[:4,:], color='blue')
    draw_graph(axs[0], points=points[4:,:], color='green')

    draw_graph(axs[1], points=points[:4,:], included=[0, 2, 3], color='blue')
    draw_graph(axs[1], points=points[4:,:], included=[0, 1, 2], color='green')

    draw_graph(axs[2], included=[0,2,3,5,4])

    plt.tight_layout()
    plt.show()




    

if __name__ == "__main__":
    #test_order_accuracy()
    #test_global_local_accumulation_f1()
    draw_errs()

    


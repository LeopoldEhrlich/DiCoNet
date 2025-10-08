"""Examines the split block for properties

Asks the questions:
    - Why does the performance of the model depend on splitting?
    - What are the factors that make a split good or bad?

Provides a graphical display of good and bad splits
"""

from main import test
from DCN import DivideAndConquerNetwork
from data_generator import Generator
from experiments.settings import DCN_Settings
from experiments.display_splits import SplitVisualizer
from experiments.experiment_utils import *

from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.stats import entropy, ttest_ind
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

def test_split_adversive():
    """Compares an adversively trained split module to a cooperatively trained model"""
    coop_settings    = DCN_Settings(num_examples_test=64*64,batch_size=64,
                                    load_split="./models/reg",load_merge="./models/random-split")
    adverse_settings = DCN_Settings(num_examples_test=coop_settings.num_examples_test,batch_size=coop_settings.batch_size,
                                    load_split="./models/evil-split", load_merge="./models/random-split")

    coop_DCN    = create_DCN(coop_settings)
    adverse_DCN = create_DCN(adverse_settings)

    gen = create_gen(range(1,6),settings=coop_settings)

    coop_accs = get_prediction_metrics(coop_DCN,gen)[0]
    comp_accs = get_prediction_metrics(adverse_DCN,gen)[0]

    print("Average cooperative accuracy:", np.mean())
    print("Average adversive accuracy:", np.mean())

    


def visualise_adversive_coop():
    """Visualises some split results from the cooperative and adversive splits"""
    coop_settings    = DCN_Settings(num_examples_test=2,batch_size=2,
                                    load_split="./models/reg",load_merge="./models/random-split")
    adverse_settings = DCN_Settings(num_examples_test=coop_settings.num_examples_test,batch_size=coop_settings.batch_size,
                                    load_split="./models/evil-split", load_merge="./models/random-split")

    coop_DCN    = create_DCN(coop_settings)
    adverse_DCN = create_DCN(adverse_settings)

    gen = create_gen([4],settings=coop_settings)

    out_arrs = zip(run_fwd_test(coop_DCN,gen),run_fwd_test(adverse_DCN,gen))

    for pair in out_arrs:
        visualizer = SplitVisualizer((pair[0][0],pair[1][0]))
        visualizer.show()


def analyze_attributes():
    X, y, feature_names, rand_scores = calculate_metrics()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)
    #print(importance_df.to_latex())


def feature_significance():
    X, y, feature_names, rand_scores = calculate_metrics()
    
    X = pd.DataFrame(X, columns=feature_names)
    
    # Group data
    group1 = X[y == 0]
    group2 = X[y == 1]

    results = []

    for feature in feature_names:
        # Perform t-test
        stat, pval = ttest_ind(group1[feature], group2[feature], equal_var=False, nan_policy='omit')
        significant = pval < 0.05
        results.append({
            'feature': feature,
            't_statistic': stat,
            'p_value': pval,
            'significant': significant
        })

    print(pd.DataFrame(results).sort_values('p_value'))


def show_average_metrics():
    X, y, feature_names, rand_scores = calculate_metrics()

    avg_rand = sum(rand_scores)/len(rand_scores)

    print("\nMetrics:")
    print("\tAverage Adjusted Rand Index between cooperative and adversive:", "{0:0.3f}".format(avg_rand))
    print()

    score_iter = zip(np.mean(X[y],axis=0), np.mean(X[~y],axis=0), feature_names)

    for score1, score2, name in score_iter:
        print(f"\tAverage {name} for cooperative:", "{0:0.3f}".format(score1))
        print(f"\tAverage {name} for adversive:", "{0:0.3f}".format(score2))
        print()


def calculate_metrics():
    coop_settings    = DCN_Settings(num_examples_test=64,batch_size=64,
                                    load_split="./models/reg",load_merge="./models/random-split")
    adverse_settings = DCN_Settings(num_examples_test=coop_settings.num_examples_test,batch_size=coop_settings.batch_size,
                                    load_split="./models/evil-split", load_merge="./models/random-split")

    coop_DCN    = create_DCN(coop_settings)
    adverse_DCN = create_DCN(adverse_settings)

    gen = create_gen([3],settings=coop_settings)
    out_tuples = zip(run_fwd_test(coop_DCN,gen),run_fwd_test(adverse_DCN,gen))

    feature_names = ["Silhouette Score", "Davies-Bouldin Score", "Entropy of Cluster Size Distribution", "Standard Deviation of Cluster Size","Average Intra-Cluster Pairwise Distance"]
    rand_scores = []

    X_list = []
    y_list = []

    for out_tuple in out_tuples:
        for batch_tuple in zip(*out_tuple):
            batch_e_tuple = (batch_tuple[0][0][4].data.cpu().numpy(), batch_tuple[1][0][4].data.cpu().numpy())
            batch_points_tuple = (batch_tuple[0][1], batch_tuple[1][1])

            for e_tuple, points_tuple in zip(zip(*batch_e_tuple), zip(*batch_points_tuple)):
                e_tuple = tuple(map(lambda x : x[x != 1.e+06],e_tuple))
                points_tuple = tuple(map(lambda x : x[x[:,0] != -1,:],points_tuple))

                rand_scores.append(adjusted_rand_score(*e_tuple))

                for label in (False,True):
                    features = np.zeros(len(feature_names))

                    features[0] = silhouette_score(points_tuple[label], e_tuple[label])
                    features[1] = davies_bouldin_score(points_tuple[label], e_tuple[label])
                    features[2] = compute_cluster_entropy(e_tuple[label])
                    features[3] = compute_cluster_std(e_tuple[label])
                    features[4] = cluster_compactness(points_tuple[label], e_tuple[label])


                    X_list.append(features)
                    y_list.append(label)

    X = np.stack(X_list,0)
    y = np.array(y_list)

    return X, y, feature_names, rand_scores             

def compute_cluster_std(labels):
    return np.std(np.bincount(labels.astype(np.int64))/len(labels))

def compute_cluster_entropy(labels):
    counts = np.bincount(labels.astype(np.int64))
    probs = counts / np.sum(counts)
    return entropy(probs)

def cluster_compactness(points,labels):
    """Average pairwise distance between points in the same cluster"""
    unique_labels = np.unique(labels)

    cluster_dists = []
    for u in unique_labels:
        cluster = points[labels==u,:]
        if len(cluster) < 2: continue
        cluster_dists.append(np.mean(pdist(cluster)))

    return np.mean(cluster_dists)
    

if __name__ == "__main__":
    #feature_significance()
    test_split_adversive()
    #visualise_adversive_coop()
    #show_average_metrics()
    #analyze_attributes()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clustering.kmesar import *
from clustering.utils import load_dataset_3_clusters_separate

if __name__ == '__main__':
    points = load_dataset_3_clusters_separate()
    df = pd.DataFrame(points, columns=['x', 'y'])

    print(df.head())

    init_centroids = np.array([
        [-0.1, 3], [0.4, 2], [-0.6, 0]
    ])

    plt.figure(1)
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=25, label='Points to be clustered')
    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], c='black', s=100, marker='x', label='Initial centroids')
    plt.legend(loc='upper right', prop={'size': 7})

    labels = assign_points_to_centroids(points, init_centroids)
    df['label_test'] = labels
    colors = ['red', 'green', 'blue']

    plt.figure(2)
    for cluster_label in range(3):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[0, 0], init_centroids[0, 1], c='black', s=100, marker='x', label='Centroid 0')
    plt.scatter(init_centroids[1, 0], init_centroids[1, 1], c='black', s=100, marker='x', label='Centroid 1')
    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='orange', s=300, marker='x',
                label='Centroid 2 (to be annealed)')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('State of K-Means before first iteration with given initial centroids')

    plt.show()

    centroid_to_be_annealed = init_centroids[2]
    label_j = 2
    k_clusters = 3

    # Random annealing
    annealing_vector, direction_point = calculate_annealing_vector(points,
                                                                   labels,
                                                                   centroid_to_be_annealed,
                                                                   label_j,
                                                                   1,
                                                                   annealing_method='random',
                                                                   annealing_weight_function='sigmoid',
                                                                   beta=6
                                                                   )
    annealed_centroid = centroid_to_be_annealed + annealing_vector

    print(f'{1 - (1 / (1 + np.exp(-1 / 6)))}')

    plt.figure(3)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Random point within dataset bounds')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Random\' method')

    plt.show()

    # Max annealing
    annealing_vector, direction_point  = calculate_annealing_vector(points,
                                                                    labels,
                                                                    centroid_to_be_annealed,
                                                                    label_j,
                                                                    1,
                                                                    annealing_method='max',
                                                                    annealing_weight_function='log',
                                                                    beta=1.2
                                                                    )
    annealed_centroid = centroid_to_be_annealed + annealing_vector

    print(f'ln2 / ln(1 + 1.2) = {annealing_weight(1, "log", 1.2)}')

    plt.figure(4)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Point with max distance from its centroid')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Max\' method')

    plt.show()

    # Min anealing
    annealing_vector, direction_point  = calculate_annealing_vector(points,
                                                  labels,
                                                  centroid_to_be_annealed,
                                                  label_j,
                                                  2,
                                                  annealing_method='min',
                                                  annealing_weight_function='sq',
                                                  beta=1)
    annealed_centroid = centroid_to_be_annealed + annealing_vector

    print(f'ln2 / ln(1 + 1.2) = {annealing_weight(2, "sq", 1)}')

    plt.figure(5)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1],
                c='darkorange', s=100, label='Point with min distance from its centroid')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Min\' method')

    plt.show()

    # Cluster-own annealing
    annealing_vector, direction_point  = calculate_annealing_vector(points,
                                                  labels,
                                                  centroid_to_be_annealed,
                                                  label_j,
                                                  2,
                                                  annealing_method='cluster_own',
                                                  annealing_weight_function='recip',
                                                  beta=1.6)
    annealed_centroid = centroid_to_be_annealed + annealing_vector


    print(f'(1 + 1.6) / (2 + 1.6) = {annealing_weight(2, "recip", 1.6)}')

    plt.figure(6)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Random point from centroid\'s own cluster')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Cluster-own\' method')

    plt.show()

    # Cluster-other annealing
    annealing_vector, direction_point  = calculate_annealing_vector(points,
                                                  labels,
                                                  centroid_to_be_annealed,
                                                  label_j,
                                                  2,
                                                  annealing_method='cluster_other',
                                                  annealing_weight_function='sigmoid',
                                                  beta=1)
    annealed_centroid = centroid_to_be_annealed + annealing_vector


    print(f'sigmoid-like(it=1, beta=1.2) = {annealing_weight(2, "sigmoid", 1)}')

    plt.figure(7)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[2, 0], init_centroids[2, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Random point from another cluster')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Cluster-other\' method')

    plt.show()

    # Cluster-mean
    annealing_vector, direction_point  = calculate_annealing_vector(points,
                                                  labels,
                                                  centroid_to_be_annealed,
                                                  label_j,
                                                  1,
                                                  annealing_method='cluster_mean',
                                                  annealing_weight_function='exp',
                                                  beta=1.1)
    annealed_centroid = centroid_to_be_annealed + annealing_vector


    print(f'exp(-1/1.1) = {annealing_weight(1, "exp", 1.1)}')

    plt.figure(8)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Mean of 3 random points from all clusters')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Cluster-mean\' method')

    plt.show()

    # Centroid-split
    annealing_vector, direction_point = calculate_annealing_vector(points,
                                                                   labels,
                                                                   init_centroids,
                                                                   label_j,
                                                                   2,
                                                                   annealing_method='centroid_split',
                                                                   annealing_weight_function='sigmoid',
                                                                   beta=1)
    annealed_centroid = centroid_to_be_annealed + annealing_vector

    print(f'sigmoid-like(2, 1) = {annealing_weight(2, "sigmoid", 1)}')

    plt.figure(9)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100,
                label='Point in the opposite direction from nearest centroid')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Centroid-split\' method')

    plt.show()

    # Centroid-gather
    annealing_vector, direction_point = calculate_annealing_vector(points,
                                                                   labels,
                                                                   init_centroids,
                                                                   label_j,
                                                                   2,
                                                                   annealing_method='centroid_gather',
                                                                   annealing_weight_function='log',
                                                                   beta=1.1)
    annealed_centroid = centroid_to_be_annealed + annealing_vector

    print(f'ln2 / ln3.1 = {annealing_weight(2, "sigmoid", 1.1)}')

    plt.figure(10)
    for cluster_label in range(k_clusters):
        cluster_subsample = df.loc[df['label_test'] == cluster_label]
        plt.scatter(cluster_subsample['x'], cluster_subsample['y'],
                    c=colors[cluster_label], s=30, label='_nolegend_')

    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], c='black', s=100, marker='x', label='Initial centroid')
    plt.scatter(direction_point[0], direction_point[1], c='darkorange', s=100, label='Mean of all centroids')
    plt.scatter(annealed_centroid[0], annealed_centroid[1], c='brown', s=300, marker='x', label='Annealed centroid')

    plt.plot([centroid_to_be_annealed[0], annealed_centroid[0]],
             [centroid_to_be_annealed[1], annealed_centroid[1]],
             c='black')

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title('Annealing centroid with \'Centroid-gather\' method')

    plt.show()


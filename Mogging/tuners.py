import grakel
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
from statistics import mean
from sklearn.cluster import SpectralClustering
from graph_kernel_hypo import compare_graphs_kernel
import matplotlib.pyplot as plt


class KNNGraphTuner:
    distance_matrices = None
    graph_kernel = None
    min_k = None
    max_k = None
    k_step = None

    def __init__(self, distance_matrices: list[np.ndarray], graph_kernel: grakel.kernels.Kernel,
                 min_k=1, max_k=10, k_step=1):
        self.distance_matrices = distance_matrices
        self.graph_kernel = graph_kernel
        self.min_k = min_k
        self.max_k = max_k
        self.k_step = k_step

    def _construct_graph(self, k: int, distance_matrix: np.ndarray):
        knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
        G = nx.from_scipy_sparse_array(knn_graph)

        G = self._ensure_connectivity(G, distance_matrix)

        return G

    def _ensure_connectivity(self, G: nx.Graph, distance_matrix: np.ndarray):
        if not nx.is_connected(G):
            print("The KNN graph is disjoint. Ensuring connectivity...")

            components = list(nx.connected_components(G))

            for i in range(len(components) - 1):
                min_dist = np.inf
                closest_pair = None
                for node1 in components[i]:
                    for node2 in components[i + 1]:
                        dist = distance_matrix[node1, node2]
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (node1, node2)

                G.add_edge(closest_pair[0], closest_pair[1])

        return G

    def _spectral_partition(self, graph: nx.Graph, distance_matrix: np.ndarray):
        adjacency_matrix = nx.to_numpy_array(graph)
        spectral_clustering = SpectralClustering(
            n_clusters=2,
            affinity='precomputed',
            random_state=42,
            n_init=100
            )
        labels = spectral_clustering.fit_predict(adjacency_matrix)

        group1 = [node for node, label in zip(graph.nodes(), labels) if label == 0]
        group2 = [node for node, label in zip(graph.nodes(), labels) if label == 1]

        subgraph1 = graph.subgraph(group1).copy()
        subgraph2 = graph.subgraph(group2).copy()

        group1_indices = np.array(group1)
        group2_indices = np.array(group2)

        subgraph1_distance_matrix = distance_matrix[np.ix_(group1_indices, group1_indices)]
        subgraph2_distance_matrix = distance_matrix[np.ix_(group2_indices, group2_indices)]

        subgraph1 = self._ensure_connectivity(subgraph1, subgraph1_distance_matrix)
        subgraph2 = self._ensure_connectivity(subgraph2, subgraph2_distance_matrix)

        return subgraph1, subgraph2

    def _kernel_based_similarity(self, k: int):
        within_graph_scores = []
        between_graph_scores = []

        graphs = []
        for distance_matrix in self.distance_matrices:
            graph = self._construct_graph(distance_matrix=distance_matrix, k=k)
            partition1, partition2 = self._spectral_partition(graph=graph, distance_matrix=distance_matrix)
            graphs.append(
                {
                    "split_graphs": (partition1, partition2),
                    "whole_graph": graph
                    }
                )
            similarity_within = compare_graphs_kernel([partition1, partition2], self.graph_kernel)[0, 1]
            within_graph_scores.append(similarity_within)

        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                whole_graph_1 = graphs[i]['whole_graph']
                whole_graph_2 = graphs[j]['whole_graph']

                similarity_between = compare_graphs_kernel([whole_graph_1, whole_graph_2], self.graph_kernel)[0, 1]
                between_graph_scores.append(similarity_between)

        return within_graph_scores, between_graph_scores

    def calculate_graph_statistics(self):
        graph_statistics = pd.DataFrame(columns=['k', 'normality_wtihin', 'normality_between', 'average_within',
                                                 'average_between', 'parametric_p_value', 'non_parametric_p_value'])
        for k in range(self.min_k, self.max_k, self.k_step):
            idx = len(graph_statistics)
            graph_statistics.loc[idx, 'k'] = k
            within_graph_scores, between_graph_scores = self._kernel_based_similarity(k)

            # Normality tests
            stat, p_value_within = shapiro(within_graph_scores)
            stat, p_value_between = shapiro(between_graph_scores)

            graph_statistics.loc[idx, 'normality_wtihin'] = p_value_within
            graph_statistics.loc[idx, 'normality_between'] = p_value_between

            # Average Scores
            graph_statistics.loc[idx, 'average_within'] = mean(within_graph_scores)
            graph_statistics.loc[idx, 'average_between'] = mean(between_graph_scores)

            # Parametric test
            t_stat, p_value = ttest_ind(within_graph_scores, between_graph_scores)
            graph_statistics.loc[idx,'parametric_p_value'] = p_value

            # Non-parametric test
            u_stat, p_value_non_parametric = mannwhitneyu(within_graph_scores, between_graph_scores)
            graph_statistics.loc[idx,'non_parametric_p_value'] = p_value_non_parametric

        return graph_statistics

    def calculate_and_graph(self):
        graph_statistics = self.calculate_graph_statistics()
        plt.figure(figsize=(10, 6))

        plt.plot(graph_statistics['k'], graph_statistics['parametric_p_value'], label='Parametric P-Value',
                 marker='o', linestyle='-', color='blue')
        plt.plot(graph_statistics['k'], graph_statistics['non_parametric_p_value'], label='Non-Parametric P-Value',
                 marker='s', linestyle='--', color='orange')

        plt.xlabel('k Values')
        plt.ylabel('P-Values')
        plt.title('P-Values vs. k')
        plt.axhline(y=0.05, color='red', linestyle=':', label='Significance Threshold (p=0.05)')
        plt.legend(loc='best')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        return graph_statistics

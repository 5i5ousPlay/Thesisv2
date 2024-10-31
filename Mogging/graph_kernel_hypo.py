import networkx as nx
from grakel import Graph
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
from statistics import mean
from grakel.kernels import LovaszTheta
from functions import segments_to_graph


def nx_to_grakel(G):
    edges = list(G.edges())

    # Create dummy node labels if none exist
    if not nx.get_node_attributes(G, 'label'):
        labels = {node: idx for idx, node in enumerate(G.nodes())}
    else:
        labels = nx.get_node_attributes(G, 'label')

    # Ensure the graph is formatted correctly for Grakel
    return Graph(edges, node_labels=labels)


def compare_graphs_kernel(graph_list: list, graph_kernel):
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]
    kernel = graph_kernel
    if isinstance(kernel, LovaszTheta):
        max_dim = max(len(g.nodes()) for g in graph_list)
        similarity_matrix = LovaszTheta(normalize=True, max_dim=max_dim).fit_transform(grakel_graphs)
        return similarity_matrix
    similarity_matrix = kernel.fit_transform(grakel_graphs)
    return similarity_matrix


def compare_within_and_between_artists(artists_segments, k, graph_kernel):
    within_artist_scores = []
    between_artist_scores = []

    # Convert each artist's segments into split and whole graphs
    artist_graphs = {}
    for artist, segments in artists_segments.items():
        # Split the segments for within-artist comparisons
        mid_point = len(segments) // 2
        segments_1 = segments[:mid_point]
        segments_2 = segments[mid_point:]

        # Generate k-NN graphs for each half (for within-artist comparison)
        graph_1, _ = segments_to_graph(k=k, segments=segments_1, labeled_segments=None)
        graph_2, _ = segments_to_graph(k=k, segments=segments_2, labeled_segments=None)

        # Generate a single k-NN graph for the whole artist segments (for between-artist comparison)
        whole_graph, _ = segments_to_graph(k=k, segments=segments, labeled_segments=None)

        # Store the graphs
        artist_graphs[artist] = {
            "split_graphs": (graph_1, graph_2),
            "whole_graph": whole_graph
        }

        # Compute WL kernel similarity for within-artist (same artist, split halves)
        similarity_within = compare_graphs_kernel([graph_1, graph_2], graph_kernel)[0, 1]
        within_artist_scores.append(similarity_within)

    # Compare between artists using whole graphs
    artist_names = list(artist_graphs.keys())
    for i in range(len(artist_names)):
        for j in range(i + 1, len(artist_names)):
            artist_1, artist_2 = artist_names[i], artist_names[j]
            whole_graph_1 = artist_graphs[artist_1]["whole_graph"]
            whole_graph_2 = artist_graphs[artist_2]["whole_graph"]

            # Compute WL kernel similarity between different artists
            similarity_between = compare_graphs_kernel([whole_graph_1, whole_graph_2], graph_kernel)[0, 1]
            between_artist_scores.append(similarity_between)

    return within_artist_scores, between_artist_scores


def graph_kernel_hypothesis_testing(artists_segments, k, graph_kernel):
    within_artist_scores, between_artist_scores = compare_within_and_between_artists(artists_segments, k, graph_kernel)

    print("===================================")
    print("RESULTS")

    stat, p_value_within = shapiro(within_artist_scores)
    stat, p_value_between = shapiro(between_artist_scores)

    print(f"Average Within-artist score: {mean(within_artist_scores)}")
    print(f"Average Between-artist score: {mean(between_artist_scores)}")

    # If p_value < 0.05, the data is not normally distributed
    print(f"Within-artist scores normality p-value: {p_value_within}")
    print(f"Between-artist scores normality p-value: {p_value_between}")

    # Parametric test (t-test)
    t_stat, p_value = ttest_ind(within_artist_scores, between_artist_scores)
    print(f"T-test p-value: {p_value}")

    # Non-parametric test (Mann-Whitney U test)
    u_stat, p_value_nonparametric = mannwhitneyu(within_artist_scores, between_artist_scores)
    print(f"Mann-Whitney U-test p-value: {p_value_nonparametric}")
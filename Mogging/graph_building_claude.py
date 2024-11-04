"""
Graph Building Module for Music Analysis

This module provides functions for building and analyzing graphs from musical segments.
It includes utilities for distance matrix calculation, graph construction, spectral analysis,
and comparison of musical styles between artists.
"""

import os

from networkx import Graph
from numpy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.manifold import MDS

from functions import *
from music21 import converter, environment
from grakel.kernels import WeisfeilerLehman
import music21
import pickle

# Configure MuseScore paths
env = environment.Environment()
env['musicxmlPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
env['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'

us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'


# ===============================
# File I/O Operations
# ===============================

def save_to_pickle(data, filename):
    """
    Saves a Python object to a pickle file.

    Args:
        data: Any Python object to save
        filename (str): Target filepath for the pickle file

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


def load_from_pickle(filename):
    """
    Loads a Python object from a pickle file.

    Args:
        filename (str): Source filepath of the pickle file

    Returns:
        The deserialized Python object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def get_directories_with_min_files(root_dir, min_file_count=5):
    """
    Finds directories containing at least the specified minimum number of files.

    Args:
        root_dir (str): Root directory to start search
        min_file_count (int): Minimum number of files required (default: 5)

    Returns:
        list: Directory names meeting the minimum file count criterion
    """
    qualifying_directories = []
    for dirpath, _, filenames in os.walk(root_dir):
        file_count = len([name for name in filenames if os.path.isfile(os.path.join(dirpath, name))])
        if file_count > min_file_count:
            qualifying_directories.append(os.path.basename(dirpath))
    return qualifying_directories


# ===============================
# Distance Matrix Operations
# ===============================

def segments_to_distance_matrix(segments: list[pd.DataFrame], cores=None):
    """
    Converts musical segments to a distance matrix using parallel processing.

    Args:
        segments (list[pd.DataFrame]): List of segment DataFrames
        cores (int, optional): Number of CPU cores to use for parallel processing

    Returns:
        np.array: Distance matrix comparing all segments
    """
    if __name__ == '__main__':
        if cores is not None and cores > cpu_count():
            raise ValueError(f"Insufficient cores. System has {cpu_count()} cores.")

        seg_np = [segment.to_numpy() for segment in segments]
        num_segments = len(seg_np)
        distance_matrix = np.zeros((num_segments, num_segments))

        args_list = [(i, j, segments[i], segments[j])
                     for i in range(num_segments)
                     for j in range(i + 1, num_segments)]

        with Manager() as manager:
            message_list = manager.list()

            def log_message(message):
                message_list.append(message)

            with Pool() as pool:
                results = pool.map(worker.calculate_distance, args_list)

            for i, j, distance, message in results:
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                log_message(message)

            for message in message_list:
                print(message)

        return distance_matrix


def segments_to_distance_matrices(segments: dict, pickle_dir=None, pickle_file=None):
    """
    Creates distance matrices for multiple composers' segments.

    Args:
        segments (dict): Dictionary mapping composers to their segments
        pickle_dir (str, optional): Directory to save pickle file
        pickle_file (str, optional): Custom filename for pickle file

    Returns:
        dict: Mapping of composers to their distance matrices
    """
    dist_mats = {}
    for composer, segments in segments.items():
        print(f'Composer: {composer} | Segments: {len(segments)}')
        dist_mats[composer] = segments_to_distance_matrix(segments)

    if pickle_dir:
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(dist_mats, os.path.join(pickle_dir, output_filename))

    return dist_mats


# ===============================
# Graph Construction & Visualization
# ===============================

def distance_matrix_to_knn_graph(k: int, distance_matrix: np.array, graph_title: str,
                                 seed: int, iterations: int, force_connect=False, show_labels=False):
    """
    Creates and visualizes a k-nearest neighbors graph from a distance matrix.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix
        graph_title (str): Title for the graph
        seed (int): Random seed for layout
        iterations (int): Number of layout iterations
        force_connect (bool): Whether to force graph connectivity
        show_labels (bool): Whether to show node labels

    Returns:
        None (displays plot)
    """
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    if not nx.is_connected(G) and force_connect:
        print("Connecting disjoint graph components...")
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

    pos = nx.spring_layout(G, seed=seed, iterations=iterations)
    nx.draw(G, node_size=50, pos=pos)

    if show_labels:
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(graph_title + f" (K={k})")
    plt.show()


def distance_matrix_to_knn_graph_scaled(k: int, distance_matrix: np.array, graph_title: str,
                                        seed: int):
    """
    Creates a KNN graph with node positions scaled according to the distance matrix.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix
        graph_title (str): Title for the graph
        seed (int): Random seed for reproducibility

    Returns:
        None (displays plot)
    """

    def adjust_overlapping_nodes(pos, threshold=0.01, adjustment=0.05):
        nodes = list(pos.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                xi, yi = pos[node_i]
                xj, yj = pos[node_j]
                distance = np.hypot(xi - xj, yi - yj)
                if distance < threshold:
                    pos[node_j] = (xj + adjustment, yj + adjustment)
        return pos

    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    positions = mds.fit_transform(distance_matrix)
    pos = {i: positions[i] for i in range(len(positions))}
    pos = adjust_overlapping_nodes(pos, threshold=1, adjustment=0.5)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4481FB")
    nx.draw_networkx_edges(G, pos)

    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white')

    plt.title(graph_title + f" (K={k})")
    plt.axis('equal')
    plt.axis('off')
    plt.show()


# ===============================
# Graph Analysis
# ===============================

def spectral_partition(distance_matrix):
    """
    Performs spectral partitioning on a distance matrix.

    Args:
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        tuple: Two arrays containing indices for the partitioned groups
    """
    sigma = np.mean(distance_matrix[np.nonzero(distance_matrix)])
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * sigma ** 2))
    np.fill_diagonal(similarity_matrix, 0)

    laplacian = csgraph.laplacian(similarity_matrix, normed=True)
    eigenvalues, eigenvectors = eigh(laplacian)
    fiedler_vector = eigenvectors[:, 1]

    partition = fiedler_vector > 0
    return np.where(partition)[0], np.where(~partition)[0]


def get_sub_distance_matrix(distance_matrix, group_indices):
    """
    Extracts a sub-matrix from a distance matrix based on group indices.

    Args:
        distance_matrix (np.array): Original distance matrix
        group_indices (np.array): Indices to extract

    Returns:
        np.array: Sub-matrix containing only the specified indices
    """
    return distance_matrix[np.ix_(group_indices, group_indices)]


def dist_mat_to_graph(k: int, distance_matrix):
    """
    Converts a distance matrix to a k-NN graph.

    Args:
        k (int): Number of nearest neighbors
        distance_matrix (np.array): Pairwise distance matrix

    Returns:
        networkx.Graph: K-nearest neighbors graph
    """
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    return nx.from_scipy_sparse_array(knn_graph)


def nx_to_grakel(G):
    """
    Converts a NetworkX graph to a GraKeL graph format.

    Args:
        G (networkx.Graph): Input NetworkX graph

    Returns:
        grakel.Graph: Converted graph in GraKeL format
    """
    edges = list(G.edges())
    labels = {node: idx for idx, node in enumerate(G.nodes())}
    return Graph(edges, node_labels=labels)


def compare_graphs(graph_list):
    """
    Compares multiple graphs using the Weisfeiler-Lehman kernel.

    Args:
        graph_list (list): List of NetworkX graphs to compare

    Returns:
        np.array: Similarity matrix between graphs
    """
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
    return wl_kernel.fit_transform(grakel_graphs)


def compare_within_and_between_artists(artists_dist_mat, k, min_segments=60):
    """
    Analyzes similarities within and between artists using graph-based comparison.

    Args:
        artists_dist_mat (dict): Dictionary mapping artists to their distance matrices
        k (int): Number of neighbors for k-NN graph construction
        min_segments (int): Minimum number of segments required for analysis

    Returns:
        tuple: (within_artist_df, between_artist_df, stats_dict)
            - within_artist_df: DataFrame of within-artist similarities
            - between_artist_df: DataFrame of between-artist similarities
            - stats_dict: Dictionary of analysis statistics

    Raises:
        ValueError: If input parameters are invalid or no artists can be processed
    """
    if not isinstance(artists_dist_mat, dict) or not artists_dist_mat:
        raise ValueError("artists_dist_mat must be a non-empty dictionary")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    # Initialize tracking variables
    processed_artists = {}
    stats_dict = {
        'total_artists': len(artists_dist_mat),
        'processed_artists': 0,
        'skipped_artists': 0,
        'average_segments': 0
    }

    # Process each artist
    for artist, dist_mat in artists_dist_mat.items():
        try:
            if len(dist_mat) < min_segments:
                stats_dict['skipped_artists'] += 1
                print(f"Skipping {artist}: insufficient segments ({len(dist_mat)})")
                continue

            group1, group2 = spectral_partition(dist_mat)
            D_group1 = get_sub_distance_matrix(dist_mat, group1)
            D_group2 = get_sub_distance_matrix(dist_mat, group2)

            processed_artists[artist] = {
                'partitions': (
                    dist_mat_to_graph(k, D_group1),
                    dist_mat_to_graph(k, D_group2)
                ),
                'full_graph': dist_mat_to_graph(k, dist_mat),
                'num_segments': len(dist_mat)
            }

            stats_dict['processed_artists'] += 1
            stats_dict['average_segments'] += len(dist_mat)

        except Exception as e:
            print(f"Error processing artist {artist}: {str(e)}")
            stats_dict['skipped_artists'] += 1
            continue

    if not processed_artists:
        raise ValueError("No artists could be processed with the given parameters")

    # Calculate average segments
    if stats_dict['processed_artists'] > 0:
        stats_dict['average_segments'] /= stats_dict['processed_artists']

    # Compute similarities
    within_artist_scores = []
    between_artist_scores = []

    # Within-artist comparisons
    for artist, data in processed_artists.items():
        try:
            graph1, graph2 = data['partitions']
            similarity_within = compare_graphs([graph1, graph2])[0, 1]
            within_artist_scores.append({
                'Artist': artist,
                'Within_Similarity': similarity_within,
                'Num_Segments': data['num_segments']
            })
        except Exception as e:
            print(f"Error computing within-artist similarity for {artist}: {str(e)}")

    # Between-artist comparisons
    artist_names = list(processed_artists.keys())
    for i, artist_1 in enumerate(artist_names):
        for artist_2 in artist_names[i:]:
            try:
                full_graph1 = processed_artists[artist_1]['full_graph']
                full_graph2 = processed_artists[artist_2]['full_graph']

                similarity_between = compare_graphs([full_graph1, full_graph2])[0, 1]
                between_artist_scores.append({
                    'Artist_1': artist_1,
                    'Artist_2': artist_2,
                    'Between_Similarity': similarity_between,
                    'Segments_1': processed_artists[artist_1]['num_segments'],
                    'Segments_2': processed_artists[artist_2]['num_segments']
                })
            except Exception as e:
                print(f"Error computing between-artist similarity for {artist_1} and {artist_2}: {str(e)}")

    # Create DataFrames and add metadata
    timestamp = pd.Timestamp.now()
    within_artist_df = pd.DataFrame(within_artist_scores)
    between_artist_df = pd.DataFrame(between_artist_scores)

    within_artist_df['Analysis_Date'] = timestamp
    between_artist_df['Analysis_Date'] = timestamp

    # Update statistics
    stats_dict.update({
        'within_artist_mean': within_artist_df['Within_Similarity'].mean(),
        'between_artist_mean': between_artist_df['Between_Similarity'].mean(),
        'analysis_timestamp': timestamp
    })

    return within_artist_df, between_artist_df, stats_dict


# ===============================
# Music Processing Functions
# ===============================

def mass_produce_segments(filepath, pickle_dir=None, pickle_file=None):
    """
    Processes multiple music files to generate segments for analysis.

    Args:
        filepath (str): Directory containing music files
        pickle_dir (str, optional): Directory to save pickle file
        pickle_file (str, optional): Custom filename for pickle file

    Returns:
        dict: Dictionary mapping composers to their segments

    Notes:
        This function assumes the existence of helper functions:
        - parse_score_elements
        - assign_ir_symbols
        - ir_symbols_to_matrix
        - assign_ir_pattern_indices
        - segmentgestalt
        - preprocess_segments
    """
    directories = os.listdir(filepath)
    composer_segments = dict.fromkeys(directories, None)
    piece_count = 0

    for piece in os.listdir(filepath):
        piece_path = os.path.join(filepath, piece)
        try:
            # Parse and process the music score
            parsed_score = converter.parse(piece_path)
            nmat, narr, sarr = parse_score_elements(parsed_score)
            ir_symbols = assign_ir_symbols(narr)
            ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
            ir_nmat = assign_ir_pattern_indices(ir_nmat)

            # Generate and preprocess segments
            segments = segmentgestalt(ir_nmat)
            prepped_segments = preprocess_segments(segments)

            piece_count += 1
            print(f'Composer: {piece} | Piece Count: {piece_count} \n Processed Segments: {len(prepped_segments)}')

            composer_segments[piece] = prepped_segments

        except Exception as e:
            print(f"Error processing piece {piece}: {str(e)}")
            continue

    # Save results if directory is specified
    if pickle_dir:
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(composer_segments, os.path.join(pickle_dir, output_filename))

    return composer_segments
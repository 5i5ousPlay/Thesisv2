import os

from networkx import Graph
from numpy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.manifold import MDS

from functions import *
from music21 import converter, environment
from grakel.kernels import WeisfeilerLehman

# Set the paths to MuseScore executable
env = environment.Environment()
env['musicxmlPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'  # Path to MuseScore executable
env['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'  # Path to MuseScore executable
import music21

us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
import pickle


def save_to_pickle(data, filename):
    """
    Saves a Python object to a pickle file.

    Parameters:
    - data: The Python object to save.
    - filename: The filename to save the pickle file to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


def load_from_pickle(filename):
    """
    Loads a Python object from a pickle file.

    Parameters:
    - filename: The filename of the pickle file to load.

    Returns:
    - The deserialized Python object.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def segments_to_distance_matrix(segments: list[pd.DataFrame], cores=None):
    if __name__ == '__main__':

        if cores is not None and cores > cpu_count():
            raise ValueError(
                f"You don't have enough cores! Please specify a value within your system's number of cores. \n Core "
                f"Count: {cpu_count()}")

        seg_np = [segment.to_numpy() for segment in segments]

        num_segments = len(seg_np)
        distance_matrix = np.zeros((num_segments, num_segments))

        # Create argument list for multiprocessing
        args_list = []
        for i in range(num_segments):
            for j in range(i + 1, num_segments):
                args_list.append((i, j, segments[i], segments[j]))

        with Manager() as manager:
            message_list = manager.list()

            def log_message(message):
                message_list.append(message)

            # Use multiprocessing Pool to parallelize the calculations
            with Pool() as pool:
                results = pool.map(worker.calculate_distance, args_list)

            # Update distance matrix with the results
            for i, j, distance, message in results:
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Reflect along the diagonal
                log_message(message)

            # Print messages from the shared list
            for message in message_list:
                print(message)

        return distance_matrix


def distance_matrix_to_knn_graph(k: int, distance_matrix: np.array, graph_title: str,
                                 seed: int, iterations: int, force_connect=False, show_labels=False):
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    # Detect if the graph is disjoint
    if not nx.is_connected(G) and force_connect:
        print("The KNN graph is disjoint. Ensuring connectivity...")

        # Calculate the connected components
        components = list(nx.connected_components(G))

        # Connect the components
        for i in range(len(components) - 1):
            min_dist = np.inf
            closest_pair = None
            for node1 in components[i]:
                for node2 in components[i + 1]:
                    dist = distance_matrix[node1, node2]
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)

            # Add an edge between the closest pair of nodes from different components
            G.add_edge(closest_pair[0], closest_pair[1])

    # Plot the final connected graph
    pos = nx.spring_layout(G, seed=seed, iterations=iterations)
    nx.draw(G, node_size=50, pos=pos)

    # Add labels if show_labels is True
    if show_labels:
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(graph_title + f" (K={k})")
    plt.show()
    plt.title(graph_title + f" (K={k})")
    plt.show()


def get_directories_with_min_files(root_dir, min_file_count=5):
    qualifying_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Count the number of files in the current directory
        file_count = len([name for name in filenames if os.path.isfile(os.path.join(dirpath, name))])

        # Check if the current directory has at least min_file_count files
        if file_count > min_file_count:
            qualifying_directories.append(os.path.basename(dirpath))

    return qualifying_directories


def mass_produce_segments(filepath, pickle_dir=None, pickle_file=None):
    directories = os.listdir(filepath)
    composer_segments = dict.fromkeys(directories, None)

    piece_count = 0
    for piece in os.listdir(filepath):
        piece_path = os.path.join(filepath, piece)
        parsed_score = converter.parse(piece_path)
        nmat, narr, sarr = parse_score_elements(parsed_score)
        ir_symbols = assign_ir_symbols(narr)
        ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
        ir_nmat = assign_ir_pattern_indices(ir_nmat)
        segments = segmentgestalt(ir_nmat)
        prepped_segments = preprocess_segments(segments)

        piece_count += 1

        print(f'Composer: {piece} | Piece Count: {piece_count} \n Processed Segments: {len(prepped_segments)}')

        composer_segments[piece] = prepped_segments

    if pickle_dir:
        # Use pickle_file name if provided, otherwise use default name
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(composer_segments, os.path.join(pickle_dir, output_filename))

    return composer_segments


def segments_to_distance_matrices(segments: dict, pickle_dir=None, pickle_file=None):
    dist_mats = {}

    for composer, segments in segments.items():
        print(f'composer: {composer} | Segments: {len(segments)}')
        dist_mats[composer] = segments_to_distance_matrix(segments)

    if pickle_dir:
        output_filename = pickle_file if pickle_file else 'composer_segments.pickle'
        save_to_pickle(dist_mats, os.path.join(pickle_dir, output_filename))


def distance_matrices_to_knn_graph_scaled(k: int, distance_matrix: np.array, graph_title: str,
                                          seed: int):
    """
    Creates and plots a KNN graph where node positions are scaled according to the distance matrix.
    Nodes that are very close (distance close to 0) are positioned beside each other to avoid overlap.

    Parameters:
    - k (int): Number of nearest neighbors.
    - distance_matrix (np.array): Pairwise distance matrix.
    - graph_title (str): Title of the graph.
    - seed (int): Random seed for reproducibility.
    """

    def adjust_overlapping_nodes(pos, threshold=0.01, adjustment=0.05):
        """
        Adjust positions of nodes to avoid overlap.

        Parameters:
        - pos (dict): Dictionary of positions {node: (x, y)}.
        - threshold (float): Minimum distance between nodes to consider them overlapping.
        - adjustment (float): Amount to adjust positions to separate overlapping nodes.

        Returns:
        - pos (dict): Adjusted positions.
        """
        nodes = list(pos.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                xi, yi = pos[node_i]
                xj, yj = pos[node_j]
                distance = np.hypot(xi - xj, yi - yj)
                if distance < threshold:
                    # Adjust positions slightly to avoid overlap
                    pos[node_j] = (xj + adjustment, yj + adjustment)
        return pos

    # Step 1: Create the KNN graph
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')

    # Convert to a NetworkX graph
    G = nx.from_scipy_sparse_array(knn_graph)

    # Step 2: Compute positions using MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    positions = mds.fit_transform(distance_matrix)

    # Convert positions to a dictionary for NetworkX
    pos = {i: positions[i] for i in range(len(positions))}

    # Step 3: Adjust positions to avoid overlap
    pos = adjust_overlapping_nodes(pos, threshold=1, adjustment=0.5)

    # Step 4: Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4481FB")  # , node_size=500
    nx.draw_networkx_edges(G, pos)

    # Add labels to the nodes
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white')

    # Adjust plot settings
    plt.title(graph_title + f" (K={k})")
    plt.axis('equal')  # Ensure equal scaling on both axes
    plt.axis('off')  # Hide the axes
    plt.show()


def spectral_partition(distance_matrix):
    # Convert distance to similarity
    sigma = np.mean(distance_matrix[np.nonzero(distance_matrix)])
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * sigma ** 2))
    np.fill_diagonal(similarity_matrix, 0)
    # Compute Laplacian
    laplacian = csgraph.laplacian(similarity_matrix, normed=True)
    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(laplacian)
    # Fiedler vector
    fiedler_vector = eigenvectors[:, 1]
    # Partition
    partition = fiedler_vector > 0
    group_1 = np.where(partition)[0]
    group_2 = np.where(~partition)[0]
    return group_1, group_2


def get_sub_distance_matrix(distance_matrix, group_indices):
    return distance_matrix[np.ix_(group_indices, group_indices)]


def dist_mat_to_graph(k: int, distance_matrix):
    # Compute the k-NN graph
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    # Convert the k-NN graph to a NetworkX graph
    G = nx.from_scipy_sparse_array(knn_graph)
    return G


def nx_to_grakel(G):
    edges = list(G.edges())
    labels = {node: idx for idx, node in enumerate(G.nodes())}
    return Graph(edges, node_labels=labels)


def compare_graphs(graph_list):
    # Convert all graphs to Grakel format
    grakel_graphs = [nx_to_grakel(g) for g in graph_list]

    # Initialize the Weisfeiler-Lehman kernel
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)

    # Compute the WL kernel similarity matrix
    similarity_matrix = wl_kernel.fit_transform(grakel_graphs)

    return similarity_matrix


def compare_within_and_between_artists(artists_dist_mat, k, min_segments=60):
    """
    Compares within-artist and between-artist similarity scores using graph-based analysis.

    Parameters:
    - artists_dist_mat (dict): Dictionary mapping artist names to their distance matrices
    - k (int): Number of neighbors for k-NN graph construction
    - min_segments (int): Minimum number of segments required for analysis (default: 60)

    Returns:
    - tuple: (within_artist_df, between_artist_df, stats_dict)
        - within_artist_df (pd.DataFrame): Within-artist similarity scores
        - between_artist_df (pd.DataFrame): Between-artist similarity scores
        - stats_dict (dict): Summary statistics of the analysis
    """
    if not isinstance(artists_dist_mat, dict) or not artists_dist_mat:
        raise ValueError("artists_dist_mat must be a non-empty dictionary")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    within_artist_scores = []
    between_artist_scores = []

    # Track statistics for analysis
    stats_dict = {
        'total_artists': len(artists_dist_mat),
        'processed_artists': 0,
        'skipped_artists': 0,
        'average_segments': 0
    }

    # Pre-process and store graphs for valid artists
    processed_artists = {}
    for artist, dist_mat in artists_dist_mat.items():
        try:
            if len(dist_mat) < min_segments:
                stats_dict['skipped_artists'] += 1
                print(f"Skipping {artist}: insufficient segments ({len(dist_mat)})")
                continue

            # Partition the distance matrix
            try:
                group1, group2 = spectral_partition(dist_mat)
            except Exception as e:
                print(f"Error in spectral partitioning for {artist}: {str(e)}")
                stats_dict['skipped_artists'] += 1
                continue

            # Get sub-matrices and create graphs
            D_group1 = get_sub_distance_matrix(dist_mat, group1)
            D_group2 = get_sub_distance_matrix(dist_mat, group2)

            # Store all graph information for the artist
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

    # Calculate average segments for processed artists
    if stats_dict['processed_artists'] > 0:
        stats_dict['average_segments'] /= stats_dict['processed_artists']

    # Compute within-artist similarities
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

    # Compute between-artist similarities more efficiently
    artist_names = list(processed_artists.keys())
    for i, artist_1 in enumerate(artist_names):
        # Only compare with artists we haven't compared yet
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

    # Convert to DataFrames with additional metadata
    within_artist_df = pd.DataFrame(within_artist_scores)
    between_artist_df = pd.DataFrame(between_artist_scores)

    # Add analysis timestamp
    timestamp = pd.Timestamp.now()
    within_artist_df['Analysis_Date'] = timestamp
    between_artist_df['Analysis_Date'] = timestamp

    # Add summary statistics
    stats_dict.update({
        'within_artist_mean': within_artist_df['Within_Similarity'].mean(),
        'between_artist_mean': between_artist_df['Between_Similarity'].mean(),
        'analysis_timestamp': timestamp
    })

    return within_artist_df, between_artist_df, stats_dict

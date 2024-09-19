from multiprocessing import cpu_count, Manager, Pool
from sklearn.neighbors import kneighbors_graph
import worker
import networkx as nx
import numpy as np
import pandas as pd
from music21 import chord, note, stream, meter
import matplotlib.pyplot as plt
from networkx.algorithms.cuts import conductance
from IPython.display import display, HTML
from pyvis.network import Network

# # Usage
# Current flow: \
# converter.parse() output -> parse_score_elements() ->
# assign_ir_symbols -> ir_symbols_to_matrix ->
# assign_ir_pattern_indices -> segmentgestalt

def extract_score_elements(score):
    """
    Extracts elements from a music21 score object and organizes them into a DataFrame.

    Parameters:
    score (music21.stream.Score): The music21 score object to extract elements from.

    Returns:
    pd.DataFrame: A DataFrame containing part index, offset, duration, type, and pitch information for each element.
    """
    elements = []

    for part_index, part in enumerate(score.parts):
        for element in part.flatten():
            element_info = {
                'part_index': part_index,
                'offset': element.offset,
                'duration': element.duration.quarterLength,
                'type': type(element).__name__
            }

            if isinstance(element, chord.Chord):
                element_info['pitches'] = [p.midi for p in element.pitches]
            elif isinstance(element, note.Rest):
                element_info['pitch'] = 0  # Representing rest with 0 pitch
            elif isinstance(element, note.Note):
                element_info['pitch'] = element.pitch.midi
            elif isinstance(element, meter.TimeSignature):
                element_info['numerator'] = element.numerator
                element_info['denominator'] = element.denominator
            else:
                continue  # Skip other element types for simplicity

            elements.append(element_info)

    elements_df = pd.DataFrame(elements)
    return elements_df


def recreate_score(elements_df):
    """
    Recreates a music21 score object from a DataFrame of score elements.

    Parameters:
    elements_df (pd.DataFrame): A DataFrame containing part index, offset, duration, type, and pitch information for each element.

    Returns:
    music21.stream.Score: The recreated music21 score object.
    """
    score = stream.Score()
    parts_dict = {}

    for _, row in elements_df.iterrows():
        part_index = row['part_index']
        if part_index not in parts_dict:
            parts_dict[part_index] = stream.Part()

        element_type = row['type']
        offset = row['offset']
        duration = row['duration']

        if element_type == 'Chord':
            pitches = [note.Note(p) for p in row['pitches']]
            element = chord.Chord(pitches, quarterLength=duration)
        elif element_type == 'Rest':
            element = note.Rest(quarterLength=duration)
        elif element_type == 'Note':
            element = note.Note(row['pitch'], quarterLength=duration)
        elif element_type == 'TimeSignature':
            element = meter.TimeSignature(f"{row['numerator']}/{row['denominator']}")
        else:
            continue

        element.offset = offset
        parts_dict[part_index].append(element)

    for part in parts_dict.values():
        score.append(part)

    return score


def parse_score_elements(score: stream.Score) -> tuple[pd.DataFrame, list, list]:
    """
    Parses a music21 score object into a DataFrame of note attributes and a list of note and chord elements.

    Parameters:
    score (music21.stream.Score): The music21 score object to parse.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: A DataFrame with onset, duration, and pitch for each note.
        - list: A list of note and chord elements.
    """
    trashed_elements = 0
    narr = []
    sarr = []
    nmat = pd.DataFrame(columns=['onset_beats', 'duration_beats', 'midi_pitch'])

    for part in score.parts:
        for element in part.flatten():
            sarr.append(element)
            row = [element.offset, element.duration.quarterLength]

            if isinstance(element, chord.Chord):
                row.append(element.root().midi)
                nmat.loc[len(nmat)] = row
                narr.append(element)
            elif isinstance(element, note.Rest):
                row.append(0)  # Representing rest with 0 pitch
                nmat.loc[len(nmat)] = row
                narr.append(element)
            elif isinstance(element, note.Note):
                row.append(element.pitch.midi)
                nmat.loc[len(nmat)] = row
                narr.append(element)
            else:
                trashed_elements += 1

    return nmat, narr, sarr


def calculate_ir_symbol(interval1, interval2, threshold=5):
    """
    Calculates the IR (Intervallic Relationship) symbol based on the intervals between notes.

    Parameters:
    interval1 (int): The interval between the first and second notes.
    interval2 (int): The interval between the second and third notes.
    threshold (int): The threshold value for determining the type of relationship.

    Returns:
    str: The IR symbol representing the relationship between the intervals.
    """
    direction = interval1 * interval2
    abs_difference = abs(interval2 - interval1)

    if direction > 0 and abs_difference < threshold:
        return 'P'  # Process
    elif interval1 == interval2 == 0:
        return 'D'  # Duplication
    elif (interval1 * interval2 < 0) and (-threshold <= abs(abs_difference) <= threshold) and (abs(interval2) != abs(interval1)):
        return 'IP'  # Intervallic Process
    elif (interval1 * interval2 < 0) and (abs(interval2) == abs(interval1)):
        return 'ID'  # Intervallic Duplication
    elif (interval1 * interval2 > 0) and (abs_difference >= threshold) and (abs(interval1) <= threshold):
        return 'VP'  # Vector Process
    elif (interval1 * interval2 < 0) and (abs(abs_difference) >= threshold) and (abs(interval1) >= threshold):
        return 'R'  # Reversal
    elif (interval1 * interval2 > 0) and (abs(abs_difference) >= threshold) and (abs(interval1) >= threshold):
        return 'IR'  # Intervallic Reversal
    elif (interval1 * interval2 < 0) and (abs_difference >= threshold) and (abs(interval1) <= threshold):
        return 'VR'  # Vector Reversal
    elif interval2 == 0 and not (interval1 < -5 or interval1 > 5):
        return 'IP'
    elif interval2 == 0 and (interval1 < -5 or interval1 > 5):
        return 'R'
    elif interval1 == 0 and not (interval2 < -5 or interval2 > 5):
        return 'P'
    elif interval1 == 0 and (interval2 < -5 or interval2 > 5):
        return 'VR'


def assign_ir_symbols(note_array):
    """
    Assigns IR symbols and colors to each element in the score array.

    Parameters:
    score_array (list): A list of music21 note and chord elements.

    Returns:
    list: A list of tuples containing each element, its IR symbol, and its color.
    """
    symbols = []
    current_group = []
    group_pitches = []

    color_map = {
        'P': 'blue',  # IR1: P (Process)
        'D': 'green',  # IR2: D (Duplication)
        'IP': 'red',  # IR3: IP (Intervallic Process)
        'ID': 'orange',  # IR4: ID (Intervallic Duplication)
        'VP': 'purple',  # IR5: VP (Vector Process)
        'R': 'cyan',  # IR6: R (Reversal)
        'IR': 'magenta',  # IR7: IR (Intervallic Reversal)
        'VR': 'yellow',  # IR8: VR (Vector Reversal)
        'M': 'pink',  # IR9: M (Monad)
        'd': 'lime',  # IR10 d (Dyad)
    }

    def evaluate_current_group():
        if len(current_group) == 3:
            interval1 = group_pitches[1] - group_pitches[0]
            interval2 = group_pitches[2] - group_pitches[1]
            symbol = calculate_ir_symbol(interval1, interval2)
            color = color_map.get(symbol, 'black')  # Default to black if symbol is not predefined
            symbols.extend([(note, symbol, color) for note in current_group])
        elif len(current_group) == 2:
            symbols.extend([(note, 'd', color_map['d']) for note in current_group])  # Dyad
        elif len(current_group) == 1:
            symbols.extend([(note, 'M', color_map['M']) for note in current_group])  # Monad
        current_group.clear()
        group_pitches.clear()

    for element in note_array:
        if isinstance(element, note.Note):
            current_group.append(element)
            group_pitches.append(element.pitch.ps)
            if len(current_group) == 3:
                evaluate_current_group()
        elif isinstance(element, chord.Chord):
            current_group.append(element)
            group_pitches.append(element.root().ps)
            if len(current_group) == 3:
                evaluate_current_group()
        elif isinstance(element, note.Rest):
            rest_tuple = (element, 'rest', 'black')
            evaluate_current_group()
            symbols.append(rest_tuple)
        else:
            if current_group:
                evaluate_current_group()

    # Handle any remaining notes
    if current_group:
        evaluate_current_group()

    return symbols


def visualize_notes_with_symbols(notes_with_symbols):
    """
    Visualizes notes with their assigned IR symbols and colors in a music21 score.

    Parameters:
    notes_with_symbols (list): A list of tuples containing each note, its IR symbol, and its color.

    Returns:
    None
    """
    s = stream.Score()
    part = stream.Part()
    for note, symbol, color in notes_with_symbols:
        note.style.color = color
        note.lyric = symbol
        part.append(note)
    s.append(part)
    s.show()


def ir_symbols_to_matrix(note_array, note_matrix):
    """
    Assigns IR symbols to the note matrix based on the note array.

    Parameters:
    note_array (list): A list of tuples containing note data, IR symbols, and colors.
    note_matrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    pd.DataFrame: The updated DataFrame with assigned IR symbols.
    """
    for pointer, (note_data, ir_symbol, color) in enumerate(note_array):
        note_matrix.at[pointer, 'ir_symbol'] = ir_symbol
    return note_matrix


def assign_ir_pattern_indices(notematrix):
    """
    Assigns pattern indices to the note matrix based on IR symbols.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes and IR symbols.

    Returns:
    pd.DataFrame: The updated DataFrame with assigned pattern indices.
    """
    pattern_index = 0
    indices = []
    i = 0
    while i < len(notematrix):
        ir_symbol = notematrix.iloc[i]['ir_symbol']
        if ir_symbol == 'd':
            indices.extend([pattern_index, pattern_index])
            i += 2
        elif ir_symbol == 'M' or ir_symbol == 'rest':
            indices.append(pattern_index)
            i += 1
        else:
            indices.extend([pattern_index, pattern_index, pattern_index])
            i += 3
        pattern_index += 1
    notematrix['pattern_index'] = indices
    return notematrix


def get_onset(notematrix: pd.DataFrame, timetype='beat'):
    """
    Retrieves the onset times from the note matrix.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    timetype (str): The type of time to retrieve (default is 'beat').

    Returns:
    pd.Series: A series containing the onset times.
    """
    if timetype == 'beat':
        return notematrix['onset_beats']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def get_duration(notematrix: pd.DataFrame, timetype='beat') -> pd.Series:
    """
    Retrieves the duration times from the note matrix.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    timetype (str): The type of time to retrieve (default is 'beat').

    Returns:
    pd.Series: A series containing the duration times.
    """
    if timetype == 'beat':
        return notematrix['duration_beats']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def calculate_clang_boundaries(notematrix: pd.DataFrame):
    """
    Calculates clang boundaries based on note matrix attributes.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    tuple: A tuple containing:
        - list: A list of indices representing clang boundaries.
        - pd.Series: A series indicating clang boundaries with boolean values.
    """
    cl = 2 * (get_onset(notematrix).diff().fillna(0) + get_duration(notematrix).shift(-1).fillna(0)) + abs(
        notematrix['midi_pitch'].diff().fillna(0))
    cl = cl.infer_objects()  # Ensure correct data types
    clb = (cl.shift(-1).fillna(0) > cl) & (cl.shift(1).fillna(0) > cl)
    clind = cl.index[clb].tolist()
    return clind, clb


def calculate_segment_boundaries(notematrix, clind):
    """
    Calculates segment boundaries based on clang boundaries and note attributes.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.
    clind (list): A list of clang boundary indices.

    Returns:
    pd.Series: A series indicating segment boundaries with boolean values.
    """
    first = [0] + clind
    last = [i - 1 for i in clind] + [len(notematrix) - 1]

    mean_pitch = []
    for i in range(len(first)):
        segment = notematrix.iloc[first[i]:last[i] + 1]
        weighted_pitch_sum = (segment['midi_pitch'] * segment['duration_beats']).sum()
        total_duration = segment['duration_beats'].sum()
        if total_duration > 0:
            mean_pitch.append(weighted_pitch_sum / total_duration)
        else:
            mean_pitch.append(0)  # Avoid division by zero by assigning 0 if total_duration is 0

    segdist = []
    for i in range(1, len(first)):
        distance = (abs(mean_pitch[i] - mean_pitch[i - 1]) +
                    notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats'] +
                    notematrix.iloc[first[i]]['duration_beats'] + notematrix.iloc[last[i - 1]]['duration_beats'] +
                    2 * (notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats']))
        segdist.append(distance)

    segb = [(segdist[i] > segdist[i - 1] and segdist[i] > segdist[i + 1]) for i in range(1, len(segdist) - 1)]
    segind = [clind[i] for i in range(1, len(segdist) - 1) if segb[i - 1]]

    s = pd.Series(0, index=range(len(notematrix)))
    s.iloc[segind] = 1

    return s


def adjust_segment_boundaries(notematrix, s):
    """
    Adjusts segment boundaries to ensure IR patterns are not split.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes and IR symbols.
    s (pd.Series): A series indicating initial segment boundaries with boolean values.

    Returns:
    pd.Series: The adjusted series indicating segment boundaries.
    """
    adjusted_s = s.copy()
    indices_with_ones = np.where(s == 1)[0].tolist()

    for i in indices_with_ones:
        current_pattern = notematrix.iloc[i]['pattern_index']
        ir_symbol = notematrix.iloc[i]['ir_symbol']

        if ir_symbol == 'M' or ir_symbol == 'rest':
            continue
        elif ir_symbol == 'd':
            if 0 < i < len(notematrix) - 1:
                prev_index = indices_with_ones[indices_with_ones.index(i) - 1] if indices_with_ones.index(i) > 0 else 0
                next_index = indices_with_ones[indices_with_ones.index(i) + 1] if indices_with_ones.index(i) < len(
                    indices_with_ones) - 1 else len(notematrix) - 1

                if (i - prev_index) > (next_index - i):
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i + 1] = 1
                else:
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i - 1] = 1
            continue

        if i > 1:
            previous_pattern1 = notematrix.iloc[i - 1]['pattern_index']
            previous_pattern2 = notematrix.iloc[i - 2]['pattern_index']

            if (current_pattern == previous_pattern1) and (current_pattern == previous_pattern2):
                continue
            elif current_pattern == previous_pattern1 and current_pattern != previous_pattern2:
                adjusted_s.iloc[i] = 0
                adjusted_s.iloc[i + 1] = 1
            elif current_pattern != previous_pattern1 and current_pattern != previous_pattern2:
                adjusted_s.iloc[i] = 0
                adjusted_s.iloc[i - 1] = 1

    return adjusted_s


def segmentgestalt(notematrix):
    """
    Segments the note matrix into meaningful groups based on IR patterns and boundaries.

    Parameters:
    notematrix (pd.DataFrame): A DataFrame containing note attributes.

    Returns:
    list: A list of segmented DataFrames.
    """
    if notematrix.empty:
        return None

    notematrix = assign_ir_pattern_indices(notematrix)
    clind, clb = calculate_clang_boundaries(notematrix)
    s = calculate_segment_boundaries(notematrix, clind)
    s = adjust_segment_boundaries(notematrix, s)

    c = pd.Series(0, index=range(len(notematrix)))
    c.iloc[clind] = 1

    segments = []
    start_idx = 0
    for end_idx in s[s == 1].index:
        segments.append(notematrix.iloc[start_idx:end_idx + 1])
        start_idx = end_idx + 1
    segments.append(notematrix.iloc[start_idx:])

    return segments


def preprocess_segments(segments: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Drops the pattern_index column and one-hot encodes the ir_symbol column for each DataFrame in the list of segments.

    Ensures that each DataFrame has columns for all specified states.

    Parameters:
    segments (list[pd.DataFrame]): List of DataFrames representing segments.

    Returns:
    list[pd.DataFrame]: List of preprocessed DataFrames.
    """
    # Define the possible states
    states = ['P', 'D', 'IP', 'ID', 'VP', 'R', 'IR', 'VR', 'M', 'd', 'rest']
    state_columns = [f'ir_symbol_{state}' for state in states]

    preprocessed_segments = []

    for segment in segments:
        # Drop the pattern_index column
        segment = segment.drop(columns=['pattern_index'])

        # One-hot encode the ir_symbol column
        segment = pd.get_dummies(segment, columns=['ir_symbol'])

        # Ensure all state columns are present
        for state_column in state_columns:
            if state_column not in segment.columns:
                segment[state_column] = 0
        segment[state_columns] = segment[state_columns].astype(int)

        # Reorder columns to ensure the state columns are in the correct order
        segment = segment[['onset_beats', 'duration_beats', 'midi_pitch'] + state_columns]

        preprocessed_segments.append(segment)

    return preprocessed_segments


def segments_to_distance_matrix(segments: list[pd.DataFrame], cores=None):
    """
    Converts segments to a distance matrix using multiprocessing.

    Parameters:
    segments (list[pd.DataFrame]): A list of segmented DataFrames.
    cores (int): The number of CPU cores to use for multiprocessing (default is None).

    Returns:
    np.ndarray: A distance matrix representing distances between segments.
    """
    if cores is not None and cores > cpu_count():
        raise ValueError(f"You don't have enough cores! Please specify a value within your system's number of "
                         f"cores. Core Count: {cpu_count()}")

    seg_np = [segment.to_numpy() for segment in segments]

    num_segments = len(seg_np)
    distance_matrix = np.zeros((num_segments, num_segments))

    args_list = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            args_list.append((i, j, segments[i], segments[j]))

    with Manager() as manager:
        message_list = manager.list()

        def log_message(message):
            message_list.append(message)

        with Pool(cores) as pool:
            results = pool.map(worker.calculate_distance, args_list)

        for i, j, distance, message in results:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Reflect along the diagonal
            log_message(message)

        for message in message_list:
            print(message)

    return distance_matrix


def segments_to_graph(k: int, segments: list[pd.DataFrame], labeled_segments, cores=None):
    """
    Converts segments to a k-NN graph and ensures connectivity.

    Parameters:
    k (int): The number of neighbors for k-NN graph.
    segments (list[pd.DataFrame]): A list of segmented DataFrames.
    labeled_segments (list): A list of labeled segments.
    cores (int): The number of CPU cores to use for multiprocessing (default is None).

    Returns:
    tuple: A tuple containing:
        - networkx.Graph: The resulting k-NN graph.
        - np.ndarray: The distance matrix used to create the graph.
    """
    distance_matrix = segments_to_distance_matrix(segments, cores=cores)
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')
    G = nx.from_scipy_sparse_array(knn_graph)

    for i in range(len(segments)):
        G.nodes[i]['segment'] = labeled_segments[i]

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

    return G, distance_matrix

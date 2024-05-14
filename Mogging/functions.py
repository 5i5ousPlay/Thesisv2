# midi to score array function (contains all data needed for score visualization)
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


def extract_score_elements(score):
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


def parse_score_elements(score):
    trashed_elements = 0
    narr = []
    sarr = []
    nmat = pd.DataFrame(columns=['onset_beats', 'duration_beats', 'midi_pitch'])

    for part in score.parts:
        for element in part.flatten():
            if isinstance(element, chord.Chord):
                row = [element.offset, element.duration.quarterLength, element.root().midi]
                nmat.loc[len(nmat)] = row
                narr.append(element)
            elif isinstance(element, note.Rest):
                row = [element.offset, element.duration.quarterLength, 0]
                nmat.loc[len(nmat)] = row
                narr.append(element)
            else:
                try:
                    row = [element.offset, element.duration.quarterLength, element.pitch.midi]
                    nmat.loc[len(nmat)] = row
                    narr.append(element)
                except:
                    trashed_elements += 1
                    # print(f"Trashed element #{trashed_elements}:\n{note}") # for debugging
    return nmat, narr, sarr


# IR symbol calculation function
def calculate_ir_symbol(interval1, interval2, threshold=5):
    direction = interval1 * interval2
    abs_difference = abs(interval2 - interval1)

    # Process
    if direction > 0 and (abs(interval2 - interval1)) < threshold:
        return 'P'
    # IR2: D (Duplication)
    elif interval1 == interval2 == 0:
        return 'D'
    # IR3: IP (Intervallic Process)
    elif ((interval1 * interval2) < 0) and (-threshold <= (abs(interval2) - abs(interval1)) <= threshold) and (
            abs(interval2) != abs(interval1)):
        return 'IP'
    # IR4: ID (Intervallic Duplication)
    elif ((interval1 * interval2) < 0) and (abs(interval2) == abs(interval1)):
        return 'ID'
    # IR5: VP (Vector Process)
    elif (interval1 * interval2 > 0) and (abs(interval2 - interval1) >= threshold) and (abs(interval1) <= threshold):
        return 'VP'
    # IR6: R (Reversal)
    elif (interval1 * interval2 < 0) and (abs(abs(interval2) - abs(interval1)) >= threshold) and (
            abs(interval1) >= threshold):
        return 'R'
    # IR7: IR (Intervallic Reversal)
    elif (interval1 * interval2 > 0) and (abs(abs(interval2) - abs(interval1)) >= threshold) and (
            abs(interval1) >= threshold):
        return 'IR'
    # IR8: VR (Vector Reversal)
    elif (interval1 * interval2 < 0) and (abs(interval2 - interval1) >= threshold) and (abs(interval1) <= threshold):
        return 'VR'
    elif interval2 == 0 and not (interval1 < -5 or interval1 > 5):
        return 'IP'
    elif interval2 == 0 and (interval1 < -5 or interval1 > 5):
        return 'R'
    elif interval1 == 0 and not (interval2 < -5 or interval2 > 5):
        return 'P'
    elif interval1 == 0 and (interval2 < -5 or interval2 > 5):
        return 'VR'


# assign IR symbol function (original; modified)
def assign_ir_symbols(score_array):
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
            symbols.extend((note, symbol, color) for note in current_group)
        elif len(current_group) == 2:
            symbols.extend((note, 'd', color_map['d']) for note in current_group)  # Dyad
        elif len(current_group) == 1:
            symbols.extend((note, 'M', color_map['M']) for note in current_group)  # Monad
        current_group.clear()
        group_pitches.clear()

    for element in score_array:
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


# Score visualization function
def visualize_notes_with_symbols(notes_with_symbols):
    s = stream.Score()
    part = stream.Part()
    for note, symbol, color in notes_with_symbols:
        print(note, symbol, color)
        note.style.color = color
        note.lyric = symbol
        part.append(note)
    s.append(part)
    s.show()


def ir_symbols_to_matrix(note_array, note_matrix):
    for pointer, (note_data, ir_symbol, color) in enumerate(note_array):
        note_matrix.at[pointer, 'ir_symbol'] = ir_symbol
    return note_matrix


def assign_ir_pattern_indices(notematrix):
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


# onset function
def get_onset(notematrix: pd.DataFrame, timetype='beat'):
    if timetype == 'beat':
        return notematrix['onset_beats']
    else:
        ValueError(f"Invalid timetype: {timetype}")


def get_duration(notematrix: pd.DataFrame, timetype='beat') -> pd.Series:
    if timetype == 'beat':
        return notematrix['duration_beats']
    else:
        ValueError(f"Invalid timetype: {timetype}")


# Calculate Clang Boundaries Function
def calculate_clang_boundaries(notematrix: pd.DataFrame):
    cl = 2 * (get_onset(notematrix).diff().fillna(0) + get_duration(notematrix).shift(-1).fillna(0)) + abs(
        notematrix['midi_pitch'].diff().fillna(0))
    cl = cl.infer_objects()  # Ensure correct data types
    clb = (cl.shift(-1).fillna(0) > cl) & (cl.shift(1).fillna(0) > cl)
    clind = cl.index[clb].tolist()
    return clind, clb


def calculate_segment_boundaries(notematrix, clind):
    # Initialize first and last indices for segments
    first = [0] + clind
    last = [i - 1 for i in clind] + [len(notematrix) - 1]

    # Calculate mean pitch for each segment weighted by duration
    mean_pitch = []
    for i in range(len(first)):
        segment = notematrix.iloc[first[i]:last[i] + 1]
        weighted_pitch_sum = (segment['midi_pitch'] * segment['duration_beats']).sum()
        total_duration = segment['duration_beats'].sum()
        if total_duration > 0:
            mean_pitch.append(weighted_pitch_sum / total_duration)
        else:
            mean_pitch.append(0)  # Avoid division by zero by assigning 0 if total_duration is 0

    # Calculate segment distances
    segdist = []
    for i in range(1, len(first)):
        distance = (abs(mean_pitch[i] - mean_pitch[i - 1]) +
                    notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats'] +
                    notematrix.iloc[first[i]]['duration_beats'] + notematrix.iloc[last[i - 1]]['duration_beats'] +
                    2 * (notematrix.iloc[first[i]]['onset_beats'] - notematrix.iloc[last[i - 1]]['onset_beats']))
        segdist.append(distance)

    # Identify local maxima in segment distances and check pattern_index consistency
    segb = [(segdist[i] > segdist[i - 1] and segdist[i] > segdist[i + 1]) for i in range(1, len(segdist) - 1)]
    segind = [clind[i] for i in range(1, len(segdist) - 1) if segb[i - 1]]

    # Create binary vector for segment boundaries
    s = pd.Series(0, index=range(len(notematrix)))
    s.iloc[segind] = 1

    return s


def adjust_segment_boundaries(notematrix, s):
    adjusted_s = s.copy()
    indices_with_ones = np.where(s == 1)[0].tolist()
    i = 0

    while i < len(notematrix):
        if adjusted_s.iloc[i] == 1:
            current_pattern = notematrix.iloc[i]['pattern_index']
            ir_symbol = notematrix.iloc[i]['ir_symbol']

            if ir_symbol == 'M' or ir_symbol == 'rest':
                # Skip monads and rests
                i += 1
                continue

            elif ir_symbol == 'd':
                if 0 < i < len(notematrix) - 1:
                    prev_index = indices_with_ones[indices_with_ones.index(i) - 1] if indices_with_ones.index(
                        i) > 0 else 0
                    next_index = indices_with_ones[indices_with_ones.index(i) + 1] if indices_with_ones.index(i) < len(
                        indices_with_ones) - 1 else len(notematrix) - 1

                    # Check the distances to previous and next indices with ones
                    if (i - prev_index) > (next_index - i):
                        adjusted_s.iloc[i] = 0
                        adjusted_s.iloc[i + 1] = 1
                    else:
                        adjusted_s.iloc[i] = 0
                        adjusted_s.iloc[i - 1] = 1
                i += 1
                continue

            # Handle cases for triads and other patterns
            if i > 1:  # Ensure there are at least two previous elements to check
                previous_pattern1 = notematrix.iloc[i - 1]['pattern_index']
                previous_pattern2 = notematrix.iloc[i - 2]['pattern_index']

                if (current_pattern == previous_pattern1) and (current_pattern == previous_pattern2):
                    i += 1
                    continue
                elif current_pattern == previous_pattern1 and current_pattern != previous_pattern2:
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i + 1] = 1
                elif current_pattern != previous_pattern1 and current_pattern != previous_pattern2:
                    adjusted_s.iloc[i] = 0
                    adjusted_s.iloc[i - 1] = 1
            i += 1
        else:
            i += 1

    return adjusted_s


def segmentgestalt(notematrix):
    if notematrix.empty:
        return None

    # Assign IR pattern indices
    notematrix = assign_ir_pattern_indices(notematrix)

    # Calculate clang boundaries
    clind, clb = calculate_clang_boundaries(notematrix)

    # Calculate segment boundaries
    s = calculate_segment_boundaries(notematrix, clind)

    # Adjust segment boundaries to ensure IR patterns are not split
    s = adjust_segment_boundaries(notematrix, s)

    # Create binary vector for clang boundaries
    c = pd.Series(0, index=range(len(notematrix)))
    c.iloc[clind] = 1

    # Create segments based on adjusted segment boundaries
    segments = []
    start_idx = 0
    for end_idx in s[s == 1].index:
        segments.append(notematrix.iloc[start_idx:end_idx + 1])
        start_idx = end_idx + 1
    segments.append(notematrix.iloc[start_idx:])

    # return c, s, segments # return c, s indices for debugging
    return segments

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
            with Pool(cores) as pool:
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


# segments to graph function
def segments_to_graph(k: int, segments: list[pd.DataFrame], labeled_segments, cores=None):
    # Convert segments to a distance matrix
    distance_matrix = segments_to_distance_matrix(segments, cores=cores)

    # Compute the k-NN graph
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=k, mode='connectivity')

    # Convert the k-NN graph to a NetworkX graph
    G = nx.from_scipy_sparse_array(knn_graph)

    # Add segment data as attributes to each node
    for i in range(len(segments)):
        G.nodes[i]['segment'] = labeled_segments[i]  # print shit

    # Detect if the graph is disjoint
    if not nx.is_connected(G):
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

    return G, distance_matrix

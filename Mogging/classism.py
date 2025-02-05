# music_analysis.py

# Import all required dependencies at the top
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from music21 import converter
import pickle
import numpy as np
from graph_building_claude import *
from visualization import *

__all__ = ['MusicFileManager', 'MusicSegmentAnalyzer', 'MusicVisualizer']


# [Previous class definitions go here...]
class MusicFileManager:
    """Handles music file selection and management through a UI interface"""

    def __init__(self, base_path="../Music Database"):
        self.base_path = Path(base_path)
        self.files = self._initialize_file_mapping()
        self.filepath_dropdown = None
        self.path_display = None
        self._setup_widgets()

    def _initialize_file_mapping(self):
        """Initialize the mapping of display names to file paths"""
        # return {
        #     "Liszt - Ungarische Rhapsodie": self.base_path / "GTTM Database/Franz Liszt/Ungarische Rhapsodie S.244 Nr.2 cis moll.xml",
        #     "Liszt - Liebestraume": self.base_path / "GTTM Database/Franz Liszt/Liebestraume 3 Notturnos S.541 R.211 As dur.xml",
        #     "Tchaikovsky - Swan Lake Finale": self.base_path / "GTTM Database/Pyotr Il'yich Tchaikovsky/Swan Lake Op.20 No.9 Finale.xml",
        #     # Add other files...
        # }
        return {
            "Liszt - Ungarische Rhapsodie": r"..\Music Database\GTTM Database\Franz Liszt\Ungarische Rhapsodie S.244 Nr.2 cis moll.xml",
            "Liszt - Liebestraume": r"..\Music Database\GTTM Database\Franz Liszt\Liebestraume 3 Notturnos S.541 R.211 As dur.xml",
            "Tchaikovsky - Swan Lake Finale": r"..\Music Database\GTTM Database\Pyotr Il’yich Tchaikovsky\Swan Lake Op.20 No.9 Finale.xml",
            "Segment 1": r"..\Music Database\fabricated\segment1.mid",
            "Segment 2": r"..\Music Database\fabricated\segment2.mid",
            "Segment 3": r"..\Music Database\fabricated\segment3.mid",
            "Segment 4": r"..\Music Database\fabricated\segment4.mid",
            "Segment 5": r"..\Music Database\fabricated\segment5.mid",
            "Fabricated": r"..\Music Database\fabricated\fabricated.mxl",
            "Fabricated2": r"..\Music Database\fabricated\fabricated2.mxl",
            "Mountain King": r"..\\In_the_Hall_of_the_Mountain_King_Easy_variation2.mxl",
            "Bach - Minuet in G Major": r"..\Music Database\Good_maybe\Bach_-_Minuet_in_G_Major_Bach.mxl"
        }

    def _setup_widgets(self):
        """Setup the UI widgets for file selection"""
        self.filepath_dropdown = widgets.Dropdown(
            options=[(name, str(path)) for name, path in self.files.items()],
            description='Select piece:',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        )

        self.path_display = widgets.Text(
            description='Full path:',
            disabled=True,
            layout={'width': '800px'},
            style={'description_width': 'initial'}
        )

        self.filepath_dropdown.observe(self._on_selection_change, names='value')
        self.path_display.value = self.filepath_dropdown.value

    def _on_selection_change(self, change):
        """Handle selection changes in the dropdown"""
        self.path_display.value = change['new']

    def display_selector(self):
        """Display the file selection widgets"""
        display(widgets.VBox([self.filepath_dropdown, self.path_display]))

    @property
    def selected_file(self):
        """Get the currently selected file path"""
        return self.filepath_dropdown.value


class MusicSegmentAnalyzer:
    """Handles the analysis of musical segments"""

    def __init__(self, score_path=None):
        self.score_path = score_path
        self.parsed_score = None
        self.segments = None
        self.prepped_segments = None
        self.distance_matrix = None

    def load_score(self, score_path=None):
        """Load and parse a music score"""
        if score_path:
            self.score_path = score_path
        if not self.score_path:
            raise ValueError("No score path provided")

        self.parsed_score = converter.parse(self.score_path)
        return self

    def analyze_segments(self):
        """Perform segment analysis on the loaded score"""
        if not self.parsed_score:
            raise ValueError("No score loaded. Call load_score first.")

        nmat, narr, sarr = parse_score_elements(self.parsed_score)
        ir_symbols = assign_ir_symbols(narr)
        ir_nmat = ir_symbols_to_matrix(ir_symbols, nmat)
        ir_nmat = assign_ir_pattern_indices(ir_nmat)
        self.segments = segmentgestalt(ir_nmat)
        return self

    def preprocess_segments(self):
        """Preprocess the analyzed segments"""
        if self.segments is None:
            raise ValueError("No segments analyzed. Call analyze_segments first.")

        self.prepped_segments = preprocess_segments(self.segments)
        return self

    def calculate_distance_matrix(self):
        """Calculate distance matrix for preprocessed segments"""
        if self.prepped_segments is None:
            raise ValueError("No preprocessed segments. Call preprocess_segments first.")

        self.distance_matrix = segments_to_distance_matrix(self.prepped_segments)
        return self

    def save_segments(self, filepath):
        """Save segments to a pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.segments, f)

    def load_segments(self, filepath):
        """Load segments from a pickle file"""
        with open(filepath, 'rb') as f:
            self.segments = pickle.load(f)
        return self


class MusicVisualizer:
    """Handles visualization of musical segments and graphs"""

    def __init__(self, analyzer=None):
        self.analyzer = analyzer

    def visualize_colored_segments(self):
        """Create and display a score with colored segments"""
        if not self.analyzer or not self.analyzer.parsed_score:
            raise ValueError("No analyzed score available")

        colored_score = visualize_score_with_colored_segments(
            self.analyzer.parsed_score,
            self.analyzer.segments
        )
        colored_score.show()

    def visualize_multiple_segments(self, num_segments=5):
        """Display multiple segments using the MultiSegmentVisualizer"""
        if not self.analyzer or not self.analyzer.segments:
            raise ValueError("No segments available")

        MultiSegmentVisualizer(
            self.analyzer.segments,
            self.analyzer.parsed_score,
            num_segments
        )

    def visualize_knn_graph(self, k=3, seed=69, title=None):
        """Create and display a KNN graph of the segments"""
        # if not self.analyzer or not self.analyzer.distance_matrix is None:
        #     raise ValueError("No distance matrix available")

        title = title or "Segment Analysis"
        distance_matrix_to_knn_graph_scaled(
            k,
            self.analyzer.distance_matrix,
            f"{title}\n",
            seed
        )

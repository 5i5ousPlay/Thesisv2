from functions import *

def get_pitch(notematrix: pd.DataFrame, timetype='beat'):
    if timetype == 'beat':
        return notematrix['midi_pitch']
    else:
        raise ValueError(f"Invalid timetype: {timetype}")


def boundary(notematrix: pd.DataFrame) -> pd.Series:
    if len(notematrix) < 1:
        return pd.Series([], dtype=float)
    if len(notematrix) < 2:
        # With only one note, just return b = [1]
        return pd.Series([1], index=notematrix.index)

    pitch = notematrix['midi_pitch'].values
    on = notematrix['onset_beats'].values
    du = notematrix['duration_beats'].values
    off = on + du

    # Profiles
    pp = np.abs(np.diff(pitch))  # pitch profile
    po = np.diff(on)  # IOI profile
    pr = np.maximum(0, on[1:] - off[:-1])  # rest profile

    def degree_of_change(x):
        if len(x) < 2:
            return np.zeros_like(x)
        dx = np.abs(x[1:] - x[:-1]) / (1e-6 + x[1:] + x[:-1])
        dx = np.append(dx, 0)
        return dx

    rp = degree_of_change(pp)
    ro = degree_of_change(po)
    rr = degree_of_change(pr)

    def strength_profile(p_profile, r_profile):
        if len(p_profile) < 2:
            # If there's less than 2 intervals, no internal structure
            return np.zeros_like(p_profile)
        multiplier = np.zeros_like(p_profile)
        for i in range(1, len(p_profile)):
            multiplier[i] = r_profile[i - 1] + r_profile[i]
        sp = p_profile * multiplier
        mx = sp.max() if len(sp) > 0 else 0
        if mx > 0.1:
            sp = sp / mx
        return sp

    sp = strength_profile(pp, rp)
    so = strength_profile(po, ro)
    sr = strength_profile(pr, rr)

    N = len(notematrix)
    b = np.zeros(N)
    b[0] = 1
    if N > 1:
        b[1:] = 0.25 * sp + 0.5 * so + 0.25 * sr

    return pd.Series(b, index=notematrix.index)


def segment_from_boundary(notematrix: pd.DataFrame) -> list:
    """
    Similar to segmentgestalt, but uses the Cambouropoulos boundary model.
    Identifies segments based on local maxima in the boundary strength profile.
    """
    if notematrix.empty:
        return []

    b = boundary(notematrix)
    # Identify local maxima in b to define segment boundaries.
    # Consider a local maximum if b[i] > b[i-1] and b[i] > b[i+1].
    # For edge cases, the first note (i=0) is always a segment start because b[0] = 1 by definition.
    # We'll look for additional boundaries in the interior notes.

    s = pd.Series(0, index=notematrix.index)
    if len(notematrix) > 2:
        # Check local maxima (excluding the first note since it's always a start)
        for i in range(1, len(b) - 1):
            if b[i] > b[i - 1] and b[i] > b[i + 1]:
                s.iloc[i] = 1

    # The first note is the start of the first segment
    # If we want to match segmentgestalt behavior closely, the first note is inherently a boundary.
    # If we always consider the first note as segment start (b[0]=1), we do not necessarily set s[0]=1,
    # because s marks segment boundaries starting from after the first note. The initial segment starts at 0 by definition.
    # s[i]=1 marks the boundary BEFORE note i.

    # Create `c` analogous to segmentgestalt. Since we have no "clang" logic here, set it to zeros.
    c = pd.Series(0, index=notematrix.index)

    # Create segments
    segments = []
    start_idx = 0
    segment_boundaries = s[s == 1].index.tolist()
    segment_boundaries = adjust_segment_boundaries(notematrix, segment_boundaries)
    for end_idx in segment_boundaries:
        segments.append(notematrix.iloc[start_idx:end_idx])
        start_idx = end_idx
    if start_idx < len(notematrix):
        segments.append(notematrix.iloc[start_idx:])

    return segments, c, s


def segment_cambouropoulos(notematrix: pd.DataFrame):
    """
    Produce segments using Cambouropoulos's boundary detection logic,
    structured similarly to segmentgestalt.

    Returns:
    - segments: list of segmented DataFrames
    - c: series indicating "clang boundaries" (all zeros here)
    - s: series indicating segment boundaries (1 where there's a segment boundary)
    """
    if notematrix.empty:
        return None, None, None

    segments, c, s = segment_from_boundary(notematrix)
    return segments, c, s

"""Dynamic level classification — map dBFS values to musical dynamics.

Maps RMS loudness levels to standard musical dynamic markings:
  pp (pianissimo), p (piano), mp (mezzopiano), mf (mezzoforte),
  f (forte), ff (fortissimo)

The thresholds are approximate and should be calibrated to your
recording setup. VocalSet annotations can be used to refine them.
"""

import numpy as np

DYNAMIC_VOCAB = ["pp", "p", "mp", "mf", "f", "ff"]

# dBFS thresholds (lower bound of each dynamic level).
# Anything below PP_THRESHOLD is considered silence.
# These are intentionally conservative — real recordings vary widely.
DEFAULT_THRESHOLDS_DBFS = {
    "pp": -65.0,
    "p":  -55.0,
    "mp": -48.0,
    "mf": -40.0,
    "f":  -32.0,
    "ff": -24.0,
}

# Lowered to accommodate browser mics without auto-gain-control (AGC).
# Without AGC a typical laptop mic sits 10–15 dBFS lower than a
# pre-amplified recording interface.
SILENCE_THRESHOLD_DBFS = -72.0


def classify_dynamic(
    dbfs: float,
    thresholds: dict[str, float] | None = None,
) -> str | None:
    """Classify a single dBFS value into a dynamic level.

    Args:
        dbfs:       loudness in dBFS
        thresholds: optional custom threshold dict (default: DEFAULT_THRESHOLDS_DBFS)

    Returns:
        dynamic label ('pp', 'p', ..., 'ff'), or None for silence
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_DBFS

    if dbfs < SILENCE_THRESHOLD_DBFS:
        return None

    label = "pp"
    for name in DYNAMIC_VOCAB:
        if dbfs >= thresholds[name]:
            label = name
    return label


def classify_dynamic_sequence(
    dbfs_sequence: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> list[str | None]:
    """Classify a sequence of dBFS values.

    Args:
        dbfs_sequence: (T,) array of dBFS values
        thresholds:    optional custom thresholds

    Returns:
        list of dynamic labels, one per frame (None = silence)
    """
    return [classify_dynamic(float(d), thresholds) for d in dbfs_sequence]


def dynamic_to_int(label: str | None) -> int:
    """Map a dynamic label to an integer index (None → -1)."""
    if label is None:
        return -1
    return DYNAMIC_VOCAB.index(label)


def int_to_dynamic(idx: int) -> str | None:
    """Map an integer index back to a dynamic label (-1 → None)."""
    if idx < 0:
        return None
    return DYNAMIC_VOCAB[idx]

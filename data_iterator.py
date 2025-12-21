"""
data_iterator.py

Load precomputed image features and labels, convert labels to token ids,
and pack examples into batches subject to constraints (batch size and
total image area per batch). Designed for CPU-only environments.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pickle as pkl
import numpy as np

def dataIterator(
    feature_file: str,
    label_file: str,
    dictionary: Dict[str, int],
    batch_size: int,
    batch_Imagesize: int,
    maxlen: int,
    maxImagesize: int
) -> Tuple[List[List[np.ndarray]], List[List[List[int]]]]:
    """
    Create batches from pre-extracted features and label text.

    Args:
        feature_file: path to .pkl file containing a dict {uid: feature_array}
        label_file: path to .txt file where each line is: <uid> <token1> <token2> ...
        dictionary: mapping token string -> integer id
        batch_size: maximum number of samples per batch
        batch_Imagesize: maximum allowed largest_image_area * batch_count per batch
        maxlen: maximum allowed token length per sample
        maxImagesize: maximum allowed image area for a single sample

    Returns:
        (feature_batches, label_batches)
        - feature_batches: list of batches; each batch is list of numpy arrays
        - label_batches: list of batches; each batch is list of lists of ints
    """
    feature_path = Path(feature_file)
    label_path = Path(label_file)

    # --- Load features (.pkl should contain a dict: uid -> feature ndarray) ---
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    with feature_path.open("rb") as fp:
        try:
            features = pkl.load(fp)
        except Exception as e:
            raise RuntimeError(f"Failed to load feature pickle '{feature_file}': {e}")

    # --- Load labels ---
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    with label_path.open("r", encoding="utf-8") as fp2:
        labels = [line.strip() for line in fp2 if line.strip()]

    # --- Convert labels to integer sequences using dictionary ---
    targets = {}
    for line in labels:
        parts = line.split()
        uid = parts[0]
        tokens = parts[1:]
        token_ids: List[int] = []
        for t in tokens:
            if t in dictionary:
                token_ids.append(dictionary[t])
            else:
                # Raise an informative error rather than exiting the process
                raise KeyError(f"Token '{t}' (uid={uid}) not found in dictionary")
        targets[uid] = token_ids

    # --- Compute per-image sizes and filter available uids ---
    image_sizes = {}
    image_height = {}
    image_width = {}
    for uid, feat in features.items():
        # feat expected shape: (channels, H, W) or similar; compute H*W robustly
        if hasattr(feat, "shape") and len(feat.shape) >= 2:
            h = int(feat.shape[-2])
            w = int(feat.shape[-1])
            size = h * w
        else:
            raise ValueError(f"Feature for uid {uid} has unexpected shape: {getattr(feat,'shape', None)}")
        image_sizes[uid] = size
        image_height[uid] = h
        image_width[uid] = w

    # Sort uids by image size descending (largest first)
    sorted_uids = sorted(image_sizes.items(), key=lambda kv: kv[1], reverse=True)

    # --- Build batches ---
    feature_batches: List[List[np.ndarray]] = []
    label_batches: List[List[List[int]]] = []

    cur_feat_batch: List[np.ndarray] = []
    cur_label_batch: List[List[int]] = []
    cur_biggest = 0
    cur_count = 0

    total_skipped_too_long = 0
    total_skipped_too_large = 0

    for uid, size in sorted_uids:
        # Skip if uid not present in labels
        if uid not in targets:
            # skip silently or log if desired
            continue

        lab = targets[uid]

        # Skip examples that exceed per-example limits
        if len(lab) > maxlen:
            total_skipped_too_long += 1
            continue
        if size > maxImagesize:
            total_skipped_too_large += 1
            continue

        feat = features[uid]

        # Compute hypothetical batch metrics if we add this example
        hypothetical_biggest = max(cur_biggest, size)
        hypothetical_count = cur_count + 1
        hypothetical_batch_area = hypothetical_biggest * hypothetical_count

        # If adding would exceed limits or reach batch_size, flush current batch first
        if cur_count > 0 and (hypothetical_batch_area > batch_Imagesize or cur_count >= batch_size):
            feature_batches.append(cur_feat_batch)
            label_batches.append(cur_label_batch)
            # reset
            cur_feat_batch = []
            cur_label_batch = []
            cur_biggest = 0
            cur_count = 0
            # recompute after reset
            hypothetical_biggest = size
            hypothetical_count = 1

        # Add to current batch
        cur_feat_batch.append(feat)
        cur_label_batch.append(lab)
        cur_biggest = max(cur_biggest, size)
        cur_count += 1

    # Append last batch if non-empty
    if cur_count > 0:
        feature_batches.append(cur_feat_batch)
        label_batches.append(cur_label_batch)

    total_batches = len(feature_batches)
    len_label_file = len(labels)
    len_ignore = len_label_file - sum(len(b) for b in label_batches)

    print(f"Total {total_batches} batches loaded")
    print(f"Ignored {len_ignore} images (too long or too large). Skipped_too_long={total_skipped_too_long}, skipped_too_large={total_skipped_too_large}")

    return feature_batches, label_batches
import numpy as np
from scipy.spatial.distance import pdist, squareform

from paramhandling.paramhandler import parcheck, get_nparrays, get_classes

def spread(Q: np.ndarray, y: np.ndarray, factor_h: float, factor_k: float, classes: np.ndarray | None = None
           ) -> float:
    """
    Calculates a "spread" score for samples in a similarity space.

    Each row of the input matrix Q is treated as a point in an n-classes
    dimensional space, and the distances are the Euclidean distances between
    these points.

    Parameters:
        Q (np.ndarray): An (M, N) similarity matrix where M is the number of samples
                        and N is the number of classes. Q[i, c] is the similarity
                        of sample i to class c. These rows are treated as points
                        in an N-dimensional space.
        y (np.ndarray): An (M,) array of labels, where y[i] is the integer class
                        label for sample i.
        factor_h (float): A scaled factor from the RBF kernel bandwidth parameter.
        factor_k (float): A scaled factor from the number of nearest neighbors used in
                        the sparse RBF kernel.
        classes (np.ndarray | None): The complete list of unique class labels. If provided,
                                     it's used to define the class space. If None,
                                     classes are inferred from y.

    Returns:
        float: The calculated spread score. Returns 0.0 if there are fewer
               than two samples.

    Raises:
        TypeError: If Q or y cannot be converted to numpy arrays.
        ValueError: If Q is not a 2D array, y is not a 1D array, or if the number of samples in Q and y do not match.
        MemoryError: If the pairwise distance matrix is too large to fit in memory.
    """

    parcheck(Q, y, factor_h, factor_k, classes)
    Q, y = get_nparrays(Q, y)
    
    n_samples = Q.shape[0]

    # --- Core Calculation ---
    # For performance, we compute all pairwise distances at once in a vectorized
    # operation and store them in a square matrix. This is much faster than
    # Python loops but uses O(n_samples^2) memory.
    try:
        # pdist computes the condensed distance matrix, squareform converts it
        # to a full symmetric matrix.
        pairwise_dists = squareform(pdist(Q, metric="euclidean"))
    except MemoryError:
        raise MemoryError(
            f"Failed to create a {n_samples}x{n_samples} pairwise distance matrix. "
            "The input array Q is too large for this memory-intensive approach."
        )

    # Create a boolean matrix where True indicates a pair of samples
    # belongs to the same class. This is done via broadcasting for efficiency.
    is_same_class_mask = y[:, None] == y[None, :]

    # Exclude the main diagonal from all calculations, as the distance
    # from a sample to itself is always zero and not meaningful here.
    np.fill_diagonal(is_same_class_mask, False)

    # --- Average Within-Class Distance ---
    # Use the mask to select only the distances between samples of the same class.
    within_distances = pairwise_dists[is_same_class_mask]
    num_within_pairs = within_distances.size

    # Avoid division by zero if there are no within-class pairs (e.g., every
    # sample is in its own unique class).
    avg_within_dist = (
        within_distances.sum() / num_within_pairs if num_within_pairs > 0 else 0.0
    )
    # Standard deviation requires at least two samples.
    std_within_dist = np.std(within_distances) if num_within_pairs > 1 else 0.0

    # --- Average Between-Class Distance ---
    # The "different class" mask is the logical inverse of the "same class" mask.
    is_different_class_mask = ~is_same_class_mask
    np.fill_diagonal(is_different_class_mask, False)

    between_distances = pairwise_dists[is_different_class_mask]
    num_between_pairs = between_distances.size

    # Avoid division by zero if all samples belong to the same class.
    avg_between_dist = (
        between_distances.sum() / num_between_pairs if num_between_pairs > 0 else 0.0
    )
    std_between_dist = np.std(between_distances) if num_between_pairs > 1 else 0.0

    # Those factors constantly yeld good results. Please, do not change them.
    return (
        float(
            (avg_between_dist * avg_within_dist) - (std_between_dist * std_within_dist)
        ) * (1 - factor_h) * (1 - factor_k)
    )

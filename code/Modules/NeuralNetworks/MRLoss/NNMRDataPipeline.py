#PyTorch
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def symmetric_swap_pytorch_batch(X, symmetric_pos, offset=10):
    """
    Swap specified symmetric positions in a batch of feature vectors.

    For each pair of indices (i, j) in `symmetric_pos`, this function swaps:
        1. The elements at positions i and j in the base feature vector.
        2. The elements at positions (offset + i) and (offset + j).

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (N, D), where N is batch size and D is feature dimension.
    symmetric_pos : list of tuple[int, int]
        List of index pairs (i, j) representing symmetric positions to swap.
    offset : int, optional
        Offset in the feature vector for symmetry-related positions (default: 10).

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as `X` with symmetric positions swapped.

    Notes
    -----
    This is useful when the feature vector encodes position-specific properties
    in both a base range `[0, offset)` and an offset range `[offset, offset + ... )`.
    """
    X_swapped = X.clone()
    for i, j in symmetric_pos:
        i_idx = offset + i
        j_idx = offset + j
        tmp = X_swapped[:, i_idx].clone()
        X_swapped[:, i_idx] = X_swapped[:, j_idx]
        X_swapped[:, j_idx] = tmp
        # Swap positions i <-> j
        tmp = X_swapped[:, i].clone()
        X_swapped[:, i] = X_swapped[:, j]
        X_swapped[:, j] = tmp
    return X_swapped

def batch_create_nlp(X1, X2, y1, y2, symmetric_pos=[(5,9),(6,8)], offset=10):
    """
    Create augmented pairwise inputs for neural ranking models.

    Generates pairwise feature vectors and labels for ranking tasks, including:
        - Base pairs: (X1_i, X2_i)
        - Flipped pairs: (X2_i, X1_i)
        - Symmetric augmented pairs with swaps at specified positions
        - Symmetric flipped pairs

    Parameters
    ----------
    X1, X2 : torch.Tensor
        Feature tensors of shape (N, D) for the first and second items in pairs.
    y1, y2 : torch.Tensor
        Target values of shape (N,) corresponding to X1 and X2.
    symmetric_pos : list of tuple[int, int], optional
        Positions to swap for symmetric augmentation (default: [(5,9),(6,8)]).
    offset : int, optional
        Offset for symmetry-sensitive positions (default: 10).

    Returns
    -------
    tuple:
        all_pairs : torch.Tensor
            Tensor of shape (8N, D) containing all augmented pairs.
        all_labels : torch.Tensor
            Tensor of shape (4N,) with binary ranking labels (+1 if first < second, -1 otherwise).
            Each label corresponds to a **pair of consecutive rows** in `all_pairs`.
        all_true_labels : torch.Tensor
            Tensor of shape (4N,) with normalized continuous differences: abs(y1 - y2)/min(y1, y2).
            Same correspondence as `all_labels`.
    """
    N, D = X1.shape

    # Helper to create pairs in ABAB... format
    def interleave_pairs(A, B):
        return torch.stack([A, B], dim=1).reshape(-1, D)

    # Step 1: Base pairs
    base_pairs = interleave_pairs(X1, X2)
    base_labels = (y1 < y2).float() * 2 - 1 
    base_true_labels = torch.abs(y1 - y2)/torch.minimum(y1,y2)
    
    # Step 2: Add flipped (B, A) with inverse label
    flipped_pairs = interleave_pairs(X2, X1)
    flipped_labels = (y1 > y2).float() * 2 - 1 
    flipped_true_labels = torch.abs(y1 - y2)/torch.minimum(y1,y2)
    # Step 3: Symmetric transformations
    X1_sym = symmetric_swap_pytorch_batch(X1, symmetric_pos, offset)
    X2_sym = symmetric_swap_pytorch_batch(X2, symmetric_pos, offset)

    sym_pairs = torch.cat([
        interleave_pairs(X1_sym, X2),
        interleave_pairs(X1, X2_sym),
        interleave_pairs(X1_sym, X2_sym)
    ], dim=0)

    sym_labels = base_labels.repeat(3)
    sym_true_labels = base_true_labels.repeat(3)
    # Step 4: Symmetric flipped pairs (reverse order)
    sym_pairs_flipped = torch.cat([
        interleave_pairs(X2, X1_sym),
        interleave_pairs(X2_sym, X1),
        interleave_pairs(X2_sym, X1_sym)
    ], dim=0)

    sym_labels_flipped = flipped_labels.repeat(3)
    sym_true_labels_flipped = flipped_true_labels.repeat(3)
    
    # Final assembly
    all_pairs = torch.cat([
        base_pairs,
        flipped_pairs,
        sym_pairs,
        sym_pairs_flipped
    ], dim=0)  # (8N, D)

    all_labels = torch.cat([
        base_labels,
        flipped_labels,
        sym_labels,
        sym_labels_flipped
    ], dim=0)  # (4N,)
    
    all_true_labels = torch.cat([
        base_true_labels,
        flipped_true_labels,
        sym_true_labels,
        sym_true_labels_flipped
    ], dim=0)  # (4N,)
    return all_pairs, all_labels,all_true_labels

# --- Final nonlinear_pairs ---
def nonlinear_pairs(X, y, DOIs=None, symmetric_pos=[(5,9),(6,8)], offset=10, max_pairs=None):
    """
    Generate augmented pairwise inputs for nonlinear ranking tasks.

    Constructs pairs (i, j) from input tensors X and y, optionally filtered by DOI,
    excluding identical y-values. Each valid pair is then augmented using `batch_create_nlp`
    to include flipped and symmetric variations.

    Parameters
    ----------
    X : torch.Tensor
        Tensor of shape (N, D) containing feature vectors.
    y : torch.Tensor
        Tensor of shape (N,) with target values.
    DOIs : list or array-like, optional
        Identifiers for each sample. Only pairs with matching DOI are considered. Default: None.
    symmetric_pos : list of tuple[int, int], optional
        Positions to swap for symmetric augmentation (default: [(5,9),(6,8)]).
    offset : int, optional
        Offset for symmetry-sensitive region in X (default: 10).
    max_pairs : int, optional
        Maximum number of valid pairs to consider before augmentation. Default: None (all pairs).

    Returns
    -------
    tuple:
        all_pairs : torch.Tensor
            Tensor of shape (8M, D) with base, flipped, and symmetric augmented pairs.
        all_labels : torch.Tensor
            Tensor of shape (4M,s
            Tensor of shape (4M,) with normalized continuous differences: abs(y1 - y2)/min(y1, y2).

    Notes
    -----
    - Pairs are selected from the upper triangle (i < j) to avoid duplicates.
    - Identical y-values are excluded.
    - Symmetric swaps are applied according to `symmetric_pos` and `offset`.
    - Fully vectorized for efficient batch processing.
    """
    torch.manual_seed(42)
    N = len(y)
    i_idx, j_idx = torch.triu_indices(N, N, offset=1).to(DEVICE)

    # Shuffle for randomness
    perm = torch.randperm(i_idx.shape[0])
    i_idx = i_idx[perm]
    j_idx = j_idx[perm]

    # Filter by matching DOIs and y[i] ≠ y[j]
    if DOIs is not None:
        doi_mask = torch.tensor([DOIs[i] == DOIs[j] for i, j in zip(i_idx, j_idx)], device=DEVICE)
        y_mask = torch.abs(y[i_idx] - y[j_idx]) > 1e-6
        valid = doi_mask & y_mask
    else: 
        y_mask = torch.abs(y[i_idx] - y[j_idx]) > 1e-6
        valid = y_mask
    i_valid = i_idx[valid]
    j_valid = j_idx[valid]

    if max_pairs is not None:
        i_valid = i_valid[:max_pairs]
        j_valid = j_valid[:max_pairs]

    X1, X2 = X[i_valid], X[j_valid]
    y1, y2 = y[i_valid], y[j_valid]

    return batch_create_nlp(X1, X2, y1, y2, symmetric_pos, offset)

#PyTorch
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def symmetric_swap_pytorch_batch(X, symmetric_pos, offset=10):
    """
    Perform symmetric swaps over a batch of feature vectors.

    This function swaps feature values at predefined symmetric positions,
    both in the base feature indices and in a secondary block shifted by
    `offset`.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (N, D) where N is batch size, D is feature dimension.
    symmetric_pos : list of tuple[int, int]
        List of index pairs (i, j) representing symmetric positions to swap.
        Each swap is performed both at indices (i, j) and (offset+i, offset+j).
    offset : int, optional
        Starting index in the feature vector for the secondary symmetry-sensitive block.
        Default is 10.

    Returns
    -------
    torch.Tensor
        New tensor of shape (N, D) with the specified symmetric positions swapped.

    Notes
    -----
    This is useful for feature representations where positions are symmetric
    (e.g., substituents on opposite sides of a molecular core).
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
    Create augmented ranking pairs with symmetric transformations.

    Given two sets of items (X1, X2) and their target values (y1, y2),
    this function generates 8x augmented training data for learning-to-rank:

        - Base pairs: (A, B)
        - Flipped pairs: (B, A)
        - Symmetric swaps of A and/or B
        - Flipped versions of symmetric swaps

    Parameters
    ----------
    X1, X2 : torch.Tensor
        Tensors of shape (N, D), feature vectors of paired items.
    y1, y2 : torch.Tensor
        Tensors of shape (N,), target values for items in X1 and X2.
    symmetric_pos : list of tuple[int, int], optional
        Symmetric feature index pairs to swap during augmentation.
        Default is [(5, 9), (6, 8)].
    offset : int, optional
        Offset index for the second symmetry-sensitive block. Default is 10.

    Returns
    -------
    all_pairs : torch.Tensor
        Tensor of shape (8N, D) containing augmented pairs stacked as
        [A0, B0, A1, B1, ...].
    all_labels : torch.Tensor
        Tensor of shape (4N,) containing binary labels:
            - 0 if the first item has a smaller target value than the second (y1 < y2)
            - 1 if the first item has a larger target value (y1 > y2)

    Notes
    -----
    Equal target values (y1 == y2) should be excluded before calling this function.
    """
    N, D = X1.shape

    # Helper to create pairs in ABAB... format
    def interleave_pairs(A, B):
        return torch.stack([A, B], dim=1).reshape(-1, D)

    # Step 1: Base pairs
    base_pairs = interleave_pairs(X1, X2)
    base_labels = (y1 > y2).float()

    # Step 2: Add flipped (B, A) with inverse label
    flipped_pairs = interleave_pairs(X2, X1)
    flipped_labels = (y2 > y1).float()

    # Step 3: Symmetric transformations
    X1_sym = symmetric_swap_pytorch_batch(X1, symmetric_pos, offset)
    X2_sym = symmetric_swap_pytorch_batch(X2, symmetric_pos, offset)

    sym_pairs = torch.cat([
        interleave_pairs(X1_sym, X2),
        interleave_pairs(X1, X2_sym),
        interleave_pairs(X1_sym, X2_sym)
    ], dim=0)

    sym_labels = base_labels.repeat(3)

    # Step 4: Symmetric flipped pairs (reverse order)
    sym_pairs_flipped = torch.cat([
        interleave_pairs(X2, X1_sym),
        interleave_pairs(X2_sym, X1),
        interleave_pairs(X2_sym, X1_sym)
    ], dim=0)

    sym_labels_flipped = flipped_labels.repeat(3)

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

    return all_pairs, all_labels

# --- Final nonlinear_pairs ---
def nonlinear_pairs(X, y, DOIs=None, symmetric_pos=[(5,9),(6,8)], offset=10, max_pairs=None):
    """
    Generate augmented ranking pairs from a dataset.

    This function enumerates valid item pairs, optionally filters them by
    identifier (DOI) and target difference, and expands them using
    `batch_create_nlp` into an 8x augmented training set.

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix of shape (N, D).
    y : torch.Tensor
        Target values of shape (N,).
    DOIs : list or array-like, optional
        Identifiers of length N (e.g., molecule IDs). If provided, only pairs
        with matching DOIs are kept.
    symmetric_pos : list of tuple[int, int], optional
        Symmetric feature index pairs to swap during augmentation.
        Default is [(5, 9), (6, 8)].
    offset : int, optional
        Offset index for the second symmetry-sensitive block. Default is 10.
    max_pairs : int, optional
        Maximum number of unique base pairs before augmentation.
        Default is None (use all).

    Returns
    -------
    all_pairs : torch.Tensor
        Tensor of shape (8M, D), where M is the number of valid base pairs.
    all_labels : torch.Tensor
        Tensor of shape (4M,) with binary labels:
            - 0 if the first item has a smaller target value (y[i] < y[j])
            - 1 if the first item has a larger target value (y[i] > y[j])

    Notes
    -----
    Steps:
        1. Enumerate all i < j index pairs.
        2. Shuffle for randomness.
        3. Filter by DOI (if given) and exclude equal target values.
        4. Keep at most `max_pairs` pairs.
        5. Apply `batch_create_nlp` to expand each base pair into 8 augmented variants.
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

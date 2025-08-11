# Data analysis and math
import numpy as np 
INTERACTING_PAIRS = [(1,2),(2,3),(3,4),(5,6),(5,7),(5,8),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9),(0,5),(0,6),(0,7),(0,8),(0,9)]

def FirstDegreeFP(sub_dict,n_atoms=10,n_subs=2):
    """
    Generate a first-degree binary fingerprint encoding the presence of substituent types 
    at defined atom positions in a molecular core.

    The fingerprint consists of two binary "rows" concatenated and flattened:
    - Row 0 (first n_atoms bits): indicates presence of oxygen-only substituents (e.g., -OH) at each atom position.
    - Row 1 (next n_atoms bits): indicates presence of carbon-containing substituents (e.g., -CH3, -OCH3) at each position.

    Parameters
    ----------
    sub_dict : dict of int -> list of str
        Mapping from core atom indices to lists of atomic symbols representing substituents, 
        typically returned by a function like `get_sub_dict`.
    n_atoms : int, optional
        Number of atom positions in the core to consider (default is 10).
    n_subs : int, optional
        Number of substituent types tracked (default is 2: oxygen-only and carbon-containing).

    Returns
    -------
    np.ndarray, shape (n_atoms * n_subs,)
        Flattened binary fingerprint vector. Each bit corresponds to presence (1) or absence (0) 
        of a substituent type at a specific core atom position, ordered by substituent type first, 
        then position.
    """
    fp = np.zeros((n_subs,n_atoms))
    for key, sub in sub_dict.items():
        if 'C' in sub:
            fp[1][key] = 1 # Carbon-containing group at position key
        elif 'O' in sub:
            fp[0][key] = 1 
    return fp.reshape(-1)

def SecondDegreeFP_bits(bits,n_atoms=10,n_subs=2,pairs_to_account=INTERACTING_PAIRS):
    """
    Computes a second-degree molecular fingerprint by incorporating pairwise interactions 
    between substituents at specified core atom positions.

    Starting from a flattened first-degree fingerprint vector that encodes substituent 
    presence by type and position, this function computes pairwise interaction features 
    via outer products between substituent vectors at pairs of positions. Only interactions 
    for specified pairs are included in the output.

    Parameters
    ----------
    bits : np.ndarray, shape (n_atoms * n_subs,)
        Flattened first-degree fingerprint vector, typically produced by `FirstDegreeFP`.
        It encodes substituent presence per position and substituent type.
    n_atoms : int, optional
        Number of core atom positions in the fingerprint (default: 10).
    n_subs : int, optional
        Number of substituent types per position (default: 2).
    pairs_to_account : list of tuple of int, optional
        List of (i, j) pairs of atom indices for which pairwise interaction features 
        should be computed and included in the output.

    Returns
    -------
    np.ndarray
        Extended fingerprint vector concatenating:
        - The original first-degree fingerprint (`bits`),
        - The flattened pairwise interaction features computed for the specified position pairs.

        Each interaction feature encodes combined presence patterns of substituent types 
        at two positions as a 4-element binary vector (from outer product of 2-bit substituent vectors).
    """
    fp_1d = bits  # shape: (20,)
    fp_2d = fp_1d.reshape((n_subs, n_atoms)).T# shape: (10, 2)
    # Compute outer products between all pairs: shape (10, 10, 2, 2)
    B = np.einsum('ik,jl->ijkl', fp_2d, fp_2d)
    B_flat = B.reshape(B.shape[0], B.shape[1], -1)  # shape: (10, 10, 4)

    # Collect selected pairwise features
    selected = [B_flat[i, j] for (i, j) in pairs_to_account]
    addition = np.concatenate(selected, axis=0) if selected else np.array([])
    # Concatenate original fingerprint with second-degree additions
    return np.concatenate([fp_1d, addition])
    
def SecondDegreeFP(sub_dict, pairs_to_account=INTERACTING_PAIRS):
    """
    Computes a second-degree binary fingerprint for a molecule by capturing:

    1. First-degree features: Presence or absence of substituent types at specific core atom positions.
    2. Second-degree features: Pairwise interaction terms between substituents at selected position pairs.

    This function first generates a first-degree fingerprint using `FirstDegreeFP` and then 
    augments it with selected second-degree interaction features via `SecondDegreeFP_bits`.

    Parameters
    ----------
    sub_dict : dict of int -> list of str
        Dictionary mapping core atom indices to lists of substituent atomic symbols,
        typically obtained from `get_sub_dict`.
    pairs_to_account : list of tuple of int, optional
        List of (i, j) pairs of atom indices for which pairwise interaction features should 
        be computed and appended to the fingerprint.

    Returns
    -------
    np.ndarray
        Concatenated binary fingerprint vector including:
        - First-degree presence bits (length = n_atoms * n_subs),
        - Selected second-degree interaction bits corresponding to specified pairs.

        The exact length depends on the number of pairs accounted for.
    """
    fp_1d = FirstDegreeFP(sub_dict)  # shape: (20,)
    return SecondDegreeFP_bits(fp_1d,pairs_to_account=pairs_to_account)

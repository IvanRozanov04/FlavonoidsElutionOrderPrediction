import rdkit as rk
import numpy as np
from Modules.FingerPrints.MolGraph import BFS

def get_sub_dict(mol,mcs_mol):
    """
    Extract substituent groups attached to the core structure (MCS) of a molecule.

    This function identifies all atoms not belonging to the core (defined by the MCS),
    excluding hydrogens directly bonded to the core. It then builds connectivity graphs
    to group substituents attached to each core atom using breadth-first search (BFS).
    The output maps each core atom index to a list of atomic symbols of its substituent group.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The full molecule.
    mcs_mol : rdkit.Chem.Mol
        The Maximum Common Substructure (MCS) molecule defining the molecular core.

    Returns
    -------
    dict of int -> list of str
        Dictionary mapping core atom indices (in MCS) to lists of atomic symbols (str)
        representing substituent groups attached to those core atoms.

    Notes
    -----
    - Hydrogens bonded directly to core atoms are excluded from substituent groups.
    - Substituent groups are identified by BFS over their internal connectivity.
    - Different substituent groups connected to different core atoms are treated separately.
    """
    # Get the mapping of core atom indices in the input molecule
    mcs_match = mol.GetSubstructMatch(mcs_mol)
    
    # Compute the adjacency matrix of the molecule
    adj = rk.Chem.rdmolops.GetAdjacencyMatrix(mol)

    # Identify explicit hydrogen atoms attached directly to the core
    h_attached_to_core = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1 and neighbors[0].GetIdx() in mcs_match:
                h_attached_to_core.add(atom.GetIdx())

    # Identify valid substituent atoms:
    # those that are not in the core and not hydrogens attached to the core
    valid_substituent_atoms = [
        i for i in range(mol.GetNumAtoms())
        if i not in mcs_match and i not in h_attached_to_core
    ]

    # Build matrices A (core-substituent connectivity) and B (internal substituent graph)
    A = np.zeros((len(valid_substituent_atoms), len(mcs_match)))  # shape: [subs, core]
    B = np.zeros((len(valid_substituent_atoms), len(valid_substituent_atoms)))  # shape: [subs, subs]

    # Fill matrices A and B
    for i_new, i_old in enumerate(valid_substituent_atoms):
        for j, j_old in enumerate(mcs_match):
            if adj[i_old, j_old] == 1:
                A[i_new, j] = 1
        for j_new, j_old in enumerate(valid_substituent_atoms):
            if adj[i_old, j_old] == 1:
                B[i_new, j_new] = 1

    # Get atomic symbols for substituent atoms
    atom_labels = [mol.GetAtomWithIdx(i).GetSymbol() for i in valid_substituent_atoms]

    # Initialize dictionary for storing substituent groups
    sub_dict = dict()

    # For each substituent atom connected to the core, do a BFS over B to extract the full group
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                G = BFS(B, i, atom_labels)  # BFS returns a graph structure with discovery info
                l = []
                for v in G.vertices:
                    if v.discovered == 1:
                        l.append(v.type)
                if len(l) != 0:
                    sub_dict[j] = l  # j is the index of the core atom (in MCS)
                    
    return sub_dict
class Vertex:
    """
    Represents a vertex in a graph, typically corresponding to an atom in a molecular structure.

    Attributes:
    ----------
    id : int
        Index of the vertex (atom index in the molecule).
    type : str
        Atom symbol (e.g., 'C', 'O', 'N').
    dist : int
        Distance from the source node during BFS traversal.
    discovered : int
        Flag indicating whether the vertex has been discovered (0 = no, 1 = yes).
    pi : Vertex or None
        Predecessor (parent) vertex in the BFS tree.
    """
    def __init__(self,i,atom_type) -> None:
        """
        Initializes a Vertex.

        Parameters:
        ----------
        i : int
            The index of the vertex in the graph.
        atom_type : str
            Atom type (symbol) associated with this vertex.
        """
        self.type = atom_type
        self.id = i
        self.dist = 0
        self.discovered = 0
        self.pi = None
        

class Graph:
    """
    Represents a molecular graph constructed from an adjacency matrix and atom labels.

    Attributes:
    ----------
    atom_labels : list of str
        List of atom symbols corresponding to each vertex.
    vertices : list of Vertex
        List of Vertex objects for each atom.
    adj_list : list of list of Vertex
        Adjacency list representation of the graph.
    """
    def __init__(self,adjacency_matrix,atom_labels):
        """
        Constructs the Graph from an adjacency matrix and list of atom labels.

        Parameters:
        ----------
        adjacency_matrix : np.ndarray
            Adjacency matrix of the molecule's subgraph (e.g., substituents).
        atom_labels : list of str
            List of atom symbols corresponding to each row/column in the matrix.
        """
        self.atom_labels = atom_labels
        self.vertices = [Vertex(i,atom_type=atom_labels[i]) for i in range(adjacency_matrix.shape[0])]
        self.adj_list = self.adjacency_list(adjacency_matrix)

    def adjacency_list(self,adjacency_matrix):
        """
        Converts an adjacency matrix into an adjacency list.

        Parameters:
        ----------
        adjacency_matrix : np.ndarray
            The adjacency matrix to convert.

        Returns:
        -------
        list of list of Vertex
            Adjacency list where each entry contains the neighboring vertices for a given vertex.
        """
        l = []
        for i in range(adjacency_matrix.shape[0]):
            l.append([self.vertices[i] for i in list(adjacency_matrix[i].nonzero()[0])])
        return l
    

def BFS(adj_matrix,i,atom_labels):
    """
    Performs Breadth-First Search (BFS) on a molecular subgraph starting from a given vertex.

    Used to identify connected components (e.g., substituents) attached to a molecule’s core.

    Parameters:
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix representing the connectivity of substituent atoms.
    i : int
        Index of the starting atom for BFS traversal.
    atom_labels : list of str
        List of atomic symbols corresponding to each node in the graph.

    Returns:
    -------
    Graph
        A Graph object with updated `dist`, `discovered`, and `pi` attributes for each vertex
        indicating the BFS traversal tree.
    """
    G = Graph(adj_matrix,atom_labels)
    queue = [G.vertices[i]]
    while len(queue) != 0:
        curr = queue.pop(0)
        if curr.discovered == 0:
            curr.discovered = 1
            for v in G.adj_list[curr.id]:
                v.dist = curr.dist + 1
                v.pi = curr
                queue.append(v)
    return G
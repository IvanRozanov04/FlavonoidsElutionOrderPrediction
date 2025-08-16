#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Modules.NeuralNetworks.BCELoss.NNBCEDataPipeline import nonlinear_pairs
class RankNetDataset(Dataset):
    """
    PyTorch Dataset for RankNet training.

    This dataset assumes the input `parameters` array contains concatenated feature
    vectors for item pairs, such that for the k-th label:
        - The features for item A are at index 2*k
        - The features for item B are at index 2*k + 1

    Parameters
    ----------
    parameters : array-like or torch.Tensor
        Feature vectors for all items, ordered as (A1, B1, A2, B2, ...).
    labels : array-like or torch.Tensor
        Binary labels indicating pairwise preference:
            - 0 if itemA is ranked lower (yA < yB)
            - 1 if itemA is ranked higher (yA > yB)

    Methods
    -------
    __len__():
        Returns the number of pairs in the dataset.
    __getitem__(idx):
        Returns the features of item A, item B, and the label for the given pair index.
    """
    def __init__(self, parameters,labels):
        self.parameters = parameters
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        itemA, itemB, label = self.parameters[2*idx],self.parameters[2*idx + 1],self.labels[idx]
        return itemA, \
               itemB, \
               label

class RankNet(nn.Module):
    """
    Feed-forward neural network implementing the RankNet scoring function.

    The model takes a feature vector and outputs a single score, which can be
    used to rank items. Pairwise differences in scores are used for training.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input feature vectors.
    hidden_dim : int, optional
        Number of units in each hidden layer (default: 32).
    n_hidden : int, optional
        Number of hidden layers (default: 2).

    Architecture
    ------------
    input_layer  -> Tanh -> residual_hidden_layers -> output_layer

    The residual_hidden_layers add the layer output to the input (skip connection)
    before passing to the next layer.

    Methods
    -------
    forward(x):
        Computes the RankNet score for input x.
    """
    def __init__(self,input_dim, hidden_dim=32, n_hidden=2):
        super(RankNet, self).__init__()
        self.activation = nn.Tanh()
        self.input_layer = nn.Linear(input_dim,hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(hidden_dim,1)
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x)) + x
        score = self.output_layer(x)
        return score

def DataloaderRankNet(X,y,dois=None,batch_size=256):
    """
    Create a PyTorch DataLoader for RankNet training.

    This function generates pairwise training examples from feature matrix X and
    target values y using `nonlinear_pairs`. Each pair is represented as two
    consecutive feature vectors in the returned dataset.

    Parameters
    ----------
    X : array-like or torch.Tensor
        Feature matrix for all items.
    y : array-like or torch.Tensor
        Target values used for generating pairwise preferences.
    dois : array-like or None, optional
        Optional grouping or identifier values used by `nonlinear_pairs` to
        restrict pair generation within the same group. If None, pairs are formed
        globally.
    batch_size : int, optional
        Number of pairs per batch in the DataLoader (default: 256).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader providing batches of (itemA, itemB, label) tuples
        for RankNet training, where:
            - label = 0 if itemA < itemB
            - label = 1 if itemA > itemB
    """
    pairs, labels = nonlinear_pairs(X,y,dois)
    train_dataset = RankNetDataset(pairs,labels)
    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    return train_loader


def train_RankNet(model, train_loader, num_epochs=10,criterion=nn.BCEWithLogitsLoss(), optimizer = None):
    """
    Train a RankNet model using pairwise preference learning.

    Parameters
    ----------
    model : torch.nn.Module
        RankNet model instance to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding batches of (itemA, itemB, label) tuples.
        - itemA, itemB : tensors of shape (batch_size, n_features)
        - label : tensor of shape (batch_size,), where
            * 0 means itemA < itemB
            * 1 means itemA > itemB
    num_epochs : int, optional
        Number of training epochs (default: 10).
    criterion : torch.nn.modules.loss._Loss, optional
        Loss function to use (default: nn.BCEWithLogitsLoss()).
    optimizer : torch.optim.Optimizer, optional
        Optimizer for updating model parameters. Must be provided.

    Notes
    -----
    The loss is computed as:
        loss = criterion(scoreA - scoreB, label)
    where scoreA and scoreB are scalar relevance scores predicted by the model.
    """
    torch.manual_seed(42)
    model.train()
    for _ in range(num_epochs):
        total_loss = 0
        for itemA, itemB, label in train_loader:
            scoreA = model(itemA).squeeze()
            scoreB = model(itemB).squeeze()
            loss = criterion(scoreA - scoreB, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()



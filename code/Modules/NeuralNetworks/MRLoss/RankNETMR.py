#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Modules.NeuralNetworks.MRLoss.NNMRDataPipeline import nonlinear_pairs
import torch.nn.functional as F 
def custom_margin_ranking_loss(score1, score2, target, margin):
    """
    Compute a custom margin ranking loss for a batch of item pairs.

    This loss encourages consistent pairwise ranking: 
    - If `target = 1`, `score1` is encouraged to be higher than `score2` by at least `margin`.  
    - If `target = -1`, `score2` is encouraged to be higher than `score1` by at least `margin`.  

    Unlike standard margin ranking, this formulation ensures that score updates are
    consistent with the pairwise ranking labels, so both scores increase or decrease
    together in a direction that respects the target ranking.

    Parameters
    ----------
    score1 : torch.Tensor
        Predicted scores for the first item in each pair, shape (batch_size,).
    score2 : torch.Tensor
        Predicted scores for the second item in each pair, shape (batch_size,).
    target : torch.Tensor
        Pairwise ranking labels: +1 if `score1` should rank higher, -1 if `score2` should rank higher.
    margin : torch.Tensor
        Margin for each pair, broadcastable to batch size, specifying the minimum desired score difference.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean margin ranking loss over the batch.
    """
    loss = F.relu(target * (score1 - score2) + margin)
    return loss.mean()

class RankNetDataset(Dataset):
    """
    Dataset for training a RankNet model on pairwise data.

    Each item consists of two feature vectors forming a pair, 
    the pairwise label, and the true margin/label.

    Parameters
    ----------
    parameters : list or torch.Tensor
        Flattened list of feature vectors where pairs are consecutive items.
    labels : list or torch.Tensor
        Ranking labels for each pair (+1 or -1), shape (num_pairs,).
    true_labels : list or torch.Tensor
        True margins or differences for each pair, shape (num_pairs,).
    """
    def __init__(self, parameters,labels,true_labels):
        self.parameters = parameters
        self.labels = labels
        self.true_labels = true_labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve the idx-th pair and its labels.

        Returns
        -------
        tuple
            (itemA_features, itemB_features, label, true_label)
        """
        itemA, itemB, label,true_label = self.parameters[2*idx],self.parameters[2*idx + 1],self.labels[idx],self.true_labels[idx]
        return itemA, \
               itemB, \
               label, true_label 

class RankNet(nn.Module):
    """
    Feedforward RankNet model with residual hidden layers.

    Parameters
    ----------
    input_dim : int
        Number of features in the input vector.
    hidden_dim : int, optional
        Number of neurons in hidden layers (default=32).
    n_hidden : int, optional
        Number of hidden layers (default=2).

    Architecture
    ------------
    Input Layer: Linear(input_dim, hidden_dim) + Tanh
    Hidden Layers: n_hidden x [Linear(hidden_dim, hidden_dim) + Tanh + Residual]
    Output Layer: Linear(hidden_dim, 1)
    """
    def __init__(self,input_dim, hidden_dim=32, n_hidden=2):
        super(RankNet, self).__init__()
        self.activation = nn.Tanh()
        self.input_layer = nn.Linear(input_dim,hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(hidden_dim,1)
    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted score tensor of shape (batch_size, 1).
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x)) + x
        score = self.output_layer(x)
        return score

def DataloaderRankNet(X,y,dois=None,batch_size=256):
    """
    Create a DataLoader for RankNet training using pairwise data.

    Parameters
    ----------
    X : array-like or torch.Tensor
        Feature matrix of shape (num_samples, num_features).
    y : array-like or torch.Tensor
        Target values or ranks for each sample.
    dois : optional
        Additional information per sample (used in nonlinear_pairs).
    batch_size : int, optional
        Batch size for DataLoader (default=256).

    Returns
    -------
    DataLoader
        PyTorch DataLoader yielding batches of (itemA, itemB, label, true_label)
    """
    pairs, labels,true_labels = nonlinear_pairs(X,y,dois)
    train_dataset = RankNetDataset(pairs,labels,true_labels)
    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    return train_loader


def train_RankNet(model, train_loader, num_epochs=10,criterion=custom_margin_ranking_loss, optimizer = None):
    """
    Train a RankNet model using pairwise ranking loss.

    Parameters
    ----------
    model : RankNet
        Instance of the RankNet model.
    train_loader : DataLoader
        DataLoader providing pairwise batches.
    num_epochs : int, optional
        Number of training epochs (default=10).
    criterion : callable, optional
        Loss function accepting (score1, score2, label, margin) (default=custom_margin_ranking_loss).
    optimizer : torch.optim.Optimizer, optional
        Optimizer for model parameters. Must be provided.

    Returns
    -------
    None
        The model is trained in-place.
    """
    torch.manual_seed(42)
    model.train()
    for _ in range(num_epochs):
        total_loss = 0
        for itemA, itemB, label,true_label in train_loader:
            itemA = itemA
            itemB = itemB
            label = label

            scoreA = model(itemA).squeeze()
            scoreB = model(itemB).squeeze()
            loss = criterion(scoreA,scoreB,label,true_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

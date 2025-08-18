from sklearn.model_selection import StratifiedKFold
import numpy as np
from Modules.NeuralNetworks.BCELoss.RankNETBCE import train_RankNet,DataloaderRankNet, RankNet
import torch
from itertools import combinations
from Modules.HelperFunctions import compute_safe_linear_regression_metrics,compute_safe_interpolation_metrics
from Modules.LinearModels.CVLinearModels import PairwiseMetrics as PMetrics
from scipy.stats import kendalltau
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def PairwiseMetrics(y:np.array,pred:np.array,test_idx:np.array,formulas:np.array,):
    """
    Compute pairwise error rates for a ranking task (RankNet-style).

    Evaluates prediction performance by comparing the relative ordering
    of all sample pairs `(i, j)` where `i < j`. Computes:

    - **PER** (Pairwise Error Rate): fraction of pairs with incorrect order
    - **iPER** (Isomer-specific PER): same as PER but restricted to 
      pairs with identical molecular formulas (if `formulas` is provided)
    - **ni_***: number of isomeric pairs per split

    Pairs are categorized as:
    - **test**: both samples are in `test_idx`
    - **train**: both samples are not in `test_idx`
    - **mixed**: any pair not fully in train (includes cross train-test pairs)

    Parameters
    ----------
    y : np.array
        True target values of shape (N,).
    pred : np.array
        Predicted target values of shape (N,).
    test_idx : np.array
        Indices of the test set.
    formulas : np.array or None
        Molecular formulas or identifiers for detecting isomers.
        If None, isomer-specific metrics are skipped.

    Returns
    -------
    dict
        Dictionary containing:
        - `"PER_test"`, `"PER_train"`, `"PER_mixed"`: pairwise error rates
        - `"iPER_test"`, `"iPER_train"`, `"iPER_mixed"`: isomer-specific PER
        - `"ni_test"`, `"ni_train"`, `"ni_mixed"`: number of isomeric pairs
    """
    results = dict()
    test_idx = torch.tensor(test_idx,dtype=torch.long,device=DEVICE)
    N = len(y)
    i, j = torch.triu_indices(N, N, offset=1).to(DEVICE)
    
    valid_mask = y[i] != y[j]
    if formulas is not None:
        iso_mask = torch.as_tensor(formulas[i.cpu()] == formulas[j.cpu()],
                           dtype=torch.bool, device=DEVICE)

    test_mask = torch.isin(i, test_idx) & torch.isin(j, test_idx)
    train_mask = ~torch.isin(i, test_idx) & ~torch.isin(j, test_idx)
    mixed_mask = ~(train_mask)  # any other combo

    
    errors= (y[i] - y[j]) * (pred[i] - pred[j]) <= 0 
    
    results["PER_test"] = errors[test_mask & valid_mask].sum() / (test_mask & valid_mask).sum() if (test_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
    results["PER_train"] = errors[train_mask & valid_mask].sum() / (train_mask & valid_mask).sum() if (train_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
    results["PER_mixed"] = errors[mixed_mask & valid_mask].sum() / (mixed_mask & valid_mask).sum() if (mixed_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
    
    if formulas is not None:
        results["iPER_test"] = errors[test_mask & iso_mask & valid_mask].sum() / (test_mask & iso_mask & valid_mask).sum() if (test_mask & iso_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
        results["iPER_train"] = errors[train_mask & iso_mask & valid_mask].sum() / (train_mask & iso_mask & valid_mask).sum() if (train_mask & iso_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
        results["iPER_mixed"] = errors[mixed_mask & iso_mask & valid_mask].sum() / (mixed_mask & iso_mask & valid_mask).sum() if (mixed_mask & iso_mask & valid_mask).sum() != 0 else torch.tensor(float('nan'), device=DEVICE)
        
        results['ni_test'] = (test_mask & iso_mask & valid_mask).sum()
        results['ni_train'] = (train_mask & iso_mask & valid_mask).sum()
        results['ni_mixed'] = (mixed_mask & iso_mask & valid_mask).sum()

    return results

def RankNETCV(X, y, classes, hyperparameters_grid, n_itter=5,n_splits=5):
    """
    Perform cross-validation to select the best RankNet hyperparameters.

    Trains and evaluates RankNet models over a hyperparameter grid using
    stratified k-fold cross-validation. Multiple random shuffles of the folds
    are used to reduce variance. The hyperparameter set with the lowest 
    average PER_mixed across folds is selected.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Feature matrix of shape (N, D).
    y : np.ndarray or torch.Tensor
        Target values of shape (N,).
    classes : array-like
        Class labels for stratified splitting.
    hyperparameters_grid : list of dict
        Each dict must contain:
            - 'hidden_dim': int, hidden layer size
            - 'n_hidden': int, number of hidden layers
            - 'batch_size': int, batch size
            - 'num_epochs': int, number of training epochs
            - 'weight_decay': float, L2 regularization
    n_itter : int, optional
        Number of random repetitions for CV. Default is 5.
    n_splits : int, optional
        Number of folds per CV iteration. Default is 5.

    Returns
    -------
    RankNet
        Trained RankNet model using the best hyperparameters found.
    """
    results = []
    for i in range(n_itter):
        kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=i)
        for train_idx, test_idx in kf.split(X,classes):
            X_train,y_train = X[train_idx],y[train_idx]
            train_loader = DataloaderRankNet(X_train,y_train,None,batch_size=512)
            for hp in hyperparameters_grid:
                model = RankNet(X.shape[1],hidden_dim=hp['hidden_dim'],n_hidden=hp['n_hidden']).to(DEVICE)
                train_RankNet(model,train_loader=train_loader,num_epochs=hp['num_epochs'],optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=hp['weight_decay']))
                predictions = model(X).reshape(-1)
                metrics = PairwiseMetrics(y,predictions,test_idx,formulas=None)
                results.append(metrics['PER_mixed'])
    results_tensor = torch.stack(results).reshape(-1,len(hyperparameters_grid)).mean(axis=0) 
    best_hp = hyperparameters_grid[torch.argmin(results_tensor)]
    best_model = RankNet(X.shape[1],hidden_dim=best_hp['hidden_dim'],n_hidden=best_hp['n_hidden']).to(DEVICE)
    train_loader = DataloaderRankNet(X,y,None,batch_size=512)
    train_RankNet(best_model,train_loader=train_loader,num_epochs=best_hp['num_epochs'],optimizer=torch.optim.Adam(best_model.parameters(),lr=0.01,weight_decay=best_hp['weight_decay']))
    return best_model

def NestedCV(X, y, classes,formulas, hyperparameters_grid, inner_cv_func, n_outter_splits=5, n_inner_splits=5,n_itter_out=5,n_itter_inner=5,accumulator = None):
    """
    Perform nested cross-validation for hyperparameter tuning and model evaluation.

    This function implements a two-level cross-validation scheme:
    - **Outer loop**: Splits data to estimate unbiased model performance.
    - **Inner loop**: Selects the best hyperparameters using `inner_cv_func`.
    - Trains the model with selected hyperparameters on inner training folds
      and evaluates on the outer test fold using pairwise metrics.
    - Optionally updates an accumulator with predictions and targets for each fold.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Target values of shape (n_samples,).
    classes : array-like
        Class labels used for stratified splitting.
    formulas : array-like or None
        Molecular identifiers or groupings for isoform-specific metrics.
    hyperparameters_grid : list of dict
        Hyperparameter configurations to search during inner CV.
    inner_cv_func : callable
        Function returning a trained model given training data and hyperparameters.
        Signature: (X_train, y_train, classes_train, hyperparameters_grid,
                    n_itter_inner, n_inner_splits) -> trained_model
    n_outter_splits : int, optional
        Number of folds in the outer CV (default is 5).
    n_inner_splits : int, optional
        Number of folds in the inner CV (default is 5).
    n_itter_out : int, optional
        Number of outer CV repetitions (default is 5).
    n_itter_inner : int, optional
        Number of inner CV repetitions (default is 5).
    accumulator : object, optional
        Optional object with an `update(y, predictions, test_idx, formulas)` method
        for accumulating predictions and metrics across folds. If None, accumulation is skipped.

    Returns
    -------
    list of list of float
        Each element corresponds to an outer test fold and contains:
        [PER_test, PER_mixed, iPER_test, iPER_mixed, ni_test, ni_mixed]
        - PER_*: pairwise error rates
        - iPER_*: isoform-restricted pairwise error rates
        - ni_*: counts of isoform pairs evaluated

    Notes
    -----
    - Supports both NumPy arrays and PyTorch tensors as inputs.
    - Uses `PairwiseMetrics` to compute pairwise error rates.
    - Accumulator, if provided, is updated with predictions for each outer fold.
    """
    results = []
    for i in range(n_itter_out):
        kf = StratifiedKFold(n_outter_splits,shuffle=True,random_state=i)
        for train_idx, test_idx in kf.split(X,classes):
            X_train,y_train = X[train_idx],y[train_idx]
            classes_train = np.array(classes,dtype=int)[train_idx]
            
            best_model = inner_cv_func(X_train,y_train,classes_train,hyperparameters_grid,n_itter_inner,n_inner_splits)
            
            predictions = best_model(X).reshape(-1)
            metrics = PairwiseMetrics(y,predictions,test_idx,formulas)
            if accumulator is not None:
                accumulator.update(y.cpu().detach().numpy().reshape(-1),predictions.cpu().detach().numpy().reshape(-1),test_idx,formulas)
            results.append([metrics['PER_test'].cpu().detach().numpy(),metrics['PER_mixed'].cpu().detach().numpy(),metrics['iPER_test'].cpu().detach().numpy(),
                            metrics['iPER_mixed'].cpu().detach().numpy(),metrics['ni_test'].cpu().detach().numpy(),metrics['ni_mixed'].cpu().detach().numpy()])
    return results 


def val_PER(train_loader, X_val, y_val, hp):
    """
    Train a RankNet model and evaluate ranking performance on a validation set.

    Uses Kendall tau-based pairwise error metric: PER = (1 - Kendall tau) / 2.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    X_val : torch.Tensor
        Validation feature matrix (N_val, D).
    y_val : torch.Tensor
        Validation target vector (N_val,).
    hp : dict
        Hyperparameter dictionary:
            - 'hidden_dim': int
            - 'n_hidden': int
            - 'num_epochs': int
            - 'weight_decay': float

    Returns
    -------
    float
        Pairwise error rate (PER) on the validation set.
    """
    model = RankNet(20,hidden_dim=hp['hidden_dim'],n_hidden=hp['n_hidden']).to(DEVICE)
    train_RankNet(model,train_loader=train_loader,num_epochs=hp['num_epochs'],optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=hp['weight_decay']))
    preds = model(X_val).reshape(-1)
    CorrCoef = kendalltau(preds.detach().cpu().numpy(),y_val.detach().cpu().numpy()).statistic
    return (1 - CorrCoef)/2

def val_R2(train_loader, X_val, y_val, hp):
    """
    Train a RankNet model and evaluate prediction performance using R².

    R² metric: 1 - (Pearson correlation coefficient)^2.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    X_val : torch.Tensor
        Validation feature matrix (N_val, D).
    y_val : torch.Tensor
        Validation target vector (N_val,).
    hp : dict
        Hyperparameter dictionary:
            - 'hidden_dim': int
            - 'n_hidden': int
            - 'num_epochs': int
            - 'weight_decay': float

    Returns
    -------
    float
        Validation error based on R² (lower is better).
    """
    model = RankNet(20,hidden_dim=hp['hidden_dim'],n_hidden=hp['n_hidden']).to(DEVICE)
    train_RankNet(model,train_loader=train_loader,num_epochs=hp['num_epochs'],optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=hp['weight_decay']))
    preds = model(X_val).reshape(-1)
    CorrCoef = torch.corrcoef(torch.stack((preds,y_val)))[0][1] ** 2 
    return (1 - CorrCoef)


def ILD_cv(train_loader,X,y,classes,formulas,hyperparameters_grid,n_itter=1,n_splits=5,accumulator = None,val_type = 'R2'):
    """
    Perform cross-validated hyperparameter tuning and evaluation of RankNet
    on an external dataset using a fixed training set.

    The model is always trained on `train_loader`. The external dataset (`X`, `y`)
    is split into validation/test folds with stratified splitting. For each fold:
    - Select best hyperparameters according to `val_type` ('R2' or 'PER')
    - Retrain model on `train_loader`
    - Evaluate predictions on the validation/test fold

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for the fixed training dataset.
    X : torch.Tensor
        External dataset features (N_samples, N_features).
    y : torch.Tensor
        External dataset targets (N_samples,).
    classes : array-like
        Class labels for stratified splitting of external dataset.
    formulas : array-like or None
        Molecular identifiers for isomer-specific metrics.
    hyperparameters_grid : list of dict
        Hyperparameter configurations to search.
    n_itter : int, optional
        Number of repetitions for CV. Default is 1.
    n_splits : int, optional
        Number of folds per repetition. Default is 5.
    accumulator : list or None, optional
        If provided, stores evaluation metrics for each fold.
    val_type : str, optional
        Validation metric to use ('R2' or 'PER'). Default is 'R2'.

    Returns
    -------
    list
        Validation metrics for each fold (depends on `val_type`).
    """
    final_results = []
    reg_results = []
    int_results = []
    for i in range(n_itter):
        kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=i)
        for test_idx, val_idx in kf.split(X,classes):
            X_val,y_val = X[val_idx],y[val_idx]
            results = []
            for hp in hyperparameters_grid:
                if val_type == 'R2':
                    results.append(val_R2(train_loader,X_val,y_val,hp).cpu().detach().numpy())
                else:
                    results.append(val_PER(train_loader,X_val,y_val,hp))
            best_hp = hyperparameters_grid[np.argmin(results)]
            model = RankNet(20,hidden_dim=best_hp['hidden_dim'],n_hidden=best_hp['n_hidden']).to(DEVICE)
            train_RankNet(model,train_loader=train_loader,num_epochs=best_hp['num_epochs'],optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=best_hp['weight_decay']))
            
            preds = model(X).cpu().detach().numpy().reshape(-1)
            y_np = y.cpu().detach().numpy().reshape(-1)
            if accumulator is not None:
                accumulator.update(y_np,preds,test_idx,formulas)
            metrics = PMetrics(y_np,preds,test_idx,formulas)
            regr_metrics = compute_safe_linear_regression_metrics(preds[test_idx],y_np[test_idx])
            int_metrics = compute_safe_interpolation_metrics(preds[test_idx],y_np[test_idx])
            
            int_results.append([int_metrics['MAPE'],int_metrics['MedAPE']])
            reg_results.append([regr_metrics['MAPE'],regr_metrics['MedAPE']])
            final_results.append([metrics['PER_test'],metrics['PER_mixed'],metrics['iPER_test'],metrics['iPER_mixed'],metrics['ni_test'],metrics['ni_mixed']])
    return final_results,reg_results,int_results
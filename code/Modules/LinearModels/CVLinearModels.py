from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression,Ridge
from scipy.stats import kendalltau 
import numpy as np
from Modules.LinearModels.LMDataPipeline import linear_pairs
from itertools import combinations
from Modules.HelperFunctions import compute_safe_linear_regression_metrics,compute_safe_interpolation_metrics
def PairwiseMetrics(y:np.array,pred:np.array,test_idx:np.array,formulas:np.array,):
    """
    Computes pairwise error rates evaluating the ordering of predicted vs true values
    across all unique pairs of samples, stratified by train/test splits and optionally
    restricted to isomeric pairs.

    Parameters
    ----------
    y : np.array
        True target values (e.g., retention times).
    pred : np.array
        Predicted target values.
    test_idx : np.array
        Indices of test set samples.
    formulas : np.array, optional
        Molecular formula identifiers for each sample, used to restrict analysis to
        isomeric pairs. Default is None.

    Returns
    -------
    dict
        Dictionary containing pairwise error rates and counts for the following subsets:
        - "PER_test", "PER_train", "PER_mixed": error rates on test, train, and mixed pairs.
        - "iPER_test", "iPER_train", "iPER_mixed": error rates restricted to isomeric pairs.
        - "ni_test", "ni_train", "ni_mixed": counts of isomeric pairs in each subset.
    """
    results = dict()
    pairs = np.array(list(combinations(range(len(y)),2)))
    i = pairs[:,0]
    j = pairs[:,1]
    
    valid_mask = y[i] != y[j]
    if formulas is not None:
        iso_mask = formulas[i] == formulas[j]

    test_mask = np.isin(i, test_idx) & np.isin(j, test_idx)
    train_mask = ~np.isin(i, test_idx) & ~np.isin(j, test_idx)
    mixed_mask = ~(train_mask)  # any other combo

    
    errors= (y[i] - y[j]) * (pred[i] - pred[j]) <= 0 
    
    results["PER_test"] = errors[test_mask & valid_mask].sum() / (test_mask & valid_mask).sum() if (test_mask & valid_mask).sum() !=0 else None
    results["PER_train"] = errors[train_mask & valid_mask].sum() / (train_mask & valid_mask).sum() if (train_mask & valid_mask).sum() !=0 else None
    results["PER_mixed"] = errors[mixed_mask & valid_mask].sum() / (mixed_mask & valid_mask).sum() if (mixed_mask & valid_mask).sum() !=0 else None
    
    if formulas is not None:
        results["iPER_test"] = errors[test_mask & iso_mask & valid_mask].sum() / (test_mask & iso_mask & valid_mask).sum() if (test_mask & iso_mask & valid_mask).sum() != 0 else None
        results["iPER_train"] = errors[train_mask & iso_mask & valid_mask].sum() / (train_mask & iso_mask & valid_mask).sum() if (train_mask & iso_mask & valid_mask).sum() != 0 else None 
        results["iPER_mixed"] = errors[mixed_mask & iso_mask & valid_mask].sum() / (mixed_mask & iso_mask & valid_mask).sum() if (mixed_mask & iso_mask & valid_mask).sum() != 0 else None
        
        results['ni_test'] = (test_mask & iso_mask & valid_mask).sum()
        results['ni_train'] = (train_mask & iso_mask & valid_mask).sum()
        results['ni_mixed'] = (mixed_mask & iso_mask & valid_mask).sum()

    return results

def CV_LogReg(X, y, classes, hyperparameters_grid, n_itter=5,n_splits=5):
    """
    Performs cross-validated logistic regression to select the best regularization
    hyperparameter based on pairwise error rate metrics.

    For each hyperparameter value, multiple iterations of stratified K-fold splits
    are performed. A logistic regression model is trained on pairwise transformed data
    from the training folds, then evaluated on the test folds using pairwise error rates.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values (continuous).
    classes : np.ndarray
        Class labels used for stratified splitting.
    hyperparameters_grid : list or np.ndarray
        List of regularization strengths (C values) to evaluate.
    n_itter : int, optional
        Number of iterations with different random seeds for cross-validation (default: 5).
    n_splits : int, optional
        Number of folds for stratified K-fold splitting (default: 5).

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Logistic regression model fitted on the entire dataset using the best hyperparameter.
    """
    results = []
    for i in range(n_itter):
        kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=i)
        for train_idx, test_idx in kf.split(X,classes):
            X_train,y_train = X[train_idx],y[train_idx]
            pairs,labels = linear_pairs(X_train,y_train,np.zeros(len(y_train)))
            for hp in hyperparameters_grid:
                model = LogisticRegression(fit_intercept=False,C=hp)
                model.fit(pairs,labels)
                predictions = (model.coef_@X.T).reshape(-1)
                metrics = PairwiseMetrics(y,predictions,test_idx,formulas=None)
                results.append(metrics['PER_mixed'])
    results = np.array(results).reshape(-1,len(hyperparameters_grid)).mean(axis=0)  
    best_hp = hyperparameters_grid[np.argmin(results)]    
    best_model = LogisticRegression(fit_intercept=False,C=best_hp)
    pairs,labels = linear_pairs(X,y,np.zeros(len(y)))
    best_model.fit(pairs,labels)
    return best_model

def CV_LinReg(X, y, classes, hyperparameters_grid, n_itter=5,n_splits=5):
    """
    Performs cross-validated ridge regression to select the best regularization
    hyperparameter based on pairwise error rate metrics.

    For each hyperparameter value, multiple iterations of stratified K-fold splits
    are performed. A ridge regression model is trained on the training folds,
    then evaluated on the test folds using pairwise error rates computed from predictions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values (continuous).
    classes : np.ndarray
        Class labels used for stratified splitting.
    hyperparameters_grid : list or np.ndarray
        List of regularization strengths (alpha values) to evaluate.
    n_itter : int, optional
        Number of iterations with different random seeds for cross-validation (default: 5).
    n_splits : int, optional
        Number of folds for stratified K-fold splitting (default: 5).

    Returns
    -------
    sklearn.linear_model.Ridge
        Ridge regression model fitted on the entire dataset using the best hyperparameter.
    """
    results = []
    for hp in hyperparameters_grid:
        for i in range(n_itter):
            kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=i)
            for train_idx, test_idx in kf.split(X,classes):
                X_train,y_train = X[train_idx],y[train_idx]
                model = Ridge(alpha=hp)
                model.fit(X_train,y_train)
                predictions = (model.coef_@X.T).reshape(-1)
                metrics = PairwiseMetrics(y,predictions,test_idx,formulas=None)
                results.append(metrics['PER_mixed'])
    results = np.array(results).reshape(-1,n_itter*n_splits).mean(axis=1)  
    best_hp = hyperparameters_grid[np.argmin(results)]    
    best_model = Ridge(alpha=best_hp)
    best_model.fit(X,y)
    return best_model

def NestedCV(X, y, classes,formulas, hyperparameters_grid, inner_cv_func, n_outter_splits=5, n_inner_splits=5,n_itter_out=5,n_itter_inner=5,accumulator = None):
    """
    Perform nested cross-validation with optional accumulation of metrics.

    This function evaluates model performance with hyperparameter tuning via a nested
    cross-validation scheme. The outer loop generates stratified splits for training
    and testing. For each outer training set, the `inner_cv_func` is used to select
    optimal hyperparameters and train a model. The trained model is then evaluated
    on the outer test set using pairwise error metrics. Optionally, an accumulator
    object can be updated with predictions for additional tracking or aggregation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values (numeric or categorical).
    classes : np.ndarray
        Class labels used for stratified splitting.
    formulas : np.ndarray or None
        Molecular formulas or other groupings used in pairwise metrics (e.g., for isoform filtering).
    hyperparameters_grid : list or np.ndarray
        Candidate hyperparameter values for the inner cross-validation search.
    inner_cv_func : callable
        Inner cross-validation function that returns a trained model. Signature:
        (X_train, y_train, classes_train, hyperparameters_grid, n_itter_inner, n_inner_splits) -> model
    n_outter_splits : int, optional
        Number of folds in the outer stratified K-fold (default is 5).
    n_inner_splits : int, optional
        Number of folds in the inner stratified K-fold (default is 5).
    n_itter_out : int, optional
        Number of repeated iterations of outer cross-validation (default is 5).
    n_itter_inner : int, optional
        Number of repeated iterations of inner cross-validation (default is 5).
    accumulator : object, optional
        Optional object with an `update(y, predictions, test_idx, formulas)` method
        for accumulating metrics across folds. If None, accumulation is skipped.

    Returns
    -------
    list of list of float
        A list of results for each outer fold iteration. Each element contains:
        [PER_test, PER_mixed, iPER_test, iPER_mixed, ni_test, ni_mixed]
        - PER_*: pairwise error rates on the respective splits
        - iPER_*: isoform-restricted pairwise error rates
        - ni_*: counts of isoform pairs evaluated

    Notes
    -----
    - Uses PairwiseMetrics to compute error rates between pairs of samples.
    - The inner CV function must accept the specified signature and return a trained model.
    - If `accumulator` is provided, it is updated with predictions for each outer fold.
    """
    results = []
    for i in range(n_itter_out):
        kf = StratifiedKFold(n_outter_splits,shuffle=True,random_state=i)
        for train_idx, test_idx in kf.split(X,classes):
            X_train,y_train = X[train_idx],y[train_idx]
            classes_train = np.array(classes,dtype=int)[train_idx]
            best_model = inner_cv_func(X_train,y_train,classes_train,hyperparameters_grid,n_itter_inner,n_inner_splits)
            predictions = (best_model.coef_ @ X.T).reshape(-1)
            metrics = PairwiseMetrics(y,predictions,test_idx,formulas)
            if accumulator is not None:
                accumulator.update(y,predictions,test_idx,formulas)
            results.append([metrics['PER_test'],metrics['PER_mixed'],metrics['iPER_test'],metrics['iPER_mixed'],metrics['ni_test'],metrics['ni_mixed']])
    return results 

def val_R2(pairs,labels,X_val,y_val,hp):
    """
    Trains a logistic regression model on pairwise data and evaluates the validation R² error.

    Parameters
    ----------
    pairs : np.ndarray
        Feature pairs used for training the logistic regression model.
    labels : np.ndarray
        Target labels corresponding to the pairs.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target values.
    hp : float
        Hyperparameter value for the inverse regularization strength (C) in logistic regression.

    Returns
    -------
    float
        Validation error defined as 1 minus the squared Pearson correlation coefficient (R²)
        between the predicted and actual target values on the validation set.

    Notes
    -----
    - The logistic regression model is trained without an intercept term.
    - The prediction is computed as the dot product of the model coefficients and validation features.
    """
    model = LogisticRegression(C=hp,fit_intercept=False)
    model.fit(pairs,labels)
    preds = np.array(model.coef_ @ X_val.T).reshape(-1)
    CorrCoef = (np.corrcoef(y_val,preds)[0,1] ** 2)
    return 1-CorrCoef

def val_PER(pairs,labels,X_val,y_val,hp):
    """
    Trains a logistic regression model on pairwise data and evaluates the validation Pairwise Error Rate (PER).

    Parameters
    ----------
    pairs : np.ndarray
        Feature pairs used for training the logistic regression model.
    labels : np.ndarray
        Target labels corresponding to the pairs.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target values.
    hp : float
        Hyperparameter value for the inverse regularization strength (C) in logistic regression.

    Returns
    -------
    float
        Validation error defined as (1 - Kendall's tau) / 2, which quantifies the pairwise ordering error
        between predicted and true target values on the validation set.

    Notes
    -----
    - The logistic regression model is trained without an intercept term.
    - Kendall’s tau is a rank correlation coefficient measuring concordance between predicted and true values.
    - Lower returned values indicate better predictive ordering.
    """

    model = LogisticRegression(C=hp,fit_intercept=False)
    model.fit(pairs,labels)
    preds = np.array(model.coef_ @ X_val.T).reshape(-1)
    kendtau = kendalltau(preds,y_val).statistic
    return (1-kendtau)/2

def ILD_cv(pairs,labels,X,y,classes,formulas,hyperparameters_grid,n_itter=5,n_splits=5,accumulator=None,val_type='R2'):
    """
    Performs iterative stratified cross-validation with nested validation and testing on a full dataset.

    The function uses a provided pairwise feature set (`pairs`) and corresponding labels (`labels`) to train logistic regression models.
    The full dataset features (`X`) and targets (`y`) are split into validation and test subsets within each fold:
    - Part of `X` and `y` is used for hyperparameter validation during training.
    - Another part of `X` and `y` is held out for final test metric evaluation.
    
    Hyperparameter tuning is done on the validation subset using either an R²-based or PER-based metric.
    After selecting the best hyperparameter, the model is evaluated on the test subset.
    The function computes pairwise error rates, linear regression, and interpolation metrics on the test data.
    An optional accumulator can be updated with predictions and true values for further analysis or logging.

    Parameters
    ----------
    pairs : np.ndarray
        Pairwise features for model training.
    labels : np.ndarray
        Pairwise labels for model training.
    X : np.ndarray
        Full feature matrix for all samples.
    y : np.ndarray
        Target values for all samples.
    classes : array-like
        Class labels for stratified splitting.
    formulas : np.ndarray or None
        Optional molecular formula labels for isoform filtering in metrics.
    hyperparameters_grid : list or np.ndarray
        List of logistic regression hyperparameters (C values) to evaluate.
    n_itter : int, optional
        Number of cross-validation iterations with different random seeds. Default is 5.
    n_splits : int, optional
        Number of folds per iteration for stratified K-fold splitting. Default is 5.
    accumulator : object or None, optional
        Optional accumulator with an `update` method to collect predictions and targets across folds.
    val_type : str, optional
        Metric type to optimize during hyperparameter selection: 'R2' or 'PER'. Default is 'R2'.

    Returns
    -------
    tuple of lists
        - final_results: List of lists containing pairwise error rate metrics on test subsets.
        - reg_results: List of lists containing regression error metrics on test subsets.
        - int_results: List of lists containing interpolation error metrics on test subsets.

    Notes
    -----
    - Logistic regression models are trained without intercept terms.
    - Validation is performed within training folds; test metrics are computed on the held-out test folds.
    - The accumulator, if provided, is updated after each test fold evaluation.
    """

    final_results = []
    reg_results = []
    int_results = []
    for i in range(n_itter):
        kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=i)
        for test_idx, val_idx in kf.split(X,classes):
            results = []
            for hp in hyperparameters_grid:
                if val_type == 'R2':
                    results.append(val_R2(pairs,labels,X[val_idx],y[val_idx],hp))
                else:
                    results.append(val_PER(pairs,labels,X[val_idx],y[val_idx],hp))
            best_hp = hyperparameters_grid[np.argmin(results)]
            best_model = LogisticRegression(fit_intercept=False, C=best_hp)
            best_model.fit(pairs,labels)
            preds = np.array(best_model.coef_ @ X.T).reshape(-1)
            if accumulator is not None:
                accumulator.update(y,preds,test_idx,formulas)
            metrics = PairwiseMetrics(y,preds,test_idx,formulas)
            
            regr_metrics = compute_safe_linear_regression_metrics(preds[test_idx],y[test_idx])
            
            int_metrics = compute_safe_interpolation_metrics(preds[test_idx],y[test_idx])
            
            int_results.append([int_metrics['MAPE'],int_metrics['MedAPE']])
            reg_results.append([regr_metrics['MAPE'],regr_metrics['MedAPE']])
            
            final_results.append([metrics['PER_test'],metrics['PER_mixed'],metrics['iPER_test'],metrics['iPER_mixed'],metrics['ni_test'],metrics['ni_mixed']])
            
            
    return final_results,reg_results,int_results

    
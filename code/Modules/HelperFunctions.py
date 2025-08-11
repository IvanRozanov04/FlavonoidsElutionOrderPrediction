from itertools import product
import numpy as np 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
from itertools import combinations
def create_hp_grid(param_values):
    """
    Generate a hyperparameter grid as a list of dictionaries representing all possible combinations.

    This function is useful for grid search or exhaustive evaluation of machine learning model parameters.

    Parameters
    ----------
    param_values : dict
        Dictionary where keys are hyperparameter names and values are lists of possible values for each parameter.

    Returns
    -------
    list of dict
        List of dictionaries, each representing a unique combination of hyperparameters.

    Example
    -------
    >>> create_hp_grid({'lr': [0.01, 0.1], 'batch_size': [32, 64]})
    [{'lr': 0.01, 'batch_size': 32},
     {'lr': 0.01, 'batch_size': 64},
     {'lr': 0.1, 'batch_size': 32},
     {'lr': 0.1, 'batch_size': 64}]
    """
    keys = param_values.keys()
    values_product = product(*param_values.values())
    
    hp_grid = [dict(zip(keys, v)) for v in values_product]
    return hp_grid

def weighted_stats(values, weights):
    """
    Computes the weighted mean and standard deviation of a set of values.

    Uses `statsmodels.stats.weightstats.DescrStatsW` to account for unequal weights
    and applies Bessel's correction (ddof=1) for unbiased variance estimation.

    Parameters
    ----------
    values : array-like
        The data values for which the statistics are computed.
    weights : array-like
        The weights associated with each data point.

    Returns
    -------
    float
        Weighted mean of the values.
    float
        Weighted standard deviation of the values (with Bessel correction).

    Example
    -------
    >>> weighted_stats([1, 2, 3], [1, 1, 2])
    (2.25, 0.9574271077563381)
    """
    dsw = DescrStatsW(values, weights=weights, ddof=1)  # ddof=1 for Bessel correction
    return dsw.mean, dsw.std

def format_result(mean, std, sig_figs=2):
    """
    Formats the mean and standard deviation into a string with aligned significant figures.

    Rounds the standard deviation (`std`) to the specified number of significant figures
    and aligns the mean's decimal precision to match. If `std` is NaN, returns "± N/A".

    Parameters
    ----------
    mean : float
        The mean value to report.
    std : float
        The standard deviation associated with the mean.
    sig_figs : int, optional
        Number of significant figures to use for rounding `std`. Default is 2.

    Returns
    -------
    str
        A formatted string of the form "mean ± std", with both rounded appropriately.

    Example
    -------
    >>> format_result(1.23456, 0.04567)
    '1.23 ± 0.046'

    >>> format_result(1234.56, float('nan'))
    '1234.56 ± N/A'
    """
    # Round std to sig_figs and match mean decimal
    if np.isnan(std):
        return f"{round(mean, 2)} ± N/A"
    ci_rounded = round(std, -int(np.floor(np.log10(std))) + (sig_figs - 1))
    mean_rounded = round(mean, -int(np.floor(np.log10(ci_rounded))) + (sig_figs - 1))
    return f"{mean_rounded} ± {ci_rounded}"

def compute_safe_linear_regression_metrics(scores, rts, n_splits=5, random_state=42):
    """
    Performs cross-validated linear regression while avoiding extrapolation,
    and computes regression performance metrics.

    During each fold of cross-validation, test samples whose `scores` fall outside
    the training score range are excluded to ensure the model is not extrapolating.
    Metrics are computed only on the valid (non-extrapolated) predictions.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        The 1D predictor values (e.g., molecular fingerprints or descriptors).
    rts : array-like of shape (n_samples,)
        The target values (e.g., retention times).
    n_splits : int, optional
        Number of folds for K-fold cross-validation. Default is 5.
    random_state : int, optional
        Random seed for shuffling data in cross-validation. Default is 42.

    Returns
    -------
    dict
        A dictionary containing:
        - 'MAPE' : float
            Mean Absolute Percentage Error (%) on valid test points.
        - 'MedAPE' : float
            Median Absolute Percentage Error (%) on valid test points.
        - 'Train_R2' : float
            Average R² score on training data across all folds.
        - 'Coverage' : float
            Fraction of test points retained (i.e., not excluded due to extrapolation).

    Notes
    -----
    - Excludes test points whose score falls outside the training score range
      during each fold to prevent unreliable extrapolation.
    - MAPE and MedAPE exclude true values equal to zero to avoid division errors.

    Example
    -------
    >>> compute_safe_linear_regression_metrics(scores=[1,2,3,4], rts=[10,20,30,40])
    {'MAPE': ..., 'MedAPE': ..., 'Train_R2': ..., 'Coverage': ...}
    """
    scores = np.asarray(scores).reshape(-1, 1)
    rts = np.asarray(rts)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    r2_train_all = []
    total_test_points = 0
    used_test_points = 0

    for train_idx, test_idx in kf.split(scores):
        X_train, X_test = scores[train_idx], scores[test_idx]
        y_train, y_test = rts[train_idx], rts[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test points only within the training score range
        x_min, x_max = X_train.min(), X_train.max()
        valid_mask = (X_test[:, 0] >= x_min) & (X_test[:, 0] <= x_max)
        X_test_valid = X_test
        y_test_valid = y_test[valid_mask]

        y_pred_test = model.predict(X_test_valid)[valid_mask]
        y_pred_train = model.predict(X_train)

        y_true_all.extend(y_test_valid)
        y_pred_all.extend(y_pred_test)
        r2_train_all.append(r2_score(y_train, y_pred_train))

        total_test_points += len(y_test)
        used_test_points += len(y_test_valid)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Avoid division by zero in MAPE
    nonzero_mask = y_true_all != 0
    mape = np.mean(np.abs((y_pred_all[nonzero_mask] - y_true_all[nonzero_mask]) / y_true_all[nonzero_mask])) * 100
    medape = np.median(np.abs((y_pred_all[nonzero_mask] - y_true_all[nonzero_mask]) / y_true_all[nonzero_mask])) * 100

    return {
        'MAPE': mape,
        'MedAPE': medape,
        'Train_R2': np.mean(r2_train_all),
        'Coverage': used_test_points / total_test_points
    }



def compute_safe_interpolation_metrics(scores, rts, n_splits=5, kind='linear', random_state=42):
    """
    Compute regression performance metrics using interpolation within training data range,
    excluding extrapolated test points, via cross-validation.

    For each fold, a 1D interpolator (linear, quadratic, or cubic) is fitted to the training data.
    Predictions on test points falling outside the training score range are excluded to avoid extrapolation.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Predictor values.
    rts : array-like of shape (n_samples,)
        Target values.
    n_splits : int, optional
        Number of folds for K-fold cross-validation. Default is 5.
    kind : {'linear', 'quadratic', 'cubic'}, optional
        Type of interpolation to use. Default is 'linear'.
    random_state : int, optional
        Random seed for shuffling the data before splitting. Default is 42.

    Returns
    -------
    dict
        Dictionary containing:
        - 'MAPE' : float
            Mean Absolute Percentage Error (%) on non-extrapolated test points.
        - 'MedAPE' : float
            Median Absolute Percentage Error (%) on non-extrapolated test points.
        - 'Train_R2' : float
            Average R² score of the interpolation on training data across folds.
        - 'Coverage' : float
            Fraction of test points retained (not excluded due to extrapolation).

    Notes
    -----
    - Interpolation excludes test points with predictor values outside the training fold range.
    - MAPE and MedAPE are computed excluding true values equal to zero to avoid division errors.

    Example
    -------
    >>> compute_safe_interpolation_metrics(scores=[1,2,3,4,5], rts=[10,20,30,40,50], kind='cubic')
    {'MAPE': ..., 'MedAPE': ..., 'Train_R2': ..., 'Coverage': ...}
    """
    scores = np.asarray(scores)
    rts = np.asarray(rts)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    r2_train_all = []
    total_test_points = 0
    used_test_points = 0

    for train_idx, test_idx in kf.split(scores):
        x_train, x_test = scores[train_idx], scores[test_idx]
        y_train, y_test = rts[train_idx], rts[test_idx]

        # Sort for interpolation
        sort_idx = np.argsort(x_train)
        x_train_sorted = x_train[sort_idx]
        y_train_sorted = y_train[sort_idx]

        f_interp = interp1d(x_train_sorted, y_train_sorted, kind=kind, bounds_error=False, fill_value=np.nan)

        # Predict test only where inside the train range
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        valid_mask = (x_test >= x_train_sorted[0]) & (x_test <= x_train_sorted[-1])
        y_pred_test = f_interp(x_test[valid_mask])
        y_pred_train = f_interp(x_train)

        y_true_all.extend(y_test[valid_mask])
        y_pred_all.extend(y_pred_test)
        r2_train_all.append(r2_score(y_train, y_pred_train))

        total_test_points += len(y_test)
        used_test_points += len(y_pred_test)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    nonzero_mask = y_true_all != 0
    mape = np.mean(np.abs((y_pred_all[nonzero_mask] - y_true_all[nonzero_mask]) / y_true_all[nonzero_mask])) * 100
    medape = np.median(np.abs((y_pred_all[nonzero_mask] - y_true_all[nonzero_mask]) / y_true_all[nonzero_mask])) * 100

    return {
        'MAPE': mape,
        'MedAPE': medape,
        'Train_R2': np.mean(r2_train_all),
        'Coverage': used_test_points / total_test_points
    }

class RelDiffErrorAccumulator:
    """
    Accumulates relative differences and error flags for molecular pairs across CV folds.

    This class processes pairs of molecules defined by their indices, computing
    the relative difference of retention times (RT) and whether the predicted order
    matches the true order. It supports filtering pairs by their presence in test/mixed sets.

    Attributes
    ----------
    rel_diffs : list of float
        Relative differences for all processed pairs.
    errors : list of bool
        Boolean flags indicating prediction error (True if predicted order disagrees with true order).
    isomeric_flags : list of bool
        Flags indicating whether the molecular formulas of the pair are identical (isomeric).
    pair_type : str
        Specifies which pairs to accumulate: 'all', 'test+mixed', 'test', or 'mixed'.
    test_indices_per_fold : list of arrays
        Stores test indices used in each fold.

    Parameters
    ----------
    pair_type : str, optional
        Type of pairs to accumulate:
        - 'all': all pairs,
        - 'test+mixed': pairs with at least one molecule in test set,
        - 'test': pairs where both molecules are in test set,
        - 'mixed': pairs where exactly one molecule is in test set,
        Default is 'test+mixed'.
    """
    def __init__(self, pair_type='test+mixed'):
        self.rel_diffs = []
        self.errors = []
        self.isomeric_flags = []
        self.pair_type = pair_type
        self.test_indices_per_fold = []

    def update(self, y_true, y_pred, test_idx, formulas):
        """
        Update the accumulator with relative differences and error flags from one fold.

        For each pair of molecules, calculates the relative difference of RT values and
        whether the predicted order disagrees with the true order, then stores these
        values if the pair meets the specified filtering criteria.

        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            True retention time values for molecules.
        y_pred : array-like, shape (n_samples,)
            Predicted retention time values for molecules.
        test_idx : array-like of int
            Indices of molecules used as test set in the current fold.
        formulas : list of str
            Molecular formula strings for the molecules, aligned with y_true and y_pred.
        """
        test_idx_set = set(test_idx)
        self.test_indices_per_fold.append(test_idx)
        pair_indices = np.array(list(combinations(range(len(y_true)), 2)))

        for i1, i2 in pair_indices:
            in_test1 = i1 in test_idx_set
            in_test2 = i2 in test_idx_set

            # Apply pair_type filtering
            if self.pair_type == 'test' and not (in_test1 and in_test2):
                continue
            elif self.pair_type == 'mixed' and not (in_test1 != in_test2):
                continue
            elif self.pair_type == 'test+mixed' and not (in_test1 or in_test2):
                continue
            elif self.pair_type == 'all':
                pass

            rt1, rt2 = y_true[i1], y_true[i2]
            pred1, pred2 = y_pred[i1], y_pred[i2]

            if min(abs(rt1), abs(rt2)) < 1e-6:  # Avoid division by zero
                continue

            rel_diff = abs(rt1 - rt2) / min(abs(rt1), abs(rt2))
            error = np.sign(pred1 - pred2) != np.sign(rt1 - rt2)

            is_isomeric = formulas[i1] == formulas[i2]

            self.rel_diffs.append(rel_diff)
            self.errors.append(error)
            self.isomeric_flags.append(is_isomeric)

    def get_data(self):
        """
        Retrieve accumulated relative differences, errors, and isomeric flags.

        Returns
        -------
        rel_diffs : np.ndarray, shape (n_pairs,)
            Array of relative differences for accumulated pairs.
        errors : np.ndarray of bool, shape (n_pairs,)
            Boolean array indicating prediction errors for each pair.
        isomeric_flags : np.ndarray of bool, shape (n_pairs,)
            Boolean array indicating isomeric pairs.
        """
        return (
            np.array(self.rel_diffs),
            np.array(self.errors),
            np.array(self.isomeric_flags)
        )



def plot_rel_diff_error_distribution(accumulator, bin_edges=None, isomeric_only=None, title_suffix=""):
    """
    Plot the distribution of prediction errors as a function of relative difference.

    This function visualizes the number of molecular pairs and prediction errors
    within specified relative difference bins. It can filter pairs by isomeric status
    and overlays an error rate curve on a secondary y-axis.

    Parameters
    ----------
    accumulator : RelDiffErrorAccumulator
        An instance of RelDiffErrorAccumulator containing accumulated data.
    bin_edges : array-like, optional
        Sequence of bin edges for grouping relative differences.
        Default is 21 equally spaced bins from 0 to 2.0 (step size 0.1).
    isomeric_only : bool or None, optional
        Filter pairs by isomeric status:
        - True: plot only isomeric pairs,
        - False: plot only non-isomeric pairs,
        - None: plot all pairs (default).
    title_suffix : str, optional
        String appended to the plot title for additional description.

    Returns
    -------
    None
        Displays a matplotlib figure with bar plots for total and error pairs,
        and a line plot for error rate (%).

    Examples
    --------
    >>> acc = RelDiffErrorAccumulator()
    >>> # after multiple updates to acc ...
    >>> plot_rel_diff_error_distribution(acc, isomeric_only=True, title_suffix=" (Fold 1)")
    """
    rel_diff, errors, isomeric_flags = accumulator.get_data()

    # Filter by isomeric status if needed
    if isomeric_only is True:
        mask = isomeric_flags
        label = "Isomeric Pairs"
    elif isomeric_only is False:
        mask = ~isomeric_flags
        label = "Non-Isomeric Pairs"
    else:
        mask = np.ones_like(rel_diff, dtype=bool)
        label = "All Pairs"

    rel_diff = rel_diff[mask]
    errors = errors[mask]

    # Binning
    if bin_edges is None:
        bin_edges = np.linspace(0, 2.0, 21)  # 0.1 step default
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total_counts = np.histogram(rel_diff, bins=bin_edges)[0]
    error_counts = np.histogram(rel_diff[errors], bins=bin_edges)[0]
    error_rate = (error_counts / total_counts) * 100
    error_rate = np.nan_to_num(error_rate)  # Replace NaN with 0 for empty bins

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    width = (bin_edges[1] - bin_edges[0]) * 0.9
    ax1.bar(bin_centers, total_counts, width=width, color='#d7ebf2', label='Total Pairs')
    ax1.bar(bin_centers, error_counts, width=width, color='#dc3041', alpha=1, label='Error Pairs')

    ax1.set_xlabel('Relative Difference (rel_diff)')
    ax1.set_ylabel('Number of Pairs')
    ax1.set_title(f'Error vs Relative Difference [{label}]{title_suffix}')
    ax1.legend(loc='upper left')

    # Right y-axis for error rate
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, error_rate, color='black', linestyle='--', marker='o', linewidth=2, label='Error Rate (%)')
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_ylim(0, 100)

    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

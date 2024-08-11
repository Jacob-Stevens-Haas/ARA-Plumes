import itertools
from typing import cast
from typing import Literal
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from .typing import Float1D
from .typing import Float2D
from .typing import NpFlt


default_rng = np.random.default_rng(1)


def regress_frame_mean(
    arr: Float2D,
    method: str,
    poly_deg: int = 2,
) -> Float1D:
    """
    Regressed mean_points from concentric_circle().

    Parameters:
    ----------
    arr:
        array of mean points from frame. [r(k), x(k), y(k)]

    method:
        Regression methods to apply to arr.
        'linear':    Applies explicit linear regression to (x,y)
        'poly':      Applies explicit polynomial regression to (x,y) with degree
                     up to poly_deg
        'poly_inv':  Applies explcity polynomial regression to (y,x) with degree
                     up to poly_deg
        'poly_para': Applies parametric poly regression to (r,x) and (r,y) with
                     degree up to poly_deg
    poly_deg:
        degree of regression for all poly methods. Note 'linear' ignores this argument.

    Returns:
    -------
    coef:
        tuple containing coefficients, in decesing order in terms of highest degree,
        of regressed function. For 'poly_para' returns concatenated results.
    """

    if method == "linear":
        X = arr[:, 1]
        Y = arr[:, 2]
        coef = do_polynomial_regression(X, Y, poly_deg=1)

    if method == "poly":
        X = arr[:, 1]
        Y = arr[:, 2]
        coef = do_polynomial_regression(X, Y, poly_deg)

    if method == "poly_inv":
        X = arr[:, 2]
        Y = arr[:, 1]
        coef = do_polynomial_regression(X, Y, poly_deg)

    if method == "poly_para":
        X = arr[:, 0]
        Y = arr[:, 1:]
        coef = do_parametric_regression(X, Y, poly_deg)

    return coef


def do_polynomial_regression(X: Float1D, Y: Float1D, poly_deg: int = 2) -> Float1D:
    """Return regressed poly coefficients"""
    if len(X) < poly_deg + 1:
        raise np.linalg.LinAlgError(
            "Number of points insufficients for unique regression with poly_deg ="
            f" {poly_deg}."
        )
    coef = np.polyfit(X, Y, deg=poly_deg)
    return coef


def do_inv_quadratic_regression(
    X: Float1D, Y: Float1D, coef0: Optional[Float1D] = None
) -> Float1D:
    r"""Fit a curve x = a y^2 + by + c minimizing squared error in y

    .. math::

        x = a y^2 + b y + c  \\
        y = \sqrt{(x-c)/a + \frac{b^2}{4a^2}} - \frac{b}{2a}  \\
        y = \sqrt{\tilde a x + \tilde b} - \tilde c  \\

        a = 1/ \tilde a  \\
        b = 2 \frac{\tilde c}{\tilde a}  \\
        c = \frac{\tilde c ^2 - \tilde b}{\tilde a}
    """
    if coef0 is None:
        coef0 = cast(Float1D, default_rng.normal(size=(3,)))
    n_obs = len(Y)

    def discriminant(a_til: float, b_til: float) -> Float1D:
        return a_til * X + b_til

    def y_hat(a_til: float, b_til: float, c_til: float) -> Float1D:
        return np.sqrt(discriminant(a_til, b_til)) - c_til

    def residuals(abc_til: Float1D) -> Float1D:
        a_til, b_til, c_til = abc_til
        return Y - y_hat(a_til, b_til, c_til)

    def jacobian(abc_til: Float1D) -> Float2D:
        a_til, b_til, c_til = abc_til
        r_yhat = -np.eye(n_obs)
        yhat_d = 1 / (2 * np.sqrt(discriminant(a_til, b_til)))
        yhat_d = np.diag(yhat_d)
        d_a = np.reshape(X, (-1, 1))
        d_b = np.ones_like(d_a)
        yhat_a = yhat_d @ d_a
        yhat_b = yhat_d @ d_b
        yhat_c = np.ones((len(Y), 1))
        yhat_coef = np.hstack((yhat_a, yhat_b, yhat_c))
        return r_yhat @ (yhat_coef)

    result = least_squares(residuals, coef0, jacobian)  # type: ignore
    c_tilde = result.x
    coeff = _untildify(c_tilde)
    return coeff


def _untildify(abc_til: Float1D) -> Float1D:
    a_til, b_til, c_til = abc_til
    return np.array([1 / a_til, 2 * c_til / a_til, (c_til**2 - b_til) / a_til])


def do_sinusoid_regression(
    X: Float2D,
    Y: Float1D,
    initial_guess: tuple[float, float, float, float],
) -> Float1D:
    """
    Return regressed sinusoid coefficients (a,w,g,t) to function
    d(t,r) = a*sin(w*r - g*t) + b*r
    """

    def sinusoid_func(X, A, w, gamma, B):
        t, r = X
        return A * np.sin(w * r - gamma * t) + B * r

    coef, _ = curve_fit(sinusoid_func, (X[:, 0], X[:, 1]), Y, initial_guess)
    return coef


def do_parametric_regression(
    X: Float1D,
    Y: Float2D,
    poly_deg: int = 2,
) -> Float1D:
    """Learn parametric poly coefficients"""
    coef = []
    for i in range(len(Y.T)):
        coef = np.append(coef, do_polynomial_regression(X, Y[:, i], poly_deg=poly_deg))
    return coef


####################
# Edge Regressions #
####################


def edge_regression(
    X: Float2D,
    Y: Float1D,
    regression_method: str,
    initial_guess: tuple = (1, 1, 1, 1),
) -> Float1D:
    """
    Takes a nx3 array of (t,x,y) coordinate values and applies some form
    of regression or curve fitting returning a set of optimized regression coefficients.

    Parameters:
    -----------
    X:
        array containing independent variables.

    Y:
        array containing dependent variables

    regression_method: str
        specifies the regression method to be used on array, with the following
        options listed below.
        - `linear`: Performs standard least squares regression with scipy.linalg.stlsq

        - `sinusoid`: Fits a growing sinusoid of the form `y=A*sin(w*x - gamma*t)+Bx`
                          where x is the independent variable. Multiple fitting
                          techniques can be employed which can be specified in the
                          regression_kws dictionary under "method".

    initial_guess:
        The initial guess used when regression_method=='sinusoid'.

    Returns:
    -------
    coef:
        list of learned coefficients.

    """

    if regression_method == "linear":
        coef = do_lstsq_regression(X, Y)

    if regression_method == "sinusoid":
        coef = do_sinusoid_regression(X, Y, initial_guess)

    return coef


def do_lstsq_regression(X: Float2D, Y: Float1D) -> Float1D:
    "Calculate multivariate lienar regression. Bias term is first returned term"
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, _, _, _ = lstsq(X, Y)
    return coef


def var_ensemble_learn(
    X_train: Float2D,
    Y_train: Float2D,
    X_test: Float2D,
    Y_test: Float2D,
    n_samples: int,
    trials: int,
    replace: bool = False,
    kernel_fit: bool = False,
    bandwidth: int = 1,
    plotting: bool = True,
) -> tuple[
    np.ndarray[tuple[Literal[4]], NpFlt] | tuple[float, float, float, float],
    np.ndarray[tuple[int, Literal[4]], NpFlt],
]:
    """
    Apply ensembling to training data via sinusoid regression and provide training
    and test accuracy. Produce histogram of learned params.
    """
    if not kernel_fit:
        param_opt, param_hist = var_ensemble_train(
            X=X_train, Y=Y_train, n_samples=n_samples, trials=trials, replace=replace
        )

        A_opt, w_opt, g_opt, B_opt = param_opt

        def learned_sinusoid_func(t, x):
            return A_opt * np.sin(w_opt * x - g_opt * t) + B_opt * x

        # get train accuracy
        Y_train_learn = learned_sinusoid_func(X_train[:, 0], X_train[:, 1])
        err = np.linalg.norm(Y_train_learn - Y_train) / np.linalg.norm(Y_train)
        train_acc = 1 - err

        print("train accuracy:", train_acc)

        # get validation accuracy
        Y_test_learn = learned_sinusoid_func(X_test[:, 0], X_test[:, 1])
        err = np.linalg.norm(Y_test - Y_test_learn) / np.linalg.norm(Y_test)
        test_acc = 1 - err

        print("test accuracy:", test_acc)

        # plot histograms
        if plotting:
            num_cols = param_hist.shape[1]
            fig, axs = plt.subplots(1, num_cols, figsize=(15, 3))

            titles = ["A_opt", "w_opt", "g_opt", "B_opt"]

            for i in range(num_cols):
                axs[i].hist(param_hist[:, i], bins=50, density=True, alpha=0.8)
                axs[i].set_title(titles[i])
                axs[i].set_xlabel("val")
                axs[i].set_ylabel("Frequency")
                axs[i].axvline(param_opt[i], c="red", linestyle="--")

            plt.tight_layout()
            plt.show()

        return param_opt, param_hist

    elif kernel_fit is True:
        # To do:
        # - X_trian into a train and validation set
        #   so we can try all combinations if bimodal behavior appears
        # - check if there is bimodal behaior
        #   - Kernel density approx, find local max
        # - try all possible combinations on a validation set?
        # - Plot red line to indicate selection

        # randomize selection
        indices = np.arange(len(X_train))
        shuffled_indicies = np.random.permutation(indices)
        train_index = int(len(X_train) * 0.9)

        # Split X_train & Y_train into train and validation set
        X_val = X_train[shuffled_indicies[train_index:]]
        Y_val = Y_train[shuffled_indicies[train_index:]]

        X_train = X_train[shuffled_indicies[:train_index]]
        Y_train = Y_train[shuffled_indicies[:train_index]]

        # Apply ensembling
        _, param_hist = var_ensemble_train(
            X=X_train, Y=Y_train, n_samples=n_samples, trials=trials, replace=replace
        )

        # Apply kernel density fit and idetify param candidates
        param_opt_candidates, kde_models = kernel_density_fit(
            param_hist=param_hist, bandwidth=bandwidth
        )

        # Test all candidates on validation data
        val_acc = -np.inf
        param_opt = None

        for AwgB_i in list(itertools.product(*param_opt_candidates)):
            A_opt, w_opt, g_opt, B_opt = AwgB_i

            def learned_sinusoid_func(t, x):
                return A_opt * np.sin(w_opt * x - g_opt * t) + B_opt * x

            Y_val_learn = learned_sinusoid_func(X_val[:, 0], X_val[:, 1])
            err = np.linalg.norm(Y_val_learn - Y_val) / np.linalg.norm(Y_val)
            val_acc_i = 1 - err

            # Update if validation accuracy increases
            if val_acc_i >= val_acc:
                val_acc = val_acc_i
                param_opt = cast(tuple[float, float, float, float], AwgB_i)

        # print accuracies
        A_opt, w_opt, g_opt, B_opt = param_opt

        def learned_sinusoid_func(t, x):
            return A_opt * np.sin(w_opt * x - g_opt * t) + B_opt * x

        Y_train_learn = learned_sinusoid_func(X_train[:, 0], X_train[:, 1])
        err = np.linalg.norm(Y_train_learn - Y_train) / np.linalg.norm(Y_train)
        train_acc = 1 - err

        Y_test_learn = learned_sinusoid_func(X_test[:, 0], X_test[:, 1])
        err = np.linalg.norm(Y_test_learn - Y_test) / np.linalg.norm(Y_test)
        test_acc = 1 - err

        print("Train accuracy:", train_acc)
        print("Validation accuracy:", val_acc)
        print("Test accuracy:", test_acc)

        # Plotting
        if plotting is True:
            # Plot histograms first
            num_cols = param_hist.shape[1]
            fig, axs = plt.subplots(1, num_cols, figsize=(15, 3))

            titles = ["A_opt", "w_opt", "g_opt", "B_opt"]

            for i in range(num_cols):
                axs[i].hist(param_hist[:, i], bins=50, density=True, alpha=0.8)
                axs[i].set_title(titles[i])
                axs[i].set_xlabel("val")
                axs[i].set_ylabel("Frequency")
                # plot candidate options
                # for candidate_j in param_opt_candidates[i]:
                #     axs[i].axvline(candidate_j, c="black", linestyle="--")
                # plot selected options
                axs[i].axvline(param_opt[i], c="red", linestyle="--")
            plt.tight_layout()
            plt.show()
        return param_opt, param_hist


def var_ensemble_train(
    X: Float2D, Y: Float2D, n_samples: int, trials: int, replace: bool = False
) -> tuple[
    np.ndarray[tuple[Literal[4]], NpFlt], np.ndarray[tuple[int, Literal[4]], NpFlt]
]:
    """
    Ensemble sinusoid regression fit to data
    """
    initial_guess = (1, 1, 1, 1)
    AwgB = np.zeros(shape=(trials, 4))
    AwgB = []
    fits_failed_count = 0
    error = None
    for i in tqdm(range(trials)):
        indices = np.random.choice(a=len(X), size=n_samples, replace=replace)

        Xi = X[indices]
        Yi = Y[indices]
        try:
            A_i, w_i, g_i, B_i = edge_regression(
                X=Xi, Y=Yi, regression_method="sinusoid", initial_guess=initial_guess
            )

            AwgB.append([A_i, w_i, g_i, B_i])
            # initial_guess = AwgB[i]
        except Exception as e:
            fits_failed_count += 1
            error = e
            continue

    if fits_failed_count > 0:
        print("fits failed:", fits_failed_count)
        print(error)

    AwgB = np.array(AwgB)
    param_opt = AwgB.mean(axis=0)
    param_history = AwgB

    return param_opt, param_history


def kernel_density_fit(param_hist, bandwidth=1):
    """
    Find local maxima and mean of histograms using kernel density estimate.
    Applies a kernel density estimate to histogram of data sets, then returns x value
    of local maxima for kernel density function and mean values of data that is fitted.

    Also returns the learned kernel models
    """

    # Create kernel density objects
    kde_models = []
    for i in range(param_hist.shape[1]):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(param_hist[:, i].reshape(-1, 1))
        kde_models.append(kde)

    # instantiate list to store optimal param candidates
    param_opt_canidates = []
    for i, kde_model in enumerate(kde_models):
        data = param_hist[:, i]

        # grab mean value from each column
        mean_val = np.mean(data)

        # create linspace
        x_min = min(data)
        x_max = max(data)
        x_spread = max(np.abs(x_max - mean_val), np.abs(x_min - mean_val))
        buffer = 1.1
        x0 = mean_val - x_spread * buffer
        x1 = mean_val + x_spread * buffer
        x = np.linspace(x0, x1, 1000)

        # Create the density esimtae for each array
        log_density = kde_model.score_samples(x[:, None])

        # Evaluate mean_val on kde fit
        log_density_mean_x = kde_model.score_samples(np.array(mean_val).reshape(1, -1))
        kde_mean_x = np.exp(log_density_mean_x)[0]

        # Find other local maxima
        local_maxima_indicies, _ = find_peaks(np.exp(log_density))

        # store opt_params found
        param_opt_i = [mean_val]
        for max_index in local_maxima_indicies:
            max_index_val_x = np.array(x[max_index]).reshape(1, -1)
            log_density_max_x = kde_model.score_samples(max_index_val_x)
            kde_max_x = np.exp(log_density_max_x)[0]

            # Ensure max is larger than mean
            if kde_max_x >= kde_mean_x:
                param_opt_i.append(x[max_index])

        param_opt_canidates.append(param_opt_i)

    return param_opt_canidates, kde_models

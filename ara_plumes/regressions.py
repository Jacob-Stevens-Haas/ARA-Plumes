import itertools
from typing import Any, cast, Literal
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NBitBase
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
FloatND = np.ndarray[Any, NpFlt]

def var_ensemble_learn(
    X_train: Float2D,
    Y_train: Float2D,
    X_test: Float2D,
    Y_test: Float2D,
    n_samples: int,
    trials: int,
    replace: bool=False,
    kernel_fit: bool=False,
    bandwidth: int=1,
    plotting: bool=True,
) -> tuple[
    np.ndarray[tuple[Literal[4]], NpFlt] | tuple[float, float, float, float],
    np.ndarray[tuple[int, Literal[4]], NpFlt]
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
        if plotting is True:
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
    X: Float2D,
    Y: Float2D,
    n_samples: int,
    trials: int,
    replace: bool=False
) -> tuple[
     np.ndarray[tuple[Literal[4]], NpFlt],
     np.ndarray[tuple[int, Literal[4]], NpFlt]]:
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
            A_i, w_i, g_i, B_i = regression(
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


def regression(
    X: np.ndarray,
    Y: np.ndarray,
    regression_method: str,
    initial_guess: tuple = None,
    regression_kws: dict = {},
):
    """
    Takes a nx2 array of (x,y) coordinate values and applies some form
    of regression or curve fitting returning a set of optimized regression coefficients.

    Parameters:
    -----------
    X: np.ndarray
        array containing independent variables

    Y: np.ndarray
        array containing dependeant variables

    regression_method: str
        specifies the regression method to be used on array, with the following
        options listed below.
            - 'poly': Performs polynomial regression using np.polyfit.

            - 'sinusoid': Fits a growing sinusoid of the form A*sin(w*x - gamma*t)+Bx
                          where x is the independent variable. Multiple fitting
                          techniques can be employed which can be specified in the
                          regression_kws dictionary under "method".

    initial_guess: tuple (default None)
        The initial guess used for some regression_method techniques such as 'sinusoid'
        where if the None type is selected the default guess becomes (1,1,1,1).

    regression_kws: dict (default {})
        Additional keyword arguments for regression method selected.

    """

    if regression_method == "poly":
        if "poly_deg" in regression_kws:
            poly_deg = regression_kws["poly_deg"]
        else:
            poly_deg = 2

        regression_coeff = np.polyfit(X, Y, deg=poly_deg)

        return regression_coeff

    if regression_method == "sinusoid":
        if "method" in regression_kws:
            method = regression_kws["method"]
        else:
            method = "scipy_curve_fit"

        if initial_guess is None:
            initial_guess = (1, 1, 1, 1)

        if method == "scipy_curve_fit":

            def sinusoid_func(X, A, w, gamma, B):
                t, x = X
                return A * np.sin(w * x - gamma * t) + B * x

            regression_coeff, pcov = curve_fit(
                sinusoid_func, (X[:, 0], X[:, 1]), Y, initial_guess
            )
        if method == "FFT_linear_fit":
            print("Fill out method")

        return regression_coeff


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


def flatten_vari_dist(vari_dist):
    """
    Convert vari_dist list [(t0,[[x0,y0],...[xn,yn]]),...]
    to flattned array ->[[t0,x0,y0], [t0,x1,y1],...[t0,xn,yn],[t1,...],...]
    """
    t_x_y = []
    for vari in vari_dist:
        nt = len(vari[1])
        ti = vari[0]
        if nt == 0:
            continue
        t_x_y_i = np.array([ti for _ in range(nt)]).reshape(nt, -1)
        t_x_y_i = np.hstack((t_x_y_i, vari[1]))
        if len(t_x_y) == 0:
            t_x_y = t_x_y_i
        else:
            t_x_y = np.vstack((t_x_y, t_x_y_i))
    return t_x_y


# Old Functions
def plot_sinusoid(X_i, Y_i, t_i, regress=True, initial_guess=(1, 1, 1, 1)):
    fig = plt.figure(figsize=(8, 6))
    if regress is True:
        try:
            A, w, gamma, B = sinusoid_regression(X_i, Y_i, t_i, initial_guess)
            x = np.linspace(0, X_i[-1])

            def sinusoid_func(x):
                return A * np.sin(w * x - gamma * t_i) + B * x

            y = sinusoid_func(x)

            plt.plot(x, y, color="red", label="sinusoid")
            new_guess = (A, w, gamma, B)
        except Exception as e:
            print(f"Sinusoid could not fit. Error: {e}")
            new_guess = initial_guess

    plt.scatter(X_i, Y_i, color="blue", label="var points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sinusoid, t=" + str(t_i))
    plt.legend()
    # plt.grid(True)
    # plt.show(block=False)
    return fig, new_guess


# Might not need this function anymore
def sinusoid_regression(X, Y, t, initial_guess):

    # Define the function
    def sinusoid(x, A, w, gamma, B):
        return A * np.sin(w * x - gamma * t) + B * x

    # initial_guess = (1, 1, 1, 1)
    params, covariance = curve_fit(sinusoid, X, Y, initial_guess)
    A_opt, w_opt, gamma_opt, B_opt = params
    return (A_opt, w_opt, gamma_opt, B_opt)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def var_train(X_train, Y_train, X_test, Y_test, n_samples, trials, replace=False):
    param_opt, param_hist = var_learn(
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

    # get test accuracy
    Y_test_learn = learned_sinusoid_func(X_test[:, 0], X_test[:, 1])
    err = np.linalg.norm(Y_test - Y_test_learn) / np.linalg.norm(Y_test)
    test_acc = 1 - err

    print("test accuracy:", test_acc)

    # plot histograms
    num_cols = param_hist.shape[1]
    fig, axs = plt.subplots(1, num_cols, figsize=(15, 3))

    titles = ["A_opt", "w_opt", "g_opt", "B_opt"]

    for i in range(num_cols):
        axs[i].hist(param_hist[:, i], bins=50)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("val")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return param_opt, param_hist


def var_learn(X, Y, n_samples, trials, replace=False):
    """ """
    initial_guess = (1, 1, 1, 1)
    AwgB = np.zeros(shape=(trials, 4))
    for i in tqdm(range(trials)):
        indices = np.random.choice(a=len(X), size=n_samples, replace=replace)

        Xi = X[indices]
        Yi = Y[indices]

        A_i, w_i, g_i, B_i = regression(
            X=Xi, Y=Yi, regression_method="sinusoid", initial_guess=initial_guess
        )

        AwgB[i] = [A_i, w_i, g_i, B_i]
        # initial_guess = AwgB[i]

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

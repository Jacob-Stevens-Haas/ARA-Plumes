import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def var_ensemble_learn(
    X_train, Y_train, X_test, Y_test, n_samples, trials, replace=False
):
    """
    Apply ensembling to training data via sinusoid regression and provide training
    and test accuracy. Produce histogram of learned params.
    """
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


def var_ensemble_train(X, Y, n_samples, trials, replace=False):
    """
    Ensemble sinusoid regression fit to data
    """
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

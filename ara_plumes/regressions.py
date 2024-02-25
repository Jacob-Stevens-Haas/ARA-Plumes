import numpy as np
from scipy.optimize import curve_fit


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

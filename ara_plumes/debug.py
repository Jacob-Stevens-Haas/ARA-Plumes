from typing import TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ara_plumes.regressions import do_polynomial_regression

CMAP = mpl.color_sequences["tab10"]

true_data = np.array(
    [
        [0, 0],
        [-0.25, 0.5],
        [-1, 1],
        [-4, 2],
    ],
    dtype=float,
)

plot_y = np.arange(0, 2.3, 0.1)
# plt.scatter(true_data[:, 0], true_data[:, 1], color="r", marker="o")

x_trains = []
x_preds = []
y_trains = []
y_preds = []
coeffs_list = []
ScalarOrArray = TypeVar("ScalarOrArray", float, npt.NDArray[np.floating])


def fit_curve(y: ScalarOrArray, a, b, c) -> ScalarOrArray:
    return a * y**2 + b * y + c


def fit_inverse(x: ScalarOrArray, a, b, c) -> ScalarOrArray:
    disc = (x - c) / a + b**2 / (4 * a**2) + 1e-14
    return np.sqrt(disc) - b / (2 * a)


n = 4
for i, delta in enumerate(np.arange(0.0, 0.1 * n, 0.1)):
    X = true_data[:, 0].copy()
    Y = true_data[:, 1].copy()
    Y[2] += delta
    Y[3] -= delta
    x_trains.append(X)
    y_trains.append(Y)
    plt.scatter(X, Y, marker="x", color=CMAP[i])
    x_min_coeff = do_polynomial_regression(Y, X)
    # y_min_coeff = do_inv_quadratic_regression(X, Y)
    a, b, c = x_min_coeff
    coeffs_list.append(x_min_coeff)
    x_preds.append(fit_curve(Y, a, b, c))
    y_preds.append(fit_inverse(X, a, b, c))
    plt.plot(
        fit_curve(plot_y, a, b, c), plot_y, color=CMAP[i], label=f"Model/dataset {i}"
    )

plt.title("Fit inverse polynomials in standard way (minimizing MSE of X)")
plt.legend()

# each dataset should have the lowest x-error on its fit model, since that
# model minimizes the error in x.  But it might not have the lowest y error!
error_func = lambda true, pred: np.sum((true - pred) ** 2)  # noqa
x_errors = np.array(
    [
        [error_func(fit_curve(y_train, *coeffs), X) for coeffs in coeffs_list]
        for y_train in y_trains
    ]
)
y_errors = np.array(
    [[error_func(y_pred, y_data) for y_pred in y_preds] for y_data in y_trains]
)
plt.figure()
plt.imshow(x_errors)
plt.colorbar()
plt.ylabel("dataset")
plt.xlabel("model")
plt.yticks([0.0, 1.0, 2.0, 3.0], ["dataset 1", "dataset 2", "dataset 3", "dataset 4"])
plt.xticks([0.0, 1.0, 2.0, 3.0], ["model 1", "model 2", "model 3", "model 4"])
plt.title("X squared error: Minimum for each row\nshould be on diagonal")
plt.scatter([np.argmin(x_errors[i]) for i in range(4)], range(4), color="r", marker="x")

plt.figure()
plt.imshow(y_errors)
plt.colorbar()
plt.ylabel("dataset")
plt.xlabel("model")
plt.yticks([0.0, 1.0, 2.0, 3.0], ["dataset 1", "dataset 2", "dataset 3", "dataset 4"])
plt.xticks([0.0, 1.0, 2.0, 3.0], ["model 1", "model 2", "model 3", "model 4"])
plt.title(
    "Y squared error: If minimum of each row is not on diagonal,\nfitting on y can do"
    " better"
)
plt.scatter([np.argmin(y_errors[i]) for i in range(4)], range(4), color="r", marker="x")
# plt.figure()
# y_errors_2 = np.array([
#     [error_func(fit_inverse(X, *coeffs), y_data) for coeffs in coeffs_list]
#     for y_data in y_trains
# ])
# plt.imshow(y_errors_2)
# plt.colorbar()
# plt.ylabel("dataset")
# plt.xlabel("model")
# plt.title("Y squared error")

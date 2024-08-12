from typing import cast

import numpy as np
from scipy.optimize import least_squares

from ara_plumes.typing import Float1D
from ara_plumes.typing import Float2D

rng = np.random.default_rng(1)

X = np.array([0, -0.25, -1, -4], dtype=float)
Y = np.array([0, 0.5, 1, 2], dtype=float)
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


def loss(abc_til: Float1D) -> float:
    return cast(float, 1 / 2 * np.sum(residuals(abc_til) ** 2))


def grad(abc_til: Float1D) -> Float1D:
    return residuals(abc_til) @ jacobian(abc_til)


def hess(abc_til: Float1D) -> Float2D:
    a_til, b_til, c_til = abc_til
    J = jacobian(abc_til)
    d = discriminant(a_til, b_til)
    yhat_dd_vec = -1 / 4 * np.sqrt(d**3)
    diag_inds = np.diag_indices(len(X), 3)
    yhat_dd = np.zeros((len(X), len(X), len(X)))
    yhat_dd[diag_inds] = yhat_dd_vec
    d_a = np.reshape(X, (-1, 1))
    d_b = np.ones_like(d_a)
    d_c = np.zeros_like(d_a)
    d_coef = np.hstack((d_a, d_b, d_c))
    yhat_2coef = np.einsum("ikl,lm", np.einsum("ij,jkl", d_coef.T, yhat_dd), d_coef)
    return J.T @ J + np.einsum("j,ijk", residuals(abc_til), yhat_2coef)


def non_nan_residuals(abc_til):
    res = residuals(abc_til)
    if not all(np.isfinite(res)):
        return 100 + 100 * np.ones_like(res) * np.linalg.norm(abc_til)
    else:
        return res


def tildify(abc: Float1D) -> Float1D:
    a, b, c = abc
    return np.array([1 / a, b**2 / (4 * a**2) - c / a, b / (2 * a)])


def untildify(abc_til: Float1D) -> Float1D:
    a_til, b_til, c_til = abc_til
    return np.array([1 / a_til, 2 * c_til / a_til, (c_til**2 - b_til) / a_til])


def grad_test(f, J, x, dx=None):
    if dx is None:
        dx = 1e-4 * x * np.random.random(size=x.shape)
    lhs = np.reshape(f(x) - f(x + dx), (-1, 1))
    rhs = J(x) @ dx
    return lhs - rhs


def complex_step_test(f, J, x, dx=None, step=None):
    if step is None:
        step = 1e-3
    if dx is None:
        dx = 1j * step * x * np.random.random(size=x.shape)
    lhs = np.reshape((f(x) - f(x + dx)).imag, (-1, 1))
    rhs = J(x) * step
    return lhs - rhs


coef0 = np.array([-1.0, 1e-5, 1e-5])
result = least_squares(residuals, coef0, jacobian)  # type: ignore
c_tilde = result.x
coeff = untildify(c_tilde)


def test_case(n_obs: int) -> tuple[Float1D, Float1D, Float1D]:
    coef = rng.normal.random(size=(3,))
    a, b, c = coef
    y_inputs = 3 * rng.random(size=(n_obs,))
    x_outputs = a * y_inputs**2 + b * y_inputs + c
    coef_tilde = tildify(coef)
    return y_inputs, x_outputs, coef_tilde

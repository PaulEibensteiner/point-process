import mosek
from stpy.kernels import KernelFunction
import torch
import logging
from scipy.spatial import Voronoi
from voronoi import in_box, voronoi
import numpy as np
from autograd_minimize import minimize
import cvxpy as cp

logger = logging.getLogger(__name__)


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_weights(points, a):
    bounds = [
        a.bounds[0][0].item(),
        a.bounds[0][1].item(),
        a.bounds[1][0].item(),
        a.bounds[1][1].item(),
    ]
    # bounds: [x_min, x_max, y_min, y_max]
    vor = voronoi(points, bounds)
    areas = np.zeros(len(points))
    for point_i in range(len(points)):
        region_i = vor.point_region[point_i]
        region = vor.regions[region_i]
        assert not -1 in region
        vertices = np.array([vor.vertices[j] for j in region])
        assert np.any(in_box(vertices, bounds))
        x, y = vertices[:, 0], vertices[:, 1]
        area = poly_area(x, y)
        areas[point_i] = area
    return areas


class NaivePPPEstimator:
    def __init__(self, kernel_object: KernelFunction):
        self.kernel_object = kernel_object
        self.kernel = kernel_object.kernel

    def load_data(self, A, x: torch.Tensor):
        self.observations = x.numpy()
        self.A = A

    def approximate_kernel_matrix(self, x, num_features):
        return self.kernel(x, x)

    def fit(self, roi: torch.Tensor, noise_variance=1e-5):
        logger.info("Calculating Cholesky Decomposition")
        roi = roi.cpu().numpy()
        x = np.vstack((roi, self.observations))
        K_approx = self.kernel(torch.tensor(x), torch.tensor(x)).cpu().numpy()
        K = K_approx + noise_variance * np.eye(x.shape[0])
        K_inv = np.linalg.pinv(K)
        weights = get_weights(x, self.A)

        """
        y = cp.Variable(len(x))
        newobjective = cp.Minimize(
            -cp.sum(cp.log(y[: len(self.observations)]))
            + cp.sum(cp.multiply(weights, y))
            + 0.5 * cp.quad_form(y, K_inv)
        )
        constraints = []
        constraints.append(y >= np.zeros(len(x)))
        prob = cp.Problem(newobjective, constraints)
        prob.solve(
            solver=cp.MOSEK,
            warm_start=False,
            verbose=False,
            mosek_params={
                mosek.iparam.num_threads: 4,
                mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
                mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
                mosek.dparam.intpnt_co_tol_rel_gap: 1e-4,
            },
        )
        return y.value
        """

        weights = torch.tensor(weights, requires_grad=True)
        K_inv = torch.tensor(K_inv, requires_grad=True)

        def objective(y: torch.Tensor):
            y_o = y[len(roi) :]
            sum_lkl = torch.sum(torch.log(y_o))
            int_lkl = torch.sum(weights * y)
            log_prior = 0.5 * y.T @ K_inv @ y
            return -sum_lkl + int_lkl + log_prior

        y_0 = np.zeros(len(x), dtype=np.float64)
        res = minimize(
            objective,
            y_0,
            backend="torch",
            method="L-BFGS-B",
            bounds=(0.0, 1e7),
            precision="float64",
            tol=1e-8,
            torch_device=str(torch.get_default_device()),
            options={
                "ftol": 1e-08,
                "gtol": 1e-08,
                "eps": 1e-08,
                "maxfun": 15000,
                "maxiter": 15000,
                "maxls": 20,
            },
        )
        print(f"optimum found")
        return res.x[: len(roi)]

from functools import partial
from typing import List
import numpy as np
import scipy
from stpy.borel_set import BorelSet
from stpy.kernels import KernelFunction
from tqdm import tqdm
from autograd_minimize import minimize
import torch

device = torch.get_default_device()


def sqrt(matrix: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(
        np.real(scipy.linalg.sqrtm(matrix.cpu().numpy() + 1e-5))
    ).to(device)


class LogCoxProcess:
    def __init__(self, kernel_object: KernelFunction, integral_discretization: int):
        self.kernel_object = kernel_object
        self.kernel = kernel_object.kernel
        self.integral_discretization = integral_discretization

    def load_data(self, data: List):
        # only works with 2d data!
        observations = []
        self.areas = []
        dts = []
        a_xs = []
        a_ys = []
        b_xs = []
        b_ys = []

        for A, x, dt in data:
            observations.append(x)
            a_xs.append(A.bounds[0][0])
            b_xs.append(A.bounds[0][1])
            a_ys.append(A.bounds[1][0])
            b_ys.append(A.bounds[1][1])
            dts.append(dt)
            self.areas.append((A, dt))

        self.observations = torch.cat(observations, dim=0)
        self.dt = torch.tensor(dts, dtype=torch.float64)
        self.a_x = torch.tensor(a_xs)
        self.a_y = torch.tensor(a_ys)
        self.b_x = torch.tensor(b_xs)
        self.b_y = torch.tensor(b_ys)

    def fit(self):
        # Get the map by representer theorem
        k_func = partial(self.kernel, b=self.observations)
        k_int = self.kernel_object.integral(self.a_x, self.a_y, self.b_x, self.b_y)
        k_obs = torch.cat(
            (
                k_func(a=self.observations),
                self.dt.unsqueeze(1) * k_int(self.observations),
            )
        )

        k_weights = []
        k_nodes = []
        k_factors = []
        for A, dt in self.areas:
            weights, nodes = A.return_legendre_discretization(
                self.integral_discretization
            )
            nodes = nodes.to(device)
            weights = weights.to(device)
            k_n = torch.cat((k_func(a=nodes), self.dt.unsqueeze(1) * k_int(nodes)))
            k_weights.append(weights)
            k_nodes.append(k_n)
            k_factors.append(dt)

        # integral matrix won't be symmetric! Is this a problem? Try out by mirroring
        k_int_int = []
        for A, dt in self.areas:
            weights, nodes = A.return_legendre_discretization(
                self.integral_discretization
            )
            nodes = nodes.to(device)
            weights = weights.to(device)
            integral = dt * torch.sum(
                weights * self.dt.unsqueeze(1) * k_int(nodes), dim=1
            )  # sum over nodes
            k_int_int.append(integral)

        k_int_int = torch.stack(k_int_int)
        k_obs_obs = k_func(a=self.observations)
        k_int_obs = self.dt.unsqueeze(1) * k_int(
            self.observations
        )  # number of observations is columns
        k_obs_int = k_int_obs.T

        # Create one big kernel matrix out of the above four matrices
        k_top = torch.cat((k_obs_obs, k_obs_int), dim=1)
        k_bottom = torch.cat((k_int_obs, k_int_int), dim=1)
        k_big = torch.cat((k_top, k_bottom), dim=0)

        # Check if k_big is above zero everywhere
        assert torch.all(k_big >= 0), "Kernel matrix should be strictly positive"

        # Check if k_big is approximately symmetric
        assert torch.allclose(
            k_big, k_big.T, atol=1e-4
        ), "Kernel matrix should be approximately symmetric"

        def objective(alpha):
            lkl_term_1 = (alpha @ k_obs).sum()  # Should be a single number now
            lkl_term_2 = torch.sum(
                torch.stack(
                    [
                        dt * torch.sum(w * torch.exp(alpha @ kn))
                        for w, kn, dt in zip(k_weights, k_nodes, k_factors)
                    ]
                )
            )

            regularizer = alpha.T @ k_big @ alpha
            return -lkl_term_1 + lkl_term_2 + regularizer  # * 0.5

        alpha_0 = torch.zeros([len(self.observations) + len(self.a_x)])
        res = minimize(
            objective,
            alpha_0.cpu().numpy(),
            backend="torch",
            method="L-BFGS-B",
            precision="float64",
            tol=1e-8,
            torch_device=str(device),
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

        self.alpha_opt = torch.tensor(res.x)

        def intensity(x: torch.tensor, dt):
            k_obs = torch.cat((k_func(x), k_int(x)))
            return dt * torch.exp(torch.tensor(res.x) @ k_obs).unsqueeze(1)

        self.rate_value = intensity

        return intensity

    def get_gamma_MAP(self, n, x, a, lr=0.01, max_it=10000, eps=1e-6):
        mean = 0
        cov_Y = self.kernel(x, x)
        Q = sqrt(cov_Y)
        self.Q = Q

        def f(arg):
            y = arg @ Q + mean
            return (-0.5) * arg.pow(2).sum() + (y * n - torch.exp(y) * a).sum()

        gamma = torch.zeros(len(x), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([gamma], lr=lr)

        # Use tqdm to show progress
        prev_loss = float("inf")
        for _ in tqdm(range(max_it), desc="Optimizing gamma"):
            optimizer.zero_grad()
            loss = -f(gamma)  # we minimize -f because we want to maximize f
            if loss.item() > prev_loss:
                print("Warning: Loss did not decrease")
            prev_loss = loss.item()
            loss.backward()
            # If gradient is smaller than eps, return
            if torch.norm(gamma.grad) < eps:
                print("Solved to eps")
                break
            optimizer.step()

        assert f(gamma) > f(
            torch.distributions.MultivariateNormal(
                loc=gamma, covariance_matrix=torch.eye(len(gamma)) * 50
            ).sample()
        )

        return gamma.detach()

    def sample_mala(self, n, x, a, h, num_steps, burn_in_steps, initial_gamma=None):
        # param n is 1d tensor with the counts of points in the cells
        # param x is the discretization of the area we're interested in
        # param a is either a 2d tensor with the areas of the discretization
        # or a float that gives all areas
        # step size h
        gamma = self.get_MAP() if initial_gamma is None else initial_gamma
        mean = 0  # prior mean I think?
        cov_Y = self.kernel(x, x)
        Q = sqrt(cov_Y)
        self.Q = Q
        accept_prob_sum = 0

        for i in range(num_steps):
            # The log posterior over gamma given the data
            def log_f(arg):
                y = arg @ Q + mean
                return (-0.5) * arg.pow(2).sum() + (y * n - torch.exp(y) * a).sum()

            # Gradient of the energy
            def grad(arg):
                y = arg @ Q + mean
                return -arg + (n - torch.exp(y) * a) @ Q.T

            # mean of the proposal distribution, named \xi in paper
            def r_mean_given_arg(arg):
                return arg + (h / 2.0) * grad(arg)

            # Proposal
            proposal = torch.distributions.MultivariateNormal(
                loc=r_mean_given_arg(gamma),
                covariance_matrix=h * torch.eye(len(gamma), dtype=torch.float64),
            ).sample()

            accept_prob = torch.exp(
                log_f(proposal)
                - (gamma - r_mean_given_arg(proposal)).pow(2).sum() / (2 * h)
            ) / (
                torch.exp(
                    log_f(gamma)
                    - (proposal - r_mean_given_arg(gamma)).pow(2).sum() / (2 * h)
                )
            )

            if np.random.rand() < accept_prob:
                gamma = proposal

            accept_prob_sum += min(accept_prob.item(), 1.0)

            if i > burn_in_steps:
                yield torch.exp(gamma @ Q + mean)

        mean_accept_prob = accept_prob_sum / num_steps
        print(mean_accept_prob)

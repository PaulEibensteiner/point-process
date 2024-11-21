# Define the Process

from functools import partial
import numpy as np
import scipy
from stpy.borel_set import BorelSet
from stpy.kernels import KernelFunction
from tqdm import tqdm
from autograd_minimize import minimize
import torch


def sqrt(matrix: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(
        np.real(scipy.linalg.sqrtm(matrix.cpu().numpy() + 1e-5))
    ).to(device)


class LogCoxProcess:
    def __init__(self, kernel_object: KernelFunction):
        self.kernel_object = kernel_object
        self.kernel = kernel_object.kernel

    def get_intensity_MAP(
        self,
        lower_bounds_x,
        lower_bounds_y,
        upper_bounds_x,
        upper_bounds_y,
        observations,
        b: BorelSet,
        integral_discretization: int,
    ):
        # Get the map by representer theorem
        k_func = partial(self.kernel, b=observations)
        k_int = self.kernel_object.integral(
            lower_bounds_x, lower_bounds_y, upper_bounds_x, upper_bounds_y
        )

        def objective(alpha):
            k_obs = torch.cat((k_func(a=observations), k_int(observations)))
            lkl_term_1 = (alpha @ k_obs).sum()  # Should be a single number now

            weights, nodes = b.return_legendre_discretization(integral_discretization)
            nodes = nodes.to(device)
            weights = weights.to(device)
            k_nodes = torch.cat((k_func(a=nodes), k_int(nodes)))
            lkl_term_2 = torch.sum(weights * torch.exp(alpha @ k_nodes))

            regularizer = (alpha**2).sum()

            return -lkl_term_1 + lkl_term_2 + 0.5 * regularizer

        alpha_0 = torch.zeros([len(observations) + len(lower_bounds_x)])
        res = minimize(
            objective,
            alpha_0.cpu().numpy(),
            backend="torch",
            method="L-BFGS-B",
            precision="float64",
            tol=1e-8,
            torch_device=str(device),
        )
        print(f"optimum found")

        def intensity(x: torch.tensor):
            k_obs = torch.cat((k_func(x), k_int(x)))
            return torch.exp(torch.tensor(res.x) @ k_obs)

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

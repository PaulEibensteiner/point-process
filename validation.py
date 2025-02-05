from typing import Optional, Tuple
import numpy as np
from stpy.borel_set import BorelSet
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from stpy.point_processes.rate_estimator import RateEstimator
import torch

device = torch.get_default_device()

# Assume: Data drawn by 0-mean truncated GP intensity with RBF kernel
from stpy.helpers.posterior_sampling import tmg
import matplotlib.pyplot as plt


def check_approx_squared_integral_difference(
    poisson_process: PoissonPointProcess,
    poisson_process_approx: RateEstimator,
    domain: BorelSet,
    discretization_ppp_sampling: int,
    discretization_integral_diff: int,
    dt=1,
    dataset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the squared integral difference between two functions over a given domain:
    $\int_D (f(x) - \hat f(x))^2 dx$
    The functions must map from the domain to a skalar value and must be able to
    handle batch input i.e. the first dimension of the input is the number of
    points and must be equal to the first dimension of the output.

    Args:
        f (callable[[torch.Tensor], torch.Tensor]): The ground truth PPP.
        f_hat (callable[[torch.Tensor], torch.Tensor]): The approximated function.
        domain (BorelSet): The domain over which to integrate.
        discretization_ppp_sampling (int): The number of discretization points per dimension,
        for the sampling process
        discretization_integral_diff (int): The number of discretization points per dimension,
        for the integral estimation
        dataset: The dataset to use for fitting the estimator. If none is given
        a new dataset will be drawn from the ground truth PPP

    Returns:
        torch.Tensor: The integral of the squared difference between the two functions.
    """

    weights, nodes = domain.return_legendre_discretization(discretization_integral_diff)
    if dataset is None:
        print("Sampling from the Ground Truth Poisson Process")
        dataset = poisson_process.sample_discretized(
            domain, dt, discretization_ppp_sampling
        )

    print(
        f"Loading {len(dataset)} Sampled Data points into approximate intensity model"
    )
    poisson_process_approx.load_data([(domain, dataset, dt)])
    poisson_process_approx.fit()

    print(
        "Approximating the integral of the squared difference between the ground truth"
        " intensity and its approximation"
    )
    weights = weights.to(device)
    nodes = nodes.to(device)
    squared_integral_difference = (
        weights
        * (
            poisson_process.rate(nodes, 1)
            - poisson_process_approx.rate_value(nodes, 1).squeeze(1)
        )
        ** 2
    ).sum()

    return squared_integral_difference, dataset


def plot_approximation(
    poisson_process: PoissonPointProcess,
    poisson_process_approx: RateEstimator,
    domain: BorelSet,
    dataset,
    discretized_domain=None,
):
    gp = poisson_process.rate
    bounds = domain.bounds

    # Plot 'ground truth' truncated gp
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Extract x and y coordinates
    if discretized_domain is None:
        x_coords = gp.x_acc[:, 0].cpu().numpy()
        y_coords = gp.x_acc[:, 1].cpu().numpy()
        function_values = gp.y_acc.cpu().numpy()
    else:
        x_coords = discretized_domain[:, 0].cpu().numpy()
        y_coords = discretized_domain[:, 1].cpu().numpy()
        function_values = poisson_process.rate(discretized_domain, 1).cpu().numpy()

    # Create scatter plot with color gradient
    sc1 = axs[0].scatter(x_coords, y_coords, c=function_values, cmap="viridis")
    fig.colorbar(sc1, ax=axs[0], label="Function Value")
    axs[0].set_xlabel("X Coordinate")
    axs[0].set_ylabel("Y Coordinate")
    axs[0].set_title("Ground Truth Truncated GP")
    axs[0].set_xlim(bounds[0][0].item(), bounds[0][1].item())
    axs[0].set_ylim(bounds[1][0].item(), bounds[1][1].item())

    # Plot artificial Data
    x_artificial = dataset[:, 0].cpu().numpy()
    y_artificial = dataset[:, 1].cpu().numpy()

    # Create scatter plot
    axs[1].scatter(
        x_artificial, y_artificial, color="red", label="Artificial Data Points"
    )
    axs[1].set_xlabel("X Coordinate")
    axs[1].set_ylabel("Y Coordinate")
    axs[1].set_title("Artificial Data Points")
    axs[1].legend()
    axs[1].set_xlim(bounds[0][0].item(), bounds[0][1].item())
    axs[1].set_ylim(bounds[1][0].item(), bounds[1][1].item())

    # Plot approximation, in the same color scale as ground truth
    function_values = (
        poisson_process_approx.rate_value(
            gp.x_acc if discretized_domain is None else discretized_domain, 1
        )
        .cpu()
        .numpy()
    )

    # Create scatter plot with color gradient
    sc2 = axs[2].scatter(
        x_coords,
        y_coords,
        c=function_values,
        cmap="viridis",
        vmin=min(gp.y_acc).item(),
        vmax=max(gp.y_acc).item(),
    )
    fig.colorbar(sc2, ax=axs[2], label="Function Value")
    axs[2].set_xlabel("X Coordinate")
    axs[2].set_ylabel("Y Coordinate")
    axs[2].set_title("Approximation")
    axs[2].set_xlim(bounds[0][0].item(), bounds[0][1].item())
    axs[2].set_ylim(bounds[1][0].item(), bounds[1][1].item())

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import scipy.spatial
import sys

eps = 1e-3


def in_box(towers: np.ndarray, bounding_box: list[float]):
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= towers[:, 0] + eps, towers[:, 0] <= bounding_box[1] + eps
        ),
        np.logical_and(
            bounding_box[2] <= towers[:, 1] + eps, towers[:, 1] <= bounding_box[3] + eps
        ),
    )


def voronoi(towers: np.ndarray, bounding_box: list[float]):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x
                    and x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y
                    and y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def centroid_region(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1]
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])


if __name__ == "__main__":
    n_towers = 100
    towers = np.random.rand(n_towers, 2)
    bounding_box = np.array([0.0, 1.0, 0.0, 1.0])  # [x_min, x_max, y_min, y_max]

    vor = voronoi(towers, bounding_box)

    fig = pl.figure()
    ax = fig.gca()
    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], "b.")
    # Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], "go")
    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], "k-")
    # Compute and plot centroids
    centroids = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = centroid_region(vertices)
        centroids.append(list(centroid[0, :]))
        ax.plot(centroid[:, 0], centroid[:, 1], "r.")

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    pl.savefig("bounded_voronoi.png")

    sp.spatial.voronoi_plot_2d(vor)
    pl.savefig("voronoi.png")

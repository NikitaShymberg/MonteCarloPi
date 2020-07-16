"""
This script estimates the value of pi using the "Monte Carlo" method.
https://en.wikipedia.org/wiki/Monte_Carlo_method

Let there be a circle with radius 1 and centre (0,0).
Surrounding the circle is a square with sides of length 2.

A_circle = pi * r^2
A_square = (2r)^2 = 4r^2
r^2 = A_circle / pi = A_square / 4
pi = 4 * (A_circle / A_square)

We can estimate the ratio (A_circle / A_square) by generating random points
on the plane between (-1, 1). We count the number of points that land
within the circle and the number of points outside the circle.
Then, A_circle / A_square ~= num_points_in_circle / total_number_of_points.

So, pi ~= 4 * num_points_in_circle / total_number_of_points

The goal of this project was to play around and learn intermediate matplotlib.
"""

import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Constants
RADIUS = 1  # Radius of circle
LENGTH = 2 * RADIUS  # Square side length
AREA_SQUARE = LENGTH ** 2  # Area of square
AREA_CIRCLE = np.pi * RADIUS ** 2  # Actual area of circle
NUM_ITER = 10000  # Number of random points to place
NUM_CHECKPOINTS = 100  # Number of snapshots to take of the animation
CHECKPOINT_ITER = NUM_ITER // NUM_CHECKPOINTS

# Axes array indexes
MAIN = 0
ESTIMATE = 1
ACC_OVER_TIME = 2


def generate_point() -> np.ndarray:
    """
    Generates a random point within the square,
    returns the coordinates of the point
    """
    return np.array([random.random() * LENGTH-1, random.random() * LENGTH-1])


def is_point_in_circle(point: np.ndarray) -> bool:
    """
    Given a `point`, returns whether it is within the circle or not
    """
    return point[0] ** 2 + point[1] ** 2 < RADIUS ** 2


def setup_plot() -> (plt.axes, plt.axes, plt.axes, plt.figure):
    """
    Sets up the initial plot. Returns the 3 axes and the figure.
    """
    fig = plt.gcf()
    gs = GridSpec(3, 3)
    ax_main = fig.add_subplot(gs[0:2, 0:2])  # Main circle/square
    ax_estimate = fig.add_subplot(gs[0, 2])  # Current estimate
    ax_acc_time = fig.add_subplot(gs[2, :])  # Estimate over time

    # Figure properties
    fig.set_size_inches(10, 10)

    # Main axis properties
    ax_main.grid()
    ax_main.set_xlim([-1.1, 1.1])
    ax_main.set_ylim([-1.1, 1.1])
    ax_main.set_title("Espimating Pi using Monte Carlo Methods")

    # Estimate axis properties
    ax_estimate.axis("off")
    ax_estimate.set_title("Current Estimate")

    # Estimate over time properties
    ax_acc_time.set_xlim([0, NUM_ITER])
    ax_acc_time.set_ylim([np.pi - 0.2, np.pi + 0.2])
    ax_acc_time.hlines(y=np.pi, xmin=0, xmax=NUM_ITER, colors="b")
    ax_acc_time.grid()
    ax_acc_time.set_title("Estimate over time")

    return (ax_main, ax_estimate, ax_acc_time), fig


def update_main_axes(axes: plt.axes, inside_circle: np.ndarray,
                     outside_circle: np.ndarray, circle: plt.Circle,
                     square: plt.Rectangle) -> None:
    """
    Updates the main axes with the new information prior to taking a snapshot.
    """
    axes.scatter(*(outside_circle.T), c="red", marker='.')
    axes.scatter(*(inside_circle.T), c="green", marker='.')
    axes.add_artist(circle)
    axes.add_artist(square)


def update_estimate_axes(axes: plt.axes, num_in_circle: int,
                         current_iter: int) -> float:
    """
    Updates the estimate axes with the new information
    prior to taking a snapshot.
    """
    pi_est = estimate_pi(num_in_circle, current_iter)
    pi_formatted = f"{pi_est:.5f}"
    current_iter_formatted = f"Current iteration: {current_iter - 1}"
    num_in_circle_formatted = f"Points inside circle: {num_in_circle}"
    num_out_circle_formatted = f"Points outside circle: {current_iter - num_in_circle}"
    error = np.abs(np.pi - pi_est)
    error_formatted = f"Error: {error:.5f}"
    percent_error = error / np.pi * 100
    percent_error_formatted = f"Relative error: {percent_error:.5f}%"

    axes.text(0, 0.7, pi_formatted, size=40, bbox={"facecolor": "b", "alpha": 0.5})
    axes.text(0, 0.4, current_iter_formatted, size=15)
    axes.text(0, 0.25, num_in_circle_formatted, size=15)
    axes.text(0, 0.1, num_out_circle_formatted, size=15)
    axes.text(0, -0.05, error_formatted, size=15)
    axes.text(0, -0.2, percent_error_formatted, size=15)

    return pi_est


def update_acc_time_axes(axes: plt.axes, estimates: [float]) -> None:
    """
    Updates the accuracy over time axes
    with the new information prior to taking a snapshot.
    """
    axes.hlines(y=np.pi, xmin=0, xmax=NUM_ITER, colors="b")
    axes.plot(
        range(0, CHECKPOINT_ITER * len(estimates), CHECKPOINT_ITER),
        estimates, color="orange")


def estimate_pi(num_in_circle: int, current_iter: int) -> float:
    """
    Given the `num_in_circle`, returns the current estimate of Pi.
    """
    return num_in_circle / current_iter * AREA_SQUARE


if __name__ == "__main__":
    random.seed(datetime.now())
    points_inside_circle = np.full((NUM_ITER, 2), np.nan)
    points_outside_circle = np.full((NUM_ITER, 2), np.nan)
    estimates = []

    # Set up plot
    axes, fig = setup_plot()
    camera = Camera(fig)

    # Add initial circle / square
    circle = plt.Circle((0, 0), radius=RADIUS, color='g', fill=False)
    rect = plt.Rectangle((-1, -1), 2, 2, color='r', fill=False)
    axes[MAIN].add_artist(circle)
    axes[MAIN].add_artist(rect)

    # Throw darts
    num_in_circle = 0
    num_outside_circle = 0
    for i in tqdm(range(NUM_ITER)):
        point = generate_point()
        if is_point_in_circle(point):
            points_inside_circle[num_in_circle] = point
            num_in_circle += 1
        else:
            points_outside_circle[num_outside_circle] = point
            num_outside_circle += 1
        # Periodic screenshots for animation
        if i % CHECKPOINT_ITER == 0:
            update_main_axes(axes[MAIN], points_inside_circle,
                             points_outside_circle, circle, rect)
            estimates.append(
                update_estimate_axes(axes[ESTIMATE], num_in_circle, i+1)
                )
            update_acc_time_axes(axes[ACC_OVER_TIME], estimates)
            camera.snap()

    # Show the animation
    anim = camera.animate()
    # Save the animation before showing it because
    # https://github.com/matplotlib/matplotlib/issues/10287
    anim.save("animation.mp4")
    plt.show()

    # Results
    print("Number of points inside the circle:", num_in_circle)
    print("Number of points outside the circle:", num_outside_circle)
    print("Final PI estimate:", estimate_pi(num_in_circle, NUM_ITER))
    print("Read Value of PI:", np.pi)

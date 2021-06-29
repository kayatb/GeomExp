# ==============================================================================
# Functions to plot the features according to their corresponding landmarks in a
# understandable way. That means, not merely the separate landmarks, but show
# how various landmarks together make a feature, e.g. eye aspect ratio is the
# eye height combined with the eye width.
# ==============================================================================

from scipy.interpolate import interp1d  # https://www.scipy.org
import numpy as np  # https://numpy.org
from matplotlib.patches import Ellipse  # https://matplotlib.org


POINT_SIZE = 7


def plot_ratio_six_points(plot, coordinates, landmarks, color):
    """ Plot the aspect ratio by plotting the width line and the height line.
    Landmarks contains 6 landmarks.
    Used for: eye aspect ratio L/R. """
    h1 = (coordinates[:, landmarks[0]] + coordinates[:, landmarks[1]]) / 2
    h2 = (coordinates[:, landmarks[2]] + coordinates[:, landmarks[3]]) / 2
    w1 = coordinates[:, landmarks[4]]
    w2 = coordinates[:, landmarks[5]]

    for point in [h1, h2, w1, w2]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([h1[0], h2[0]], [h1[1], h2[1]], color=color)
    plot.plot([w1[0], w2[0]], [w1[1], w2[1]], color=color)


def plot_ratio_four_points(plot, coordinates, landmarks, color):
    """ Plot the aspect ratio using four points.
    Landmarks contains 4 landmarks.
    Used for: mouth aspect ratio. """
    w1 = coordinates[:, landmarks[0]]
    w2 = coordinates[:, landmarks[1]]

    h1 = coordinates[:, landmarks[2]]
    h2 = coordinates[:, landmarks[3]]

    for point in [h1, h2, w1, w2]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([w1[0], w2[0]], [w1[1], w2[1]], color=color)
    plot.plot([h1[0], h2[0]], [h1[1], h2[1]], color=color)


def plot_two_points_line(plot, coordinates, landmarks, color):
    """ Plot a line between two points.
    Landmarks contains 2 landmarks.
    Used for: inner eye eyebrow centre distance L/R,
    inner eye mouth top distance L/R, mouth width, mouth height, eyebrow slope L/R. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]

    for point in [p1, p2]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)


def plot_angle_horizontal(plot, coordinates, landmarks, color):
    """ Plot the angle between two points and the horizontal line.
    Landmarks contains 2 landmarks.
    Used for: upper lip angle L/R, lower lip angle L/R, eyebrow angles L/R,
    lower outer eye angles L/R, lower inner eye angles L/R,
    mouth corner - bottom angle L/R, mouth corner up angle L/R. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]

    for point in [p1, p2]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
    plot.plot([p1[0], p2[0]], [p1[1], p1[1]], color=color)  # Horizontal line


def plot_angle_vertical(plot, coordinates, landmarks, color):
    """ Plot the angle between two points and the vertical line.
    Landmarks contains 2 landmarks.
    Used for: nose tip angles - mouth corner angles L/R. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]

    for point in [p1, p2]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
    plot.plot([p2[0], p2[0]], [p1[1], p2[1]], color=color)  # Vertical line


def plot_curve(plot, coordinates, landmarks, color):
    """ Plot a curve between three points.
    Landmarks contains 3 landmarks.
    Used for: curve lower outer lip L/R, curve lower inner lip L/R,
    curve lower lip. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]
    p3 = coordinates[:, landmarks[2]]

    x = [p1[0], p2[0], p3[0]]
    y = [p1[1], p2[1], p3[1]]

    xnew = np.linspace(p1[0], p3[0], num=50, endpoint=True)

    f_cubic = interp1d(x, y, kind='quadratic')
    for point in [p1, p2, p3]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot(xnew, f_cubic(xnew), color=color)


def plot_ellipse(plot, coordinates, landmarks, color):
    """ Plot an ellipse.
    Landmarks contains 6 landmarks.
    Used for: mouth opening. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]

    centre = (p1 + p2) / 2  # Centre point of ellipse.
    height = abs(p1 - p2)[1] * 1.2  # Enlarge the ellipse slightly to capture more of the mouth.

    p3 = coordinates[:, landmarks[2]]
    p4 = coordinates[:, landmarks[3]]
    w1 = (p3 + p4) / 2

    p5 = coordinates[:, landmarks[4]]
    p6 = coordinates[:, landmarks[5]]
    w2 = (p5 + p6) / 2

    width = abs(w1 - w2)[0] * 1.5  # Enlarge the ellipse slightly.

    # Don't scatter plot the points, since the ellipse doesn't capture them perfectly.
    # Angle of the ellipse is not taken into account for simplicity's sake.
    e = Ellipse(xy=centre, width=width, height=height,
                edgecolor=color, facecolor='none', linewidth=2)
    ax = plot.gca()
    # plot.add_artist(e)
    ax.add_patch(e)


def plot_three_points_line(plot, coordinates, landmarks, color):
    """ Plot a line from point 1 to point 2 and from point 2 to point 3.
    Landmarks contains 3 landmarks.
    Used for: mouth up/low, """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]
    p3 = coordinates[:, landmarks[2]]

    for point in [p1, p2, p3]:
        plot.scatter(*point, s=POINT_SIZE, color=color)

    plot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
    plot.plot([p2[0], p3[0]], [p2[1], p3[1]], color=color)


def plot_line_two_centres(plot, coordinates, landmarks, color):
    """ Plot a line from the middle of point 1 and point 2 to the middle of point
    3 and point 4.
    Landmarks contains 4 landmarks.
    Used for middle eye middle eyebrow distance L/R. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]
    p3 = coordinates[:, landmarks[2]]
    p4 = coordinates[:, landmarks[3]]

    top = (p1 + p2) / 2
    bottom = (p3 + p4) / 2

    plot.scatter(*top, color=color, s=POINT_SIZE)
    plot.scatter(*bottom, color=color, s=POINT_SIZE)
    plot.plot([top[0], bottom[0]], [top[1], bottom[1]], color=color)


def plot_three_points_centre(plot, coordinates, landmarks, color):
    """ Plot a line from the centre of point 1 and point 2 to point 3.
    Landmarks contains 3 landmarks.
    Used for: eye inner eyebrow distance L/R, upper mouth height, lower mouth height. """
    p1 = coordinates[:, landmarks[0]]
    p2 = coordinates[:, landmarks[1]]
    p3 = coordinates[:, landmarks[2]]

    top = (p1 + p2) / 2

    plot.scatter(*top, color=color, s=POINT_SIZE)
    plot.scatter(*p3, color=color, s=POINT_SIZE)
    plot.plot([top[0], p3[0]], [top[1], p3[1]], color=color)

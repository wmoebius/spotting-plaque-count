import numpy as np

from phageid.stitching import join_images, join_images_with_points
import matplotlib.pyplot as plt
from itertools import product


def test_join_images():
    images = {(i, j): np.ones((5, 5)) * (i * 3) + j for i in range(5) for j in range(3)}
    result = join_images(images, spacing=2, background_color=1)

    # # visual check
    # plt.imshow(result)
    # plt.show()

    assert result.shape == ((5 * 5 + 4 * 2), (3 * 5 + 2 * 2))
    assert (result[:, 5] == 1).all()
    assert (result[13, :] == 1).all()


def test_join_images_with_points():
    images = {(i, j): np.ones((5, 5)) * (i * 3) + j for i in range(5) for j in range(3)}

    expected_shape = ((5 * 5 + 4 * 2), (3 * 5 + 2 * 2))
    points_x = [(7 * i) + 2 for i in range(3)]
    points_y = [(7 * i) + 2 for i in range(5)]
    expected_p = np.array(list(product(points_x, points_y)))

    points_in = {key: np.array([[2, 2]]) for key in images.keys()}

    result_im, result_points = join_images_with_points(
        images, points_in, spacing=2, background_color=1
    )

    assert result_im.shape == ((5 * 5 + 4 * 2), (3 * 5 + 2 * 2))
    assert (result_im[:, 5] == 1).all()
    assert (result_im[13, :] == 1).all()

    # visual check
    plt.imshow(result_im)
    plt.scatter(*expected_p.T, marker="x", c="orange")
    plt.scatter(*result_points.T, marker="x", c="r")
    plt.show()

    # check points
    # TODO: there is a much faster way to do this.
    for value in result_points:
        assert value in expected_p

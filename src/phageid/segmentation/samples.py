import logging

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib.widgets import Button, Slider

from phageid.dtypes import D_ImageStack, ImageStack
from phageid.user_config import PATH_CONFIG, config

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_slider_values():
    config = toml.load(PATH_CONFIG)
    slider_config = {
        k: config[k]
        for k in [
            "x_spacing",
            "x_offset",
            "y_spacing",
            "y_offset",
            "rows",
            "columns",
            "radius",
        ]
    }
    return slider_config


def set_slider_values(sliders):
    """
    Updates the `current` values in the TOML file based on the current slider values.

    Parameters:
        filename (str): Path to the TOML file.
        sliders (dict): Dictionary of slider widgets.
    """
    import toml

    # Load the existing TOML data
    config = toml.load(PATH_CONFIG)

    # Update only the `current` values
    config["x_spacing"]["current"] = float(sliders["x_spacing"].val)
    config["x_offset"]["current"] = float(sliders["x_offset"].val)
    config["y_spacing"]["current"] = float(sliders["y_spacing"].val)
    config["y_offset"]["current"] = float(sliders["y_offset"].val)
    config["rows"]["current"] = int(sliders["rows"].val)
    config["columns"]["current"] = int(sliders["columns"].val)
    config["radius"]["current"] = float(sliders["radius"].val)

    # Write the updated data back to the TOML file
    with open(PATH_CONFIG, "w") as file:
        toml.dump(config, file)


def generate_image(
    dims, x_spacing, x_offset, y_spacing, y_offset, n_rows, n_columns, radius
):
    image = np.zeros(dims)
    for row in range(int(n_rows)):
        for col in range((n_columns)):
            x = col * x_spacing + x_offset
            y = row * y_spacing + y_offset

            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
                mask = (cc - x) ** 2 + (rr - y) ** 2 <= radius**2
                image[mask] = 1
    return image


def slice_image(
    image, x_spacing, x_offset, y_spacing, y_offset, n_rows, n_columns, radius
):
    """
    Slices the image into square areas, each centred on a circle.

    Parameters:
        image (np.ndarray): The input image array.
        x_spacing (float): Horizontal spacing between circle centers.
        x_offset (float): Horizontal offset for the first column.
        y_spacing (float): Vertical spacing between circle centers.
        y_offset (float): Vertical offset for the first row.
        n_rows (int): Number of rows of circles.
        n_columns (int): Number of columns of circles.
        radius (float): Radius of each circle.

    Returns:
        list of np.ndarray: List of square slices from the image.
    """
    slices = []
    indices = []
    for row in range(n_rows):
        for col in range(n_columns):
            # Calculate the center of the circle
            x_center = int(col * x_spacing + x_offset)
            y_center = int(row * y_spacing + y_offset)

            # Determine the square boundaries
            x_start = max(x_center - radius, 0)
            x_end = min(x_center + radius, image.shape[1])
            y_start = max(y_center - radius, 0)
            y_end = min(y_center + radius, image.shape[0])

            # Slice the image
            square_slice = image[int(y_start) : int(y_end), int(x_start) : int(x_end)]
            slices.append(square_slice)
            indices.append((row, col))
    return indices, slices


def slice_circle_bounding_boxes(images: ImageStack, centers, radius):
    """
    Slice the bounding boxes of circles from an image array.

    Parameters:
    - image: numpy array, the image from which to slice the circles.
    - centers: list of tuples, the (x, y) coordinates of the circle centers.
    - radius: int, the radius of the circles.

    Returns:
    - List of numpy arrays, each representing the bounding box of a circle.
    """
    bounding_boxes = []

    images_ = np.stack(images, axis=0)

    max_shape = 0, 0, 0
    for x, y in centers:
        # Calculate the bounding box coordinates
        x_min = int(max(0, x - radius))
        x_max = int(min(images_.shape[2], x + radius + 1))
        y_min = int(max(0, y - radius))
        y_max = int(min(images_.shape[1], y + radius + 1))

        # Slice the bounding box from the image
        bounding_box = images_[:, y_min:y_max, x_min:x_max]

        if bounding_box.shape > max_shape:
            max_shape = bounding_box.shape

        bounding_boxes.append(bounding_box)

    bounding_boxes = [
        pad_array(bb, max_shape) if bb.shape != max_shape else bb
        for bb in bounding_boxes
    ]

    return np.asarray(bounding_boxes)


def create_update_function(im, final_image, scaling_factor, sliders):
    def update(val):
        x_spacing = sliders["x_spacing"].val
        x_offset = sliders["x_offset"].val
        y_spacing = sliders["y_spacing"].val
        y_offset = sliders["y_offset"].val
        n_rows = int(sliders["rows"].val)
        n_columns = int(sliders["columns"].val)
        radius = sliders["radius"].val  # Get the new radius value

        new_image = generate_image(
            final_image[::scaling_factor, ::scaling_factor].shape,
            x_spacing,
            x_offset,
            y_spacing,
            y_offset,
            n_rows,
            n_columns,
            radius,
        )
        im.set_data(new_image)

    return update


def format_key(key):
    return key.replace("_", " ")


def pad_array(arr, target_shape):
    """
    Pads a NumPy array with zeros on the right and bottom to match the target shape.

    Parameters:
        arr (numpy.ndarray): The input array.
        target_shape (tuple): The desired shape (rows, cols).

    Returns:
        numpy.ndarray: The padded array.
    """
    # Compute padding sizes
    pad_sizes = [max(0, target_shape[i] - arr.shape[i]) for i in range(arr.ndim)]

    # Pad only on the right and bottom
    out = np.pad(
        arr,
        [(0, ps) for ps in pad_sizes],
        mode="constant",
        constant_values=0,
    )
    return out


def segment_samples(images: ImageStack, visualise: bool) -> D_ImageStack:
    final_image = images[-1].copy()

    # load config
    try:
        scaling_factor = int(config["sample_segmentation"]["scaling_factor"])
        logging.info(f"loaded scaling factor of: {scaling_factor}")
    except Exception as e:
        err_msg = f"invalid scaling factor {config['sample_segmentation']['scaling_factor']} specified in {FILE_CONFIG}: {e}"
        logging.error(err_msg)
        raise ValueError(err_msg)

    # Load slider values and limits from the TOML file
    slider_values = load_slider_values()

    if visualise:
        # Initialize the figure and axes
        fig, ax = plt.subplots()
        ax.imshow(final_image[::scaling_factor, ::scaling_factor])
        # plt.subplots_adjust(bottom=2.5)
        # Adjust spacing
        plt.subplots_adjust(
            left=0.1, bottom=0.5, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )

        # Generate the initial image
        image = generate_image(
            (
                final_image.shape[0] // scaling_factor,
                final_image.shape[1] // scaling_factor,
            ),
            slider_values["x_spacing"]["current"],
            slider_values["x_offset"]["current"],
            slider_values["y_spacing"]["current"],
            slider_values["y_offset"]["current"],
            slider_values["rows"]["current"],
            slider_values["columns"]["current"],
            slider_values["radius"]["current"],
        )

        im = ax.imshow(image, cmap="gray", alpha=0.2)

        # Slider parameters (including radius control)
        slider_params = [
            (key, value["min"], value["max"], value["current"])
            for key, value in slider_values.items()
        ]

        sliders = {}
        slider_axes = []

        # Create sliders for each parameter
        for i, (label, min_val, max_val, init_val) in enumerate(slider_params):
            ax_slider = plt.axes((0.2, 0.4 - 0.045 * i, 0.65, 0.03))
            slider = Slider(ax_slider, label, min_val, max_val, valinit=init_val)
            sliders[label] = slider
            slider_axes.append(ax_slider)

        # Create the update function without using globals
        update_function = create_update_function(
            im, final_image, scaling_factor, sliders
        )

        # Attach the update function to each slider
        for slider in sliders.values():
            slider.on_changed(update_function)

        # Add "Set" button
        def on_button_clicked(event):
            set_slider_values(sliders)
            plt.close(plt.gcf())

        button_ax = plt.axes((0.4, 0.4 - 0.045 * (len(slider_params) + 1), 0.25, 0.06))
        button = Button(button_ax, "Set", color="lightblue", hovercolor="grey")
        button.on_clicked(on_button_clicked)

        fig.canvas.draw_idle()
        plt.show()

        x_spacing = sliders["x_spacing"].val
        x_offset = sliders["x_offset"].val
        y_spacing = sliders["y_spacing"].val
        y_offset = sliders["y_offset"].val
        n_rows = int(sliders["rows"].val)
        n_columns = int(sliders["columns"].val)
        radius = sliders["radius"].val

        set_slider_values(sliders=sliders)
    else:
        # read directly form the config
        x_spacing = config["x_spacing"]["current"]
        x_offset = config["x_offset"]["current"]
        y_spacing = config["y_spacing"]["current"]
        y_offset = config["y_offset"]["current"]
        n_rows = config["rows"]["current"]
        n_columns = config["columns"]["current"]
        radius = config["radius"]["current"]

    centre_xs = (
        ((np.arange(n_columns) * x_spacing) + x_offset) * scaling_factor
    ).astype(int)
    centre_ys = (((np.arange(n_rows) * y_spacing) + y_offset) * scaling_factor).astype(
        int
    )

    centres = []
    centre_inds = []
    for i, cx in enumerate(centre_xs):
        for j, cy in enumerate(centre_ys):
            centres.append(np.array((cx, cy)))
            centre_inds.append((j, i))

    circles = slice_circle_bounding_boxes(images, centres, radius * scaling_factor)
    circles = {ind: circ for ind, circ in zip(centre_inds, circles)}

    if visualise:
        plt.figure()
        plt.imshow(final_image)
        plt.scatter(*np.vstack(centres).T, c="r", marker="x")
        plt.show()
    return circles

import numpy as np
from pathlib import Path
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from phageid import FILE_CONFIG
import logging
import toml
from phageid import config
from itertools import product

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_slider_values():
    config = toml.load(FILE_CONFIG)
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
    config = toml.load(FILE_CONFIG)

    # Update only the `current` values
    config["x_spacing"]["current"] = float(sliders["x_spacing"].val)
    config["x_offset"]["current"] = float(sliders["x_offset"].val)
    config["y_spacing"]["current"] = float(sliders["y_spacing"].val)
    config["y_offset"]["current"] = float(sliders["y_offset"].val)
    config["rows"]["current"] = int(sliders["rows"].val)
    config["columns"]["current"] = int(sliders["columns"].val)
    config["radius"]["current"] = float(sliders["radius"].val)

    # Write the updated data back to the TOML file
    with open(FILE_CONFIG, "w") as file:
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


def slice_circle_bounding_boxes(images, centers, radius):
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

    max_shape = 0, 0, 0
    for x, y in centers:
        # Calculate the bounding box coordinates
        x_min = int(max(0, x - radius))
        x_max = int(min(images.shape[2], x + radius + 1))
        y_min = int(max(0, y - radius))
        y_max = int(min(images.shape[1], y + radius + 1))

        # Slice the bounding box from the image
        bounding_box = images[:, y_min:y_max, x_min:x_max]

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
    pad_height = max(0, target_shape[-2] - arr.shape[-2])  # Rows to add at the bottom
    pad_width = max(0, target_shape[-1] - arr.shape[-1])  # Columns to add on the right
    pad_sizes = [max(0, target_shape[i] - arr.shape[i]) for i in range(arr.ndim)]

    # Pad only on the right and bottom
    out = np.pad(
        arr,
        [(0, ps) for ps in pad_sizes],
        mode="constant",
        constant_values=0,
    )
    return out


def segment_samples(input_file: Path, output_dir: Path, visualise: bool):
    # confirm dirs
    if not input_file.is_file():
        logging.error(f"Input file {input_file} does not exist.")
        raise ValueError(f"Input file {input_file} does not exist.")

    if not output_dir.is_dir():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except Exception:
            logging.error(f"Output directory {output_dir} is not a valid directory.")
            raise ValueError(
                f"Output directory '{output_dir}' is not a valid directory."
            )

    # load images
    try:
        images = np.load(input_file)
        logging.info(f"Loaded images from {input}")
        assert isinstance(images, np.ndarray)
    except Exception as e:
        logging.error(f"Failed to load data from {input_file}: {e}")
    final_image = images[-1].copy()

    # load config
    try:
        scaling_factor = int(config["sample_segmentation"]["scaling_factor"])
        logging.info(f"loaded scaling factor of: {scaling_factor}")
    except Exception as e:
        logging.error(
            f"invalid scaling factor {config['sample_segmentation']['scaling_factor']} specified in {FILE_CONFIG}: {e}"
        )

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
            ax_slider = plt.axes([0.2, 0.4 - 0.045 * i, 0.65, 0.03])
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

        button_ax = plt.axes([0.4, 0.4 - 0.045 * (len(slider_params) + 1), 0.25, 0.06])
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

    centres = np.array(list(product(centre_xs, centre_ys)))
    centre_inds = list(product(range(n_columns), range(n_rows)))
    centres.shape

    circles = slice_circle_bounding_boxes(images, centres, radius * scaling_factor)
    circles = {ind: circ for ind, circ in zip(centre_inds, circles)}

    if visualise:
        plt.figure()
        plt.imshow(final_image)
        plt.scatter(*centres.T, c="r", marker="x")
        # for c in circles[0, 0]:
        #     plt.figure()
        #     plt.imshow(c)

    for (i, j), sample in circles.items():
        out_path = output_dir / "sample_{:02d}_{:02d}.npy".format(i, j)
        np.save(out_path, sample)
        logging.info(f"Saved sample to {out_path}")

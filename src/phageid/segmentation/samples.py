import numpy as np
from pathlib import Path
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from phageid import DIR_DATA, FILE_CONFIG
import logging
from datetime import datetime
from os import makedirs
import toml

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_slider_values():
    return toml.load(FILE_CONFIG)


def set_slider_values(sliders):
    """
    Updates the `current` values in the TOML file based on the current slider values.

    Parameters:
        filename (str): Path to the TOML file.
        sliders (dict): Dictionary of slider widgets.
    """
    import toml

    # Load the existing TOML data
    data = toml.load(FILE_CONFIG)

    # Update only the `current` values
    data["x_spacing"]["current"] = float(sliders["x_spacing"].val)
    data["x_offset"]["current"] = float(sliders["x_offset"].val)
    data["y_spacing"]["current"] = float(sliders["y_spacing"].val)
    data["y_offset"]["current"] = float(sliders["y_offset"].val)
    data["rows"]["current"] = int(sliders["rows"].val)
    data["columns"]["current"] = int(sliders["columns"].val)
    data["radius"]["current"] = float(sliders["radius"].val)

    # Write the updated data back to the TOML file
    with open(FILE_CONFIG, "w") as file:
        toml.dump(data, file)


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


def segment_samples(input_file: Path, visualise: bool):
    assert input_file.is_file()

    images = np.load(input_file)
    final_image = images[-1].copy()

    if visualise:
        plt.imshow(final_image)

    # In[5]:

    datetime_str = f"{str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}"

    output_dir = DIR_DATA / f"processed/splits_{datetime_str}"
    makedirs(output_dir)

    # In[6]:

    # Function to create the grid of circle centers
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

    # In[7]:

    scaling_factor = 10

    print(final_image.shape)
    print(final_image[::scaling_factor, ::scaling_factor].shape)
    # Initialize the figure and axes
    fig, ax = plt.subplots()
    ax.imshow(final_image[::scaling_factor, ::scaling_factor])
    plt.subplots_adjust(bottom=0.5)

    # Load slider values and limits from the TOML file
    slider_values = load_slider_values()

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

    def format_key(key):
        return key.replace("_", " ")

    # Slider parameters (including radius control)
    slider_params = [
        (key, value["min"], value["max"], value["current"])
        for key, value in slider_values.items()
    ]

    sliders = {}
    slider_axes = []

    # Create sliders for each parameter
    for i, (label, min_val, max_val, init_val) in enumerate(slider_params):
        ax_slider = plt.axes([0.2, 0.35 - 0.05 * i, 0.65, 0.03])
        slider = Slider(ax_slider, label, min_val, max_val, valinit=init_val)
        sliders[label] = slider
        slider_axes.append(ax_slider)

    # Update function for the sliders
    def update(val):
        global im
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

    # Attach the update function to each slider
    for slider in sliders.values():
        slider.on_changed(update)

    # Add "Set" button
    def on_button_clicked(event):
        print("Settings applied:")
        print(f"X Spacing: {sliders['x spacing'].val}")
        print(f"X Offset: {sliders['x offset'].val}")
        print(f"Y Spacing: {sliders['y spacing'].val}")
        print(f"Y Offset: {sliders['y offset'].val}")
        print(f"Rows: {int(sliders['rows'].val)}")
        print(f"Columns: {int(sliders['columns'].val)}")
        print(f"Radius: {sliders['radius'].val}")  # Show the selected radius

    button_ax = plt.axes([0.4, 0.35 - 0.05 * (len(slider_params) + 1), 0.25, 0.03])
    button = Button(button_ax, "Set", color="lightblue", hovercolor="blue")
    button.on_clicked(on_button_clicked)

    # Show the plot
    fig.canvas.draw_idle()

    plt.show()

    # In[8]:

    x_spacing = sliders["x_spacing"].val
    x_offset = sliders["x_offset"].val
    y_spacing = sliders["y_spacing"].val
    y_offset = sliders["y_offset"].val
    n_rows = int(sliders["rows"].val)
    n_columns = int(sliders["columns"].val)
    radius = sliders["radius"].val

    set_slider_values(sliders=sliders)

    # In[9]:

    centre_xs = (
        ((np.arange(n_columns) * x_spacing) + x_offset) * scaling_factor
    ).astype(int)
    centre_ys = (((np.arange(n_rows) * y_spacing) + y_offset) * scaling_factor).astype(
        int
    )

    # In[10]:

    from itertools import product

    centres = np.array(list(product(centre_xs, centre_ys)))
    centre_inds = list(product(range(n_columns), range(n_rows)))
    centres.shape

    # In[11]:

    if show_plots:
        plt.figure()
        plt.imshow(final_image)
        plt.scatter(*centres.T, c="r", marker="x")

    # In[15]:

    radius

    # In[19]:

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

        for x, y in centers:
            # Calculate the bounding box coordinates
            x_min = int(max(0, x - radius))
            x_max = int(min(images.shape[2], x + radius + 1))
            y_min = int(max(0, y - radius))
            y_max = int(min(images.shape[1], y + radius + 1))

            # Slice the bounding box from the image
            bounding_box = images[:, y_min:y_max, x_min:x_max]
            bounding_boxes.append(bounding_box)

        return np.asarray(bounding_boxes)

    # In[20]:

    circles = slice_circle_bounding_boxes(images, centres, radius * scaling_factor)
    circles = {ind: circ for ind, circ in zip(centre_inds, circles)}

    # In[21]:

    if visualise:
        for c in circles[0, 0]:
            plt.figure()
            plt.imshow(c)

    # In[70]:

    for (i, j), sample in circles.items():
        out_path = output_dir / "sample_{:02d}_{:02d}.npy".format(i, j)
        np.save(out_path, sample)
        logging.info(f"Saved sample to {out_path}")

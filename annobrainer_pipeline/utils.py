from __future__ import division

import numpy as np
from pathlib import Path
import torch.nn.functional as F
from PIL import Image, ImageDraw

from airlab.transformation.utils import compute_grid
from airlab.utils import image as iutils
from airlab.utils.image import image_from_numpy

np.random.seed(0)


def warp_image(image, displacement):
    """
    This warp_image uses "border" padding instead of default zeros.
    """
    image_size = image.size

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)

    # warp image
    warped_image = F.grid_sample(
        image.image, displacement + grid, padding_mode="border"
    )

    return iutils.Image(warped_image, image_size, image.spacing, image.origin)


def prepare_grid(image, displacement):
    """
    Shift the displacement.
    """
    image_size = image.size
    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)
    return displacement + grid


def renormalize(n, range1, range2):
    """
    Renormalize a value from one range to another, e.g. 0-1 to 0-100.
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def warp_rgb_image(image, displacement, dtype, device):
    """
    Workaround to warp n-channel image even when airlab works with grayscale images.
    """
    out = []
    for i in range(image.shape[-1]):
        image_t = image_from_numpy(
            image[:, :, i], (1.0, 1.0), (0.0, 0.0), dtype, device
        )
        warped_image_t = warp_image(image_t, displacement)
        out.append(warped_image_t.numpy())
    return np.stack(out, axis=2).astype(np.uint8)


def theta2param(theta, w, h):
    """
    Get affine transformation matrix from Airlab in the right format.
    """
    if theta.shape == (2, 3):
        theta = np.vstack((theta, [0, 0, 1]))
    theta = np.linalg.inv(theta)
    param = np.zeros([2, 3])
    param[0, 0] = theta[0, 0]
    param[0, 1] = theta[0, 1] / h * w
    param[0, 2] = (theta[0, 2] - theta[0, 0] - theta[0, 1] + 1) * w / 2
    param[1, 0] = theta[1, 0] / w * h
    param[1, 1] = theta[1, 1]
    param[1, 2] = (theta[1, 2] - theta[1, 0] - theta[1, 1] + 1) * h / 2
    return param


def generate_unit_squares(image_width, image_height):
    """Generate coordinates for a tiling of unit squares."""
    for x in range(image_width):
        for y in range(image_height):
            yield [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]


def generate_squares(image_width, image_height, side_length=3):
    """Generate coordinates for a tiling of squares."""
    scaled_width = int(image_width / side_length) + 2
    scaled_height = int(image_height / side_length) + 2
    for coords in generate_unit_squares(scaled_width, scaled_height):
        yield [(x * side_length, y * side_length) for (x, y) in coords]


def create_square_grid(height: int, width: int, side_length: int = 20):
    """Create square grid based on parameters."""
    im = Image.new("RGB", size=(width, height))
    coordinates = generate_squares(width, height, side_length=20)
    for idx, coords in enumerate(coordinates):
        ImageDraw.Draw(im).polygon(coords)
    return np.array(im)


def prepare_image(img_folder, img_name):
    """
    export third layer from provided bright-field svs image and save it as a file with png extension
    Args:
        img_folder: Folder where the image is located
        img_name: Name of the image

    Returns:
        None
    """
    img_path = Path(img_folder) / img_name
    img_ptr = HandleSingleSvs(str(img_path))
    third_layer = img_ptr.extract_xth_layer(3, 6000)
    third_layer.save(str(img_path.with_suffix(".png")))
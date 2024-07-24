import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image

import airlab as al
from airlab.utils.image import image_from_numpy
from annobrainer_pipeline.utils import theta2param, warp_image

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def affine(fixed, moving, channel, device, dtype, plot=False):
    start = time.time()
    # read all RGB channels
    if channel["moving"] == 4:
        moving_image = al.read_image_as_tensor(
            moving, dtype=dtype, device=device
        )
    else:
        # read only selected channels
        # R 0, G 1, B 2, R-B/3 3
        # moving
        moving_image = np.array(Image.open(moving))
        # transform to RGB
        im = moving_image[:, :, 0] - (moving_image[:, :, 2] / 3)
        moving_image = np.concatenate((moving_image, im[:, :, None]), axis=2)
        moving_image = image_from_numpy(
            moving_image[:, :, channel["moving"]],
            (1.0, 1.0),
            (0.0, 0.0),
            dtype=dtype,
            device=device,
        )
    if channel["fixed"] == 4:
        fixed_image = al.read_image_as_tensor(
            fixed, dtype=dtype, device=device
        )
    else:
        # fixed
        fixed_image = np.array(Image.open(fixed))
        # transform to RGB
        im = fixed_image[:, :, 0] - (fixed_image[:, :, 2] / 3)
        fixed_image = np.concatenate((fixed_image, im[:, :, None]), axis=2)
        fixed_image = image_from_numpy(
            fixed_image[:, :, channel["fixed"]],
            (1.0, 1.0),
            (0.0, 0.0),
            dtype=dtype,
            device=device,
        )

    fixed_image, moving_image = al.utils.normalize_images(
        fixed_image, moving_image
    )

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration(verbose=False)

    # choose the affine transformation model
    transformation = al.transformation.pairwise.SimilarityTransformation(
        moving_image, opt_cm=True
    )
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.NCC(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(
        transformation.parameters(), lr=0.01, amsgrad=True
    )

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(1000)

    # start the registration
    registration.start(EarlyStopping=False)

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = warp_image(moving_image, displacement)

    end = time.time()
    print("Affine registration done in: {} seconds".format(end - start))

    if plot:
        print(
            "================================================================="
        )

        # plot the results
        plt.subplot(131)
        plt.imshow(fixed_image.numpy(), cmap="gray")
        plt.title("Fixed Image")

        plt.subplot(132)
        plt.imshow(moving_image.numpy(), cmap="gray")
        plt.title("Moving Image")

        plt.subplot(133)
        plt.imshow(warped_image.numpy(), cmap="gray")
        plt.title("Warped Moving Image")

        plt.show()

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title("WARPED_affine")
        ax.imshow(warped_image.numpy(), cmap="gray")

    # get displacement
    displacement = transformation.get_displacement()
    # get trasformation matrix
    matrix = transformation.transformation_matrix.detach().cpu().numpy()
    A = theta2param(
        matrix, moving_image.numpy().shape[1], moving_image.numpy().shape[0]
    )
    # get final Loss
    loss = float(registration._closure())
    return displacement, A, loss

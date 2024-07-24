import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image

import airlab as al
from airlab.utils.image import image_from_numpy
from annobrainer_pipeline.airlab_custom_loss_functions import PM
from annobrainer_pipeline.utils import prepare_grid

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def demons(fixed, moving, channel, device, dtype, plot=False):
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
    # bring them on a joint domain
    (
        fixed_image,
        f_mask,
        moving_image,
        m_mask,
        cm_displacement,
    ) = al.get_joint_domain_images(
        fixed_image,
        moving_image,
        default_value=1,
        cm_alignment=False,
        compute_masks=False,
    )

    # create pairwise registration object
    registration = al.DemonsRegistraion(verbose=False)

    # choose the affine transformation model
    transformation = al.transformation.pairwise.NonParametricTransformation(
        moving_image.size, dtype=dtype, device=device, diffeomorphic=True
    )

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.NCC(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose a regulariser for the demons
    regulariser = al.regulariser.demons.GaussianRegulariser(
        moving_image.spacing, sigma=[2, 2], dtype=dtype, device=device
    )

    # edge_updater = al.regulariser.demons.EdgeUpdaterDisplacementIntensities(moving_image.spacing, moving_image.image)
    # regulariser1 = al.regulariser.demons.GraphDiffusionRegulariser(moving_image.size, moving_image.spacing, edge_updater, phi=1, dtype=th.float32, device=device)

    registration.set_regulariser([regulariser])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(1000)

    # start the registration
    registration.start()

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    # use the shaded version of the fixed image for visualization
    warped_image = al.transformation.utils.warp_image(
        moving_image, displacement
    )
    displacement = al.create_displacement_image_from_image(
        displacement, moving_image
    )

    end = time.time()
    print("Elastic registration done in: {} seconds".format(end - start))

    if plot:
        # plot the results
        plt.subplot(221)
        plt.imshow(fixed_image.numpy(), cmap="gray")
        plt.title("Fixed Image")

        plt.subplot(222)
        plt.imshow(moving_image.numpy(), cmap="gray")
        plt.title("Moving Image")

        plt.subplot(223)
        plt.imshow(warped_image.numpy(), cmap="gray")
        plt.title("Warped Moving Image")

        plt.subplot(224)
        plt.imshow(displacement.magnitude().numpy(), cmap="jet")
        plt.title("Magnitude Displacement")

        plt.show()

    # get displacement
    displacement = transformation.get_displacement()
    # get inverse displacement
    inverse_displacement = prepare_grid(
        moving_image, transformation.get_inverse_displacement()
    )
    # get final Loss
    loss = float(registration._closure())

    return (
        displacement,
        inverse_displacement,
        moving_image.numpy().shape,
        loss,
    )


def diffeomorphic_bspline(
    fixed,
    moving,
    landmarks,
    fixed_points,
    moving_points,
    channel,
    device,
    dtype,
    plot=False,
):
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
    # bring them on a joint domain
    (
        fixed_image,
        f_mask,
        moving_image,
        m_mask,
        cm_displacement,
    ) = al.get_joint_domain_images(
        fixed_image,
        moving_image,
        default_value=1,
        cm_alignment=False,
        compute_masks=False,
    )

    # create an image pyramide size/8, size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(
        fixed_image, [[4, 4], [2, 2]]
    )
    moving_image_pyramid = al.create_image_pyramid(
        moving_image, [[4, 4], [2, 2]]
    )

    if landmarks:
        # create an points pyramide size/4, size/2, size/
        fixed_pts = np.loadtxt(fixed_points, delimiter=",")
        moving_pts = np.loadtxt(moving_points, delimiter=",")

        fixed_pts = [fixed_pts / 4, fixed_pts / 2, fixed_pts]
        moving_pts = [moving_pts / 4, moving_pts / 2, moving_pts]

    regularisation_weight = [10000, 100000, 1000000]
    number_of_iterations = [500, 500, 500]
    sigma = [[11, 11], [11, 11], [3, 3]]

    constant_flow = None

    for level, (mov_im_level, fix_im_level) in enumerate(
        zip(moving_image_pyramid, fixed_image_pyramid)
    ):

        registration = al.PairwiseRegistration(verbose=False)

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(
            mov_im_level.size,
            sigma=sigma[level],
            order=3,
            dtype=dtype,
            device=device,
            diffeomorphic=True,
        )

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(
                constant_flow, mov_im_level.size, interpolation="linear"
            )
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        # changed to NCC
        image_loss = al.loss.pairwise.NCC(fix_im_level, mov_im_level)
        if landmarks:
            fixed_pts_level = fixed_pts[level]
            moving_pts_level = moving_pts[level]

            points = {"fixed": fixed_pts_level, "moving": moving_pts_level}

            # custom loss function for landmarks (PointsMatching)
            image_loss1 = PM(fix_im_level, mov_im_level, points=points)

            registration.set_image_loss([image_loss1, image_loss])
        else:
            registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        regulariser = al.regulariser.displacement.DiffusionRegulariser(
            mov_im_level.spacing
        )
        regulariser.set_weight(regularisation_weight[level])

        # regulariser1 = al.regulariser.displacement.TVRegulariser(mov_im_level.spacing)
        # regulariser1.set_weight(regularisation_weight[level])

        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters())

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_flow = transformation.get_flow()
    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(
        moving_image, displacement
    )

    displacement = al.create_displacement_image_from_image(
        displacement, moving_image
    )

    # create inverse displacement field
    inverse_displacement = transformation.get_inverse_displacement()
    # inverse_warped_image = al.transformation.utils.warp_image(warped_image, inverse_displacement)
    # inverse_displacement = al.create_displacement_image_from_image(inverse_displacement, moving_image)

    end = time.time()
    print("Elastic registration done in: {} seconds".format(end - start))

    if plot:
        print(
            "================================================================="
        )

        # plot the results
        plt.subplot(241)
        plt.imshow(fixed_image.numpy(), cmap="gray")
        plt.title("Fixed Image")

        plt.subplot(242)
        plt.imshow(moving_image.numpy(), cmap="gray")
        plt.title("Moving Image")

        plt.subplot(243)
        plt.imshow(warped_image.numpy(), cmap="gray")
        plt.title("Warped Moving Image")

        plt.subplot(244)
        plt.imshow(displacement.magnitude().numpy(), cmap="jet")
        plt.title("Magnitude Displacement")

        plt.show()

    # get displacement
    displacement = transformation.get_displacement()
    # displacement full coordinates
    displacement_full_coords = prepare_grid(
        moving_image, transformation.get_displacement()
    )
    # get inverse displacement
    inverse_displacement = prepare_grid(
        moving_image, transformation.get_inverse_displacement()
    )
    # get final Loss
    loss = float(registration._closure())

    return (
        displacement,
        displacement_full_coords,
        inverse_displacement,
        moving_image.numpy().shape,
        loss,
    )

import os
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from PIL import Image

import annobrainer_pipeline.affine_registration
import annobrainer_pipeline.elastic_registration
import annobrainer_pipeline.master_code_settings as mc_settings
from annobrainer_pipeline.landmarks import main as landmarks_estimation
from annobrainer_pipeline.utils import create_square_grid, renormalize, warp_rgb_image

# ENVIROMENT ===================================================================
np.random.seed(0)
th.manual_seed(0)
# noinspection PyUnresolvedReferences
th.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
th.backends.cudnn.benchmark = False


# AUX FUNCTIONS ================================================================
def stack_col_of_ones(arr):
    return np.hstack((arr, np.ones((len(arr), 1))))


def set_registration_parameters(
    config,
    results_path=None,
    base_moving_folder=None,
    base_fixed_folder=None,
    moving_images_path=None,
    moving_images_annotations_path=None,
    moving_images_pairing=None,
    fixed_images_path=None,
    fixed_images_annotations_path=None,
    fixed_images_pairing=None,
    channel_moving=0,
    channel_fixed=2,
    image_matching_indices=None,
    affine_registration_method="affine",
    elastic_registration_method="diffeomorphic_bspline",
    gpu_id=0,
    increase_folder_version=None,
    save_loss_csv=False,
    landmarks=False,
):
    """Author: Martin Vagenknecht
    Function sets parameters for registration.

    Parameters:
        config : json-like dictionary
            default = None
        results_path : str or Path, optional
            default =  str(
            mc_settings.machine_specific_default_path()
            / "registration_results"
            / str(date.today())
            / results_experiment_name
        )
        base_moving_folder : str or Path, required
        base_fixed_folder : str or Path, required
        moving_images_path : str or Path, optional
            default = str(Path(base_moving_folder) / "images")
        moving_images_annotations_path : str or Path, optional
            default = str(
                        Path(base_moving_folder) / "annotations"
                        )
        moving_images_pairing : str or Path, optional
            default = str(Path(base_moving_folder) / "pairing.csv")
        fixed_images_path : str or Path, optional
            default = str(Path(base_fixed_folder) / "images")
        fixed_images_annotations_path : str or Path, optional
            default = str(
                            Path(base_fixed_folder) /
                            "annotations_named_complete_csv"
                        )
        fixed_images_pairing : str or Path, optional
            default = str(Path(base_fixed_folder) / "pairing.csv")
        channel_moving : str or Path, optional
            default = 0
        channel_fixed : str or Path, optional
            default = 2
        image_matching_indices : list of dictionaries, required
            Matching indices of moving and fixed pairing.csv files.
            default = None
        affine_registration_method : str, optional
            default = \"affine\"
        elastic_registration_method : str, optional
            default = \"diffeomorphic_bspline\"
        gpu_id : str or Path, required
            default = 0
        save_loss_csv : bool, optional
            Flag for saving loss function values
            default = False
        landmarks: bool, optional
            default = False
            additional regularization term for registration
        increase_folder_version: bool, required
            Index of hemisphere to be anotated,
            needed for proper folder version controling.


    Returns:
      config : json-like dictionary
    """

    # Subset annotations
    if "subset_fixed_annotations" not in config["init_info"]:
        subset_fixed_annotations = {}
        for single_region in config["init_info"]["region"]:
            subset_fixed_annotations[single_region] = single_region
    elif config["init_info"]["subset_fixed_annotations"] == "None":
        subset_fixed_annotations = None
    else:
        subset_fixed_annotations = config["init_info"][
            "subset_fixed_annotations"
        ]
    
    if "subset_moving_annotations" not in config["init_info"]:
        subset_moving_annotations = None
    elif config["init_info"]["subset_moving_annotations"] == "None":
        subset_moving_annotations = None
    else:
        subset_moving_annotations = config["init_info"][
            "subset_moving_annotations"
        ]

    # moving folder setting
    if moving_images_path is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_path = str(Path(base_moving_folder) / "images")

    if moving_images_pairing is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_pairing = str(Path(base_moving_folder) / "pairing.csv")

    if moving_images_annotations_path is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_annotations_path = str(
            Path(base_moving_folder) / "annotations"
        )

    # fixed folder setting
    if fixed_images_path is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_path = str(Path(base_fixed_folder) / "images")

    if fixed_images_pairing is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_pairing = str(Path(base_fixed_folder) / "pairing.csv")

    if fixed_images_annotations_path is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_annotations_path = str(
            Path(base_fixed_folder) / "annotations_named_complete_csv"
        )

    results_experiment_name = (
        config["init_info"]["study_id"]
        + "_"
        + config["init_info"]["stain"]
        + "_"
        + "-".join(config["init_info"]["region"])
        + "_"
        + config["init_info"]["slide_number"]
        + "_"
        + config["init_info"]["layer_atlas"]
    )

    if results_path is None:
        results_path = str(
            mc_settings.machine_specific_default_path()
            / "registration_results"
            / str(date.today())
            / results_experiment_name
        )

    if len(str(results_path)) >= 240:
        results_path = str(
            mc_settings.machine_specific_default_path()
            / "registration_results"
            / str(date.today())
            / results_experiment_name
        )

        print(
            "Registration result path is too long. Will be shortened to"
            " default path: " + results_path + ". Later, we will try to "
            "copy the results into experiment folder as well."
        )

    version = 1

    results_path_appr = results_path + "_v" + str(version)

    # if we are annotating both sides at the same time we want the results to be in the same folder
    if len(config["init_info"]["side_to_annotate"]) == 1:

        while os.path.exists(results_path_appr):
            results_path_appr = results_path + "_v" + str(version)
            version += 1

    if len(config["init_info"]["side_to_annotate"]) == 2:

        if increase_folder_version is False:
            while os.path.exists(results_path_appr):
                results_path_appr = results_path + "_v" + str(version)
                version += 1

        if increase_folder_version is True:
            while os.path.exists(results_path_appr):
                version += 1
                results_path_appr = results_path + "_v" + str(version)
            results_path_appr = results_path + "_v" + str(version - 1)

    Path(results_path_appr).mkdir(parents=True, exist_ok=True)

    parameters = {
        "results_path": results_path_appr,
        "moving_images_path": moving_images_path,
        "moving_images_annotations_path": moving_images_annotations_path,
        "moving_images_pairing": moving_images_pairing,
        "fixed_images_path": fixed_images_path,
        "fixed_images_annotations_path": fixed_images_annotations_path,
        "fixed_images_pairing": fixed_images_pairing,
        "channel": {"moving": channel_moving, "fixed": channel_fixed},
        "image_matching": image_matching_indices,
        "affine_registration_method": affine_registration_method,
        "elastic_registration_method": elastic_registration_method,
        "gpu_id": gpu_id,
        "save_loss_csv": save_loss_csv,
        "subset_moving_annotations": subset_moving_annotations,
        "subset_fixed_annotations": subset_fixed_annotations,
        "landmarks": landmarks,
    }

    return parameters


def test_result_folder_lens(path):
    if len(str(path)) >= 200:
        raise Exception(
            "\n Path: " + path + " is too long. Code will fail. The"
            " MAX_PATH is 256 chars :("
        )

    return "OK"


def translation(
    an: np.ndarray, ofset: Tuple[int] = [0, 0], **kwargs: Any
) -> np.array:
    """Offset of each element."""
    # ofset
    an = an - ofset
    an = an.astype(int)

    return an


def scaling(
    an: np.ndarray, scaling_factor: float = 1.0, **kwargs: Any
) -> np.array:
    """Scaling of each element."""
    # mulitiply by ratio
    an = an * scaling_factor
    an = np.rint(an)
    an = an.astype(int)

    return an


def affine_transformation(
    an: np.ndarray, A: np.ndarray, inverse: bool = False
) -> np.array:
    """Affine transformation of each element."""
    # transform coordinates
    an_ = stack_col_of_ones(an).transpose()

    if inverse:

        def theta(theta: np.array) -> np.array:
            """Get affine transformation matrix from Airlab in the right format."""
            if theta.shape == (2, 3):
                theta = np.vstack((theta, [0, 0, 1]))
            theta = np.linalg.inv(theta)
            return theta[0:2, :]

        A = theta(A)

    an_warped = A @ an_
    an_warped = np.round(an_warped.astype(np.int32).transpose(), decimals=0)
    an = an_warped

    return an


def elastic_transformation(
    an: np.ndarray, displacement: np.ndarray, shape: Tuple
) -> np.array:
    """Elastic transformation of each element."""
    an_tr = []
    for p in an:
        d = displacement[0][p[1]][p[0]]
        x = int(renormalize(d[0], (-1, 1), (0, shape[1])))
        y = int(renormalize(d[1], (-1, 1), (0, shape[0])))
        an_tr.append([x, y])
        an = an_tr

    return an


def plot_function(
    an: np.ndarray, ann_idx: int, ax: plt.axis, label: str, color: str
) -> plt.axes:
    """plot annotations to axes object."""
    coord = an.tolist()
    coord.append(coord[0])  # repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord)  # create lists of x and y values
    if ann_idx == 0:
        ax.plot(xs, ys, label=label, color=color)
    else:
        ax.plot(xs, ys, label="_nolegend_", color=color)

    return ax


def map_funcs(
    obj: object, func_list: List[Callable], func_params: Dict
) -> object:
    """Apply list of functions to object."""
    for idx, f in enumerate(func_list):
        obj = f(obj, **func_params[idx])

    return obj


def change_last_element(path, file_prexif, sep="_"):

    last = str(path).split("/")[-1]
    file_prexif = str(file_prexif) + str(sep)

    A = str({}) + str(last)
    A = A.format(file_prexif)

    ls = str(path).split("/")
    ls[-1] = A

    return Path("/".join(ls))


def transform_annotations(
    path_source: Path,
    save: bool,
    subset: List = None,
    transformation_func: List[Callable] = None,
    transformation_func_params: List[Dict] = None,
    path_dest: Path = None,
    file_prefix: str = None,
    sep: str = None,
    plot: bool = False,
    plot_object: plt.axes = None,
    plot_funct: Callable = plot_function,
    plot_labels: Dict = None,
) -> Union[None, plt.axes]:
    """Transform and process annotations from folder according to selected parameters."""
    # load annotations

    # Hardcoding simple workaround:
    if subset == "None":
        subset = None

    for ann_idx, ann in enumerate(os.listdir(path_source)):
        if subset is not None and ann.split("_")[0] not in subset:
            continue
        an = pd.read_csv(path_source / ann, header=None)
        an = np.array(an)

        # transform annotation
        if transformation_func is not None:
            an = map_funcs(an, transformation_func, transformation_func_params)

        # save annotations
        if save and path_dest is not None:
            dest = path_dest / ann
            dest.parent.mkdir(parents=True, exist_ok=True)
            if "source_data" in str(dest):
                dest = change_last_element(dest, file_prefix, sep)
            pd.DataFrame(an).to_csv(dest, index=False, header=False)

        # plot annotations
        if plot and plot_object is not None and plot_labels is not None:
            plot_object = plot_funct(an, ann_idx, plot_object, **plot_labels)

    return plot_object


# MAIN FUNCTION ================================================================
def run_registration_process(
    results_path,
    moving_images_path,
    moving_images_annotations_path,
    moving_images_pairing,
    fixed_images_path,
    fixed_images_annotations_path,
    fixed_images_pairing,
    channel,
    image_matching,
    affine_registration_method,
    elastic_registration_method,
    gpu_id,
    save_loss_csv,
    subset_moving_annotations,
    subset_fixed_annotations,
    landmarks,
):
    """Author: Martin Vagenknecht
    Function for registration of moving image on fixed image. It creates
    folder structure with registrated images. Returns nothing.

    Parameters:
        results_path : str or Path, required,
        moving_images_path : str or Path, required,
        moving_images_annotations_path : str or Path, required,
        moving_images_pairing : str or Path, required,
        fixed_images_path : str or Path, required,
        fixed_images_annotations_path : str or Path, required,
        fixed_images_pairing : str or Path, required,
        channel : dict, required,
            Channel for moving and fixed image.
            default = {"moving": 0, "fixed": 2}
        image_matching : list of dictionaries, required,
            Matching indices of moving and fixed pairing.csv files.
        affine_registration_method : str, required,
            Self explanatory,
            Default = "affine"
        elastic_registration_method : str, required,
            Self explanatory.
            Default = "diffeomorphic_bspline"
        gpu_id: int, required,
            Index which gpu core to use
            default = 0
        save_loss_csv: Bool
            Flag for saving values of loss function
        subset_moving_annotations: list, optional
            a subset of moving annotations
        subset_fixed_annotations: list, optional
            a subset of fixed annotations
        landmarks: bool, required
            additional regularization term for registration

    Returns :
        None
    """

    # Test lenghts of paths, there is internal MAX_PATH set to 256 chars in WIN10
    paths_df = pd.Series(
        [
            results_path,
            moving_images_path,
            moving_images_annotations_path,
            moving_images_pairing,
            fixed_images_path,
            fixed_images_annotations_path,
            fixed_images_pairing,
        ]
    )

    paths_df.apply(test_result_folder_lens)

    if th.cuda.is_available():
        device = th.device("cuda:{}".format(str(gpu_id)))
    else:
        device = th.device("cpu")

    # set the used data type
    dtype = th.float32
    # moving
    # annotation mapping table
    moving_pairing = pd.read_csv(
        Path(moving_images_pairing),
        index_col=0,
        na_values=[None],
        keep_default_na=False,
    )

    # fixed
    # annotation mapping table
    fixed_pairing = pd.read_csv(
        Path(fixed_images_pairing),
        index_col=0,
        na_values=[None],
        keep_default_na=False,
    )

    # placeholder for losses
    losses = []

    # RUN LOOP
    for idx in image_matching:
        f = (
            list(moving_pairing.loc[idx["moving"]])[0],
            list(fixed_pairing.loc[idx["fixed"]])[0],
        )

        f_name = (os.path.splitext(f[0])[0], os.path.splitext(f[1])[0])
        currently_annotated_side = fixed_images_annotations_path.split("/")[
            -1
        ].split("_")[-1]

        folder_name_results = "moving_{}__fixed_{}__{}".format(
            f_name[0], f_name[1], currently_annotated_side
        )

        # where to save moving to fixed affine registered images
        # where to save fixed to moving affine registered images
        path_affine = Path(results_path) / folder_name_results / "affine"

        # where to save moving to fixed elastic registered images
        # where to save fixed to moving elastic registered images
        path_elastic = Path(results_path) / folder_name_results / "elastic"

        # moving_path
        moving = str(Path(moving_images_path) / f[0])

        # fixed_path
        fixed = str(Path(fixed_images_path) / f[1])

        # Check for existence of the images
        if not os.path.exists(fixed):
            print(fixed)
            raise Exception("Fixed file do not exists!")

        if not os.path.exists(moving):
            print(moving)
            raise Exception("Moving file do not exists!")

        # get ofset values for image (coordinates of top left corner)
        ofset_moving = tuple(
            map(int, os.path.splitext(f[0])[0].split("xx")[1].split("_"))
        )

        ofset_fixed = tuple(
            map(int, os.path.splitext(f[1])[0].split("xx")[1].split("_"))
        )

        # COPY FILES ===========================================================
        # MOVING SOURCE DATA
        # copy moving image
        dest = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "images"
            / f[0]
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.open(moving).convert("RGB").save(dest)

        # copy moving annotation
        ann_folder = list(moving_pairing.loc[idx["moving"]])[1]

        # placeholder - default state
        moving_annotations_exist = False

        if ann_folder != "":

            transform_annotations_params = {
                "path_source": Path(moving_images_annotations_path)
                / ann_folder,
                "path_dest": Path(results_path)
                / folder_name_results
                / "source_data"
                / "moving"
                / "annotations",
                "subset": subset_moving_annotations,
                "transformation_func": [translation],
                "transformation_func_params": [{"ofset": ofset_moving}],
                "save": True,
            }

            transform_annotations(**transform_annotations_params)

            if (
                os.path.exists(transform_annotations_params["path_dest"])
                and len(os.listdir(transform_annotations_params["path_dest"]))
                > 0
            ):
                moving_annotations_exist = True

        # copy moving pairing
        mc_settings.copyfile(
            moving_images_pairing,
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "pairing.csv",
        )

        # FIXED SOURCE DATAFRAME
        target_size = Image.open(moving).size
        original_size = Image.open(fixed).size
        r = [x[1] / x[0] for x in zip(original_size, target_size)]
        # resize and copy fixed image
        dest = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "images"
            / f[1]
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.open(fixed).resize(target_size).save(dest)

        # resize and copy fixed annotation
        ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]

        # placeholder - default state
        fixed_annotations_exist = False

        # Check Dictionary keys for seperator in the fixed annotation source.
        if [i for i in subset_fixed_annotations.keys()] != [""]:
            sep = "_"
        else:
            sep = ""

        if ann_folder != "":
            for prefix in subset_fixed_annotations.keys():
                transform_annotations_params = {
                    "path_source": Path(fixed_images_annotations_path)
                    / ann_folder,
                    "path_dest": Path(results_path)
                    / folder_name_results
                    / "source_data"
                    / "fixed"
                    / "annotations",
                    "file_prefix": prefix,
                    "subset": subset_fixed_annotations[prefix],
                    "transformation_func": [translation, scaling],
                    "transformation_func_params": [
                        {"ofset": ofset_fixed},
                        {"scaling_factor": r},
                    ],
                    "save": True,
                    "sep": sep,
                }

                transform_annotations(**transform_annotations_params)

                if (
                    os.path.exists(transform_annotations_params["path_dest"])
                    and len(
                        os.listdir(transform_annotations_params["path_dest"])
                    )
                    > 0
                ):
                    fixed_annotations_exist = True

                    # copy fixed pairing
        mc_settings.copyfile(
            fixed_images_pairing,
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "pairing.csv",
        )

        # CHANGE PATHS TO IMAGES AND ANNOTATIONS BASED ON SOURCE DATA
        # moving_path
        moving = str(
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "images"
            / f[0]
        )

        # fixed_path
        fixed = str(
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "images"
            / f[1]
        )

        # moving annotations path
        source_data_moving_images_annotations_path = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "annotations"
        )

        # fixed annotations path
        source_data_fixed_images_annotations_path = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "annotations"
        )

        # check if image size matches, otherwise skip
        if not cv2.imread(fixed, -1).shape == cv2.imread(moving, -1).shape:
            print(
                "================================================================="
            )
            print((f, idx))
            print("failed")
            print("fixed: {}".format(cv2.imread(fixed, -1).shape))
            print("moving: {}".format(cv2.imread(moving, -1).shape))
            continue
        # print which images are currently evaluated
        print(
            "================================================================="
        )
        print((f, idx))

        #####################
        # AFFINE REGISTRATION
        # run affine registration
        print("Affine")
        affine_registration = getattr(
            annobrainer_pipeline.affine_registration, affine_registration_method
        )

        displacement, A, loss_affine = affine_registration(
            fixed, moving, channel, device, dtype, plot=False
        )

        # warped moving image
        warped_image = np.array(Image.open(moving))
        warped_image = warp_rgb_image(
            warped_image, displacement, dtype, device
        )

        warped = path_affine / f[0]
        warped.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(warped_image).save(warped)

        # moving
        moving_image = np.array(Image.open(moving))

        # fixed
        fixed_image = np.array(Image.open(fixed))

        # make gifs
        # affine vs fixed
        im = Image.fromarray(warped_image)
        im2 = Image.fromarray(fixed_image)

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "warped(a)_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)

        # moving vs fixed
        im = Image.fromarray(moving_image)
        im2 = Image.fromarray(fixed_image)

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "moving_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)

        # transform annotations
        if moving_annotations_exist:
            transform_annotations_params = {
                "path_source": Path(
                    source_data_moving_images_annotations_path
                ),
                "path_dest": path_affine / "annotations" / "moving_to_fixed",
                "transformation_func": [affine_transformation],
                "transformation_func_params": [{"A": A}],
                "save": True,
            }

            transform_annotations(**transform_annotations_params)

        ######################
        # ELASTIC REGISTRATION

        # moving_path changes to affine transform
        moving = str(path_affine / f[0])

        # placeholder for landmark points
        fixed_points = None
        moving_points = None

        if landmarks:
            content_path = str(moving)
            style_path = str(fixed)

            feature_matching_results_path = (
                Path(results_path) / folder_name_results / "source_data"
            )

            landmarks_estimation(
                content_path,
                style_path,
                gpu_ids=gpu_id,
                path_results=str(feature_matching_results_path),
            )

            # feature matching points
            fixed_points = str(
                feature_matching_results_path
                / "CleanedPts"
                / "correspondence_B.txt"
            )

            moving_points = str(
                feature_matching_results_path
                / "CleanedPts"
                / "correspondence_A.txt"
            )

        # run elastic registration
        print("Elastic")
        elastic_registration = getattr(
            annobrainer_pipeline.elastic_registration, elastic_registration_method
        )

        (
            displacement,
            displacement_full_coords,
            inverse_displacement,
            shape,
            loss_elastic,
        ) = elastic_registration(
            fixed,
            moving,
            landmarks,
            fixed_points,
            moving_points,
            channel,
            device,
            dtype,
            plot=False,
        )

        # save loses
        losses.append(
            {
                "moving_pairing_id": idx["moving"],
                "moving": f[0],
                "fixed_pairing_id": idx["fixed"],
                "fixed": f[1],
                "loss": loss_elastic,
            }
        )

        # transform annotations
        if moving_annotations_exist:
            transform_annotations_params = {
                "path_source": path_affine / "annotations" / "moving_to_fixed",
                "path_dest": path_elastic / "annotations" / "moving_to_fixed",
                "transformation_func": [elastic_transformation],
                "transformation_func_params": [
                    {"displacement": inverse_displacement, "shape": shape}
                ],
                "save": True,
            }

            transform_annotations(**transform_annotations_params)

        # warped moving image (moving = moving original + warped affine)
        warped_image = np.array(Image.open(moving))
        warped_image = warp_rgb_image(
            warped_image, displacement, dtype, device
        )
        warped = path_elastic / f[0]
        warped.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(warped_image).save(warped)

        # square grid
        square_grid_image = create_square_grid(
            warped_image.shape[0], warped_image.shape[1]
        )

        square_grid_image = warp_rgb_image(
            square_grid_image, displacement, dtype, device
        )

        im = Image.fromarray(square_grid_image)

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "square_grid_warped(a+e)"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest)

        # moving
        moving_image = np.array(Image.open(moving))

        # fixed
        fixed_image = np.array(Image.open(fixed))

        # make gifs
        # affine + elastic vs fixed
        im = Image.fromarray(warped_image)
        im2 = Image.fromarray(fixed_image)

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "warped(a+e)_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)

        # affine vs elastic + affine
        im = Image.fromarray(moving_image)
        im2 = Image.fromarray(warped_image)

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "warped(a)_vs_warped(a+e)"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)

        # transform annotations backward fixed to moving
        if fixed_annotations_exist:

            # elastic fixed to moving
            transform_annotations_params = {
                "path_source": Path(source_data_fixed_images_annotations_path),
                "path_dest": path_elastic / "annotations" / "fixed_to_moving",
                "transformation_func": [elastic_transformation],
                "transformation_func_params": [
                    {"displacement": displacement_full_coords, "shape": shape}
                ],
                "save": True,
            }

            transform_annotations(**transform_annotations_params)

            # affine fixed to moving
            transform_annotations_params = {
                "path_source": path_elastic
                / "annotations"
                / "fixed_to_moving",
                "path_dest": path_affine / "annotations" / "fixed_to_moving",
                "transformation_func": [affine_transformation],
                "transformation_func_params": [{"A": A, "inverse": True}],
                "save": True,
            }

            transform_annotations(**transform_annotations_params)

        #################################################
        # plot annotations MOVING -> FIXED ON FIXED IMAGE
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title("moving_{}__fixed_{}".format(f_name[0], f_name[1]))
        ax.imshow(fixed_image, cmap="gray")

        # plot original
        if fixed_annotations_exist:
            transform_annotations_params = {
                "path_source": Path(source_data_fixed_images_annotations_path),
                "save": False,
                "plot": True,
                "plot_object": ax,
                "plot_labels": {"label": "Annotation fixed", "color": "green"},
            }

            ax = transform_annotations(**transform_annotations_params)

        if moving_annotations_exist:
            # plot affine
            transform_annotations_params = {
                "path_source": path_affine / "annotations" / "moving_to_fixed",
                "save": False,
                "plot": True,
                "plot_object": ax,
                "plot_labels": {
                    "label": "Annotation moving + affine",
                    "color": "red",
                },
            }

            ax = transform_annotations(**transform_annotations_params)

            # plot elastic
            transform_annotations_params = {
                "path_source": path_elastic
                / "annotations"
                / "moving_to_fixed",
                "save": False,
                "plot": True,
                "plot_object": ax,
                "plot_labels": {
                    "label": "Annotation moving + affine + elastic",
                    "color": "orange",
                },
            }

            ax = transform_annotations(**transform_annotations_params)

        # print the plot
        ax.legend()

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "annotations_moving_to_fixed_transfer"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".png")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(dest),
            bbox_inches="tight",
        )

        plt.close(fig)

        ##################################################
        # plot annotations FIXED -> MOVING ON MOVING IMAGE
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title("moving_{}__fixed_{}".format(f_name[0], f_name[1]))
        moving = str(Path(moving_images_path) / f[0])
        moving_image = np.array(Image.open(moving))
        ax.imshow(moving_image, cmap="gray")

        # plot original
        if moving_annotations_exist:
            transform_annotations_params = {
                "path_source": Path(
                    source_data_moving_images_annotations_path
                ),
                "save": False,
                "plot": True,
                "plot_object": ax,
                "plot_labels": {
                    "label": "Annotation moving",
                    "color": "green",
                },
            }

            ax = transform_annotations(**transform_annotations_params)

        # plot affine
        if fixed_annotations_exist:
            transform_annotations_params = {
                "path_source": path_affine / "annotations" / "fixed_to_moving",
                "save": False,
                "plot": True,
                "plot_object": ax,
                "plot_labels": {
                    "label": "Annotation fixed - elastic - affine",
                    "color": "red",
                },
            }

            ax = transform_annotations(**transform_annotations_params)

        # print the plot
        ax.legend()

        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "annotations_fixed_to_moving_transfer"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".png")
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(dest),
            bbox_inches="tight",
        )

        plt.close(fig)

        ####################
        # save losses to csv
        if save_loss_csv:
            pd.DataFrame(losses).to_csv(
                Path(results_path) / "losses.csv", index=False
            )

        print("Registration done")

    return None

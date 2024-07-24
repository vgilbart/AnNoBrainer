# DEPENDENCIES =================================================================
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

# AUX FUNCTIONS =================================================================


def small_grid_transform(small_grid, brains_grid):
    """This function transformes animal_id_grid."""
    transformed_grid = []

    X1 = (small_grid[:, 0] - np.mean(small_grid[:, 0])) / np.std(
        small_grid[:, 0]
    )
    X = (X1 * np.std(brains_grid[:, 0])) + np.mean(brains_grid[:, 0])

    Y1 = (small_grid[:, 1] - np.mean(small_grid[:, 1])) / np.std(
        small_grid[:, 1]
    )
    Y = (Y1 * np.std(brains_grid[:, 1])) + np.mean(brains_grid[:, 1])

    for i in range(0, len(small_grid)):
        transformed_grid.append([X[i], Y[i]])

    return np.array(transformed_grid)


def transform_grid_wrapper(
    animal_id_grid, brains_grid, path2animal_id_grid_tr
):
    """Wrapper function to transform and match animal_id_grid to brains_grid."""

    # transform grid
    trans_grid = small_grid_transform(animal_id_grid, brains_grid)
    # create matrix of weights defined by eukleidean distance
    weight_matrix = eukleid_dist(trans_grid, brains_grid)
    # calculate matching https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    cost = weight_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    # order rows in trans_grid according to matching
    trans_grid = trans_grid[col_ind]
    # save transformed grid to csv
    path2animal_id_grid_tr.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trans_grid, columns=["x", "y"]).to_csv(
        Path(path2animal_id_grid_tr), index=False
    )

    return (row_ind, col_ind)


def create_small_grid(config, path2animal_id_grid):
    """Create animal_id_grid from config info."""

    # create small grid
    path_excel = config["init_info"]["animal_id_mapping_table"]
    df = pd.read_csv(path_excel)
    scanner_id_col = df.columns[1]
    slide_id_col = df.columns[2]

    # get slide name from confing
    slide_name = config["init_info"]["slide_number"]

    # Check if slide name is scanner_id or slide_id
    if (
        df[scanner_id_col].isin([int(slide_name)]).any() or df[slide_id_col].isin([int(slide_name)]).any()
    ):
        if df[slide_id_col].isin([int(slide_name)]).any():
            column_to_use = slide_id_col
        else:
            column_to_use = scanner_id_col
    else:
        raise ValueError("Slide number or scanner id is not in pairing table!")

    df_subset = df[df[column_to_use] == int(slide_name)]
    animal_id_arr = np.array(df_subset[df.columns[4:]])
    indices = np.argwhere(animal_id_arr.astype(str) != "blank")
    # anima ids
    values = [animal_id_arr[x[0]][x[1]] for x in indices]
    indices = indices.transpose()
    small_grid = {"x": indices[1], "y": indices[0], "animal_id": values}
    path2animal_id_grid.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(small_grid).to_csv(str(path2animal_id_grid), index=False)


def eukleid_dist(trans_grid, brains_grid):
    """This function creates a matrix of weights given by euklidean distance."""
    trans_grid = np.array(trans_grid)
    brains_grid = np.array(brains_grid)

    dist_matrix = np.empty((brains_grid.shape[0], trans_grid.shape[0]))

    for i in range(0, len(brains_grid)):
        for j in range(0, len(trans_grid)):
            # HERE SHOULD BE EUKLEIDEAN DISTANCE SQRT((X-X')^2+(Y-Y')^2)
            dist_matrix[i, j] = np.sqrt(
                np.sum((brains_grid[i] - trans_grid[j]) ** 2)
            )
    return dist_matrix


def prepare_diagnostic_image(
    path_images_for_cutting,
    path2brains_grid,
    path2animal_id_grid_tr,
    path2folder,
):
    """Prepare diagnostic image."""
    files_in_folder = [
        file
        for file in os.listdir(path_images_for_cutting)
        if len(file.split(".")) > 1 and file.split(".")[1] in ["tif", "png"]
    ]
    image_name = files_in_folder[0]

    im_path = path_images_for_cutting / image_name
    im = np.array(Image.open(im_path))
    plt.imshow(im)

    # detected brains
    brains_grid = pd.read_csv(path2brains_grid, header=0)
    df_orig = pd.DataFrame(brains_grid, columns=["x", "y"])
    color = np.random.rand(len(df_orig["x"]), 3)
    plt.scatter(x=list(df_orig["x"]), y=list(df_orig["y"]), c=color, s=40)

    # animal_id grid
    trans_grid = pd.read_csv(path2animal_id_grid_tr, header=0)
    df_tr = pd.DataFrame(trans_grid, columns=["x", "y"])
    plt.scatter(x=list(df_tr["x"]), y=list(df_tr["y"]), c=color, s=40)

    plt.title(image_name)
    dest = path2folder / "{}_matched_points_visual_check.png".format(
        image_name.split(".")[0]
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(dest))
    plt.close()


def cut_brain_to_animalid_mapping(config):
    """Author: Martin Vagenknecht
    Function for mapping of brain image to correct position on the
    experiment layer. Default version before we create any algorithm is
    numbers from 1 to X.

    Parameters:
        config : json-like dictionary

    Returns:
        image_brain_mapping_dict : dictionary of dictionaries
            dict of dictionaries containing moving image name and its
            animal_id index.

    Example:
        {"r41b 22xx-11_2657": {
            "animal_id": "0B"
            },
        "r41b 22xx-29_1298": {
            "animal_id": "1B"
            }
        }
    """
    path_images_for_cutting = Path(config["brains_cutting"]["base_path_input"])
    path2folder = Path(config["brains_cutting"]["base_path_output"])
    path2brains_grid = path2folder / "row_column.csv"
    path2animal_id_grid = path2folder / "animal_id_grid.csv"
    path2animal_id_grid_tr = path2folder / "animal_id_grid_trans.csv"
    path2brain_animal_id_mapping = path2folder / "brain_animal_id_mapping.csv"

    create_small_grid(config, path2animal_id_grid)

    # !!!! only read x, y values
    brains_grid = np.array(
        pd.read_csv(
            path2brains_grid, header=0, usecols=["x", "y"]
        ).values.tolist()
    )

    animal_id_grid = np.array(
        pd.read_csv(
            path2animal_id_grid, header=0, usecols=["x", "y"]
        ).values.tolist()
    )

    row_ind, col_ind = transform_grid_wrapper(
        animal_id_grid, brains_grid, path2animal_id_grid_tr
    )

    # plot results for diagnostic purposes
    prepare_diagnostic_image(
        path_images_for_cutting,
        path2brains_grid,
        path2animal_id_grid_tr,
        path2folder,
    )

    # put detected brains and animal id together
    # read again orignal brains grid
    brains_grid_full = pd.read_csv(path2brains_grid, header=0)
    brains_grid_full["animal_id"] = "NA"

    brains_grid = brains_grid[row_ind]

    # read again original small grid
    animal_id_grid_full = pd.read_csv(path2animal_id_grid, header=0)
    for i in range(brains_grid.shape[0]):
        condition = brains_grid[i]
        animal_id_idx = col_ind[i]
        brains_grid_full.loc[
            np.logical_and(
                brains_grid_full.x == condition[0],
                brains_grid_full.y == condition[1],
            ),
            "animal_id",
        ] = animal_id_grid_full.animal_id[animal_id_idx]

    brain_animal_id_mapping = brains_grid_full[["name", "animal_id"]]
    path2brain_animal_id_mapping.parent.mkdir(parents=True, exist_ok=True)
    brain_animal_id_mapping.to_csv(path2brain_animal_id_mapping, index=False)

    experiment_folder_path = Path(config["brains_cutting"]["base_path_output"])

    folder_images_list = os.listdir(experiment_folder_path / "images")

    image_brain_mapping_dict = {
        key.split(".")[0]: {
            "animal_id": str(
                    list(brain_animal_id_mapping[
                        brain_animal_id_mapping.name == key
                    ]["animal_id"])[0]
            )
        }
        for key_index, key in enumerate(folder_images_list)
    }

    return image_brain_mapping_dict

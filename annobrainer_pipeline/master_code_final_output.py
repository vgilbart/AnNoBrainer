# DEPENDENCIES =================================================================

import csv
import os
from pathlib import Path
import json

import pandas as pd

# ENVIROMENT ===================================================================

# AUX FUNCTIONS ================================================================


def process_csv_into_halo_format(
    csv_file_path, config, animal_id, output_folder, ofset, region_prefix
):
    """Author: Michael Mateju
    Aux function for final structure creation
    """

    with open(csv_file_path["path"], "r") as csv_file:
        csv_file_df = pd.DataFrame(csv.reader(csv_file), columns=["X", "Y"])
        csv_file_df = csv_file_df.astype(int)
        # transform local coordinates to global coordinates by adding ofset
        csv_file_df.X += ofset[0]
        csv_file_df.Y += ofset[1]

    annotation_name = (
        config["init_info"]["study_id"]
        + "_"
        + config["init_info"]["stain"]
        + "_"
        + animal_id
        + "_"
        + str(region_prefix)
        + "_"
        + config["init_info"]["slide_number"]
        + "_"
        + str(csv_file_path.name)
        # Extracts side to be annotated (L or R) from the path
        + "_"
        + str(csv_file_path["path"]).split("/")[-5].split("__")[-1]
    )

    csv_file_df[
        "X3_annotation_index"
    ] = 0  # TODO: must be unique for multiple annotations
    csv_file_df[
        "X4_positive_negative"
    ] = 0  # TODO: should be loaded or recognized from file name
    csv_file_df[
        "X5_name_layer"
    ] = annotation_name  # layer name in form: R41b_PS129_548B_STR_23
    csv_file_df[
        "X6_color"
    ] = 65280  # TODO: to be confirmed with Renee / Josef Navratil
    csv_file_df[
        "X7_visibility"
    ] = 1  # TODO: to be confirmed with Renee / Josef Navratil

    # save the csv file
    pd.DataFrame(csv_file_df).to_csv(
        Path(output_folder) / (annotation_name + ".csv"),
        index=False,
        header=False,
    )

    return None

def save_current_config(config_file):
    """
    Author: Petr Hrobar

    This Function is suppose to export the current config file as a json.
    Small function that is created for the purpose of having cleaned code.

    Returns Nothing

    Args:
        config_file (json): current config file with all defined parameters
    """

    save_config_path = str(
        Path(config_file["init_info"]["result_output_folder"]) / "config.json"
    )

    with open(save_config_path, "w") as intermediate_config:
        json.dump(config_file, intermediate_config, indent=2)

    print(f"Config File Save to the following path: {save_config_path}")




def process_config_and_folders(
    folder, config, mapping_brains_animal_id, output_folder
):
    """Author: Michael Mateju
    Aux function for final structure creation
    """

    moving_image_name = folder.split("__")[0].split("moving_")[1]
    ofset = moving_image_name.split("xx")[1].split("_")
    ofset = list(map(int, ofset))

    if moving_image_name not in mapping_brains_animal_id:
        raise Exception(
            "For image: "
            + moving_image_name
            + ' there is not "animal_id" in mapping table'
        )
    else:
        animal_id = mapping_brains_animal_id[moving_image_name]["animal_id"]

    # TODO: temporary solution. Not fully intuitive, will be changed in future.
    # Martin saves the reverse annotation of: fixed -> elastic^{-1} -> affine^{-1} to affine folder.
    # It makes sense that it is the final inverse transformation done, but it would be rather clearer
    # if user just assumes that first step (affine transformation) contains only inverse affine
    # transformation) and the second step (elastic transformation) contains both inverse affine + elastic.
    csv_files_folder = (
        Path(config["registration"]["results_path"])
        / folder
        / "affine"
        / "annotations"
        / "fixed_to_moving"
    )

    csv_files = [
        Path(str(csv_files_folder) + "/" + file)
        for file in os.listdir(csv_files_folder)
    ]

    csv_files_df = pd.DataFrame(csv_files, columns=["path"])

    for region in config["init_info"]["region"]:

        csv_files_df_filtered = csv_files_df.loc[
            lambda d: d["path"]
            .astype(str)
            .str.split("/")
            .str[-1]
            .str.startswith(region)
        ]

        # Iterate over Each Region in the config file and use it within the file name
        csv_files_df_filtered.apply(
            process_csv_into_halo_format,
            axis=1,
            config=config,
            animal_id=animal_id,
            output_folder=output_folder,
            ofset=ofset,
            region_prefix=region,
        )

    return None


# MAIN FUNCTION ================================================================
def final_structure_creation(config):
    """Author: Michael Mateju
    Function with takes annotation process after registration and creates
    final structure of folders, changes the file to requested format and
    saves the csv files under requested name. Returns json-like dictionary.

    Parameters:
       config : json-like dictionary

    Returns:
        json-like dictionary
        example: {"output_paths" : str(output_folder)}
    """

    # create final structure folder
    output_folder = (
        Path(config["registration"]["results_path"])
        / config["init_info"]["slide_number"]
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    results_content = pd.Series(
        os.listdir(Path(config["registration"]["results_path"]))
    )

    results_folder = results_content[
        results_content.map(lambda f: len(f.split(".")) == 1 and "moving" in f)
    ]

    results_folder.apply(
        process_config_and_folders,
        config=config,
        mapping_brains_animal_id=config["image_brain_table"],
        output_folder=output_folder,
    )

    print("Transformation done")

    return {"output_paths": str(output_folder)}

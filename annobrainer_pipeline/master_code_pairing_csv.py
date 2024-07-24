# DEPENDENCIES =================================================================

import os
from pathlib import Path

import pandas as pd


# TAKE CARE OF FOLDERS WITHOUT ANNOTAION FILE ==================================
def create_pairing_file(config):
    """ Author: Michael Mateju
        Function which will create file called \"pairing.csv\" in experiment
        folder. It will contain structure: \"idx\", \"img\", \"anno\". This
        file is later used in registration. In the future, it might be droped.

        Parameters:
            config : json-type dictionary
                File providing necessary paths to images.

        Returns:
            str - Path to pairing.csv file

    """

    experiment_folder_path = Path(config["brains_cutting"]["base_path_output"])

    folder_images_list = os.listdir(experiment_folder_path / "images")

    images_folder_list_pairing_images_df = pd.DataFrame(
        {
            "idx": range(1, len(folder_images_list) + 1),
            "img": sorted(folder_images_list),
            "anno": [""] * len(folder_images_list),
        }
    )

    pairing_file_path = experiment_folder_path / "pairing.csv"
    images_folder_list_pairing_images_df.to_csv(pairing_file_path, index=False)

    print(
        'File "pairing.csv" was created. It has: '
        + str(len(images_folder_list_pairing_images_df))
        + " rows."
    )

    return str(pairing_file_path)

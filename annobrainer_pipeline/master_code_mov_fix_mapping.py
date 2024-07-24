# DEPENDENCIES =================================================================

import itertools
from pathlib import Path

import pandas as pd


def pairing_of_moving_to_fixed_images(config):
    """Author: Michael Mateju
    Function which creates dictionary of images which should be registrated
    against each other.

    Parameters:
       config : json-like dictionary

    Returns:
       image_brain_mapping_dict : list of dictionaries
           list of dictionaries containing moving index and fixed index.

    Example:
       [{"moving": 1, "fixed": 2}, {"moving": 2, "fixed": 2}]
    """

    # OPEN MOVING PAIRING FILE
    with open(
        Path(config["brains_cutting"]["base_path_output"]) / "pairing.csv", "r"
    ) as mov_pairing:
        mov_pairing_df = pd.read_csv(mov_pairing)

    # OPEN FIXED PAIRING FILE
    fixed_idx = [int(x) for x in [config["init_info"]["layer_atlas"]]]

    image_matching = [
        {"moving": x[0], "fixed": x[1]}
        for x in itertools.product(mov_pairing_df["idx"], fixed_idx)
    ]

    return image_matching

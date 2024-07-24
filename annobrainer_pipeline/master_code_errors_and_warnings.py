# DEPENDENCIES =================================================================
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from itertools import chain

import numpy as np
import pandas as pd
import pydantic
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    PositiveInt,
    StrictBool,
    ValidationError,
    root_validator,
    validator,
)
from pydantic.typing import Literal


def convert_dict_values_to_list(d):
    """
    Author: Petr Hrobar

    This function accepts dictionary and return all values except the NONE value in one list

    Args:
        d (dict): dictionary of subsset_fixed_annotations from config file

    Returns:
        list: list of all values.
    """
    ls = [i for i in d.values() if i != "None"]

    list_of_annotations = list(chain(*ls))

    return list_of_annotations


def machine_specific_default_path():
    return Path(f"{os.getenv['HOME']}"/'annoBrainer')


# HIGH: SET UP LOGGING =========================================================
def set_up_logging(paths_dict: dict):
    try:
        LOG_FILE_PATH = paths_dict["log_path"]
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(LOG_FILE_PATH),
                logging.StreamHandler(),
            ],
            format="%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    except:
        LOG_FILE_PATH = machine_specific_default_path() / "logfile.log"
        print("=========================")
        print('logging: no valid path provided in paths_dict["log_path"]')
        print("logging path set to default: {}".format(LOG_FILE_PATH))
        print("=========================")
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(LOG_FILE_PATH),
                logging.StreamHandler(),
            ],
            format="%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


logger = logging.getLogger(__name__)


# HIGH: PYDANTIC VALIDATION TEMPLATES ==========================================
class PathsDict(BaseModel):
    experiment_folder: DirectoryPath
    config_path: FilePath
    animal_id_path: FilePath
    atlas_path: DirectoryPath
    path_model: FilePath
    result_output_folder: Union[None, DirectoryPath]
    gpu_id: int

    @validator("experiment_folder")
    def only_one_file_of_this_type(cls, v):
        allowed_file_type_dict = {
            "config": [".json"],
            "animal_id": [".csv"],
            "image": [".png", ".tif"],
        }
        for file_type in allowed_file_type_dict.keys():
            type_list = allowed_file_type_dict[file_type]
            folder = os.listdir(v)
            files_in_folder = [
                Path(x).suffix for x in folder if Path(x).suffix != ""
            ]
            files_of_this_type = [x for x in files_in_folder if x in type_list]
            if len(files_of_this_type) != 1:
                raise ValueError("there is more than one {}".format(file_type))
        return v


class InitInfo(BaseModel):
    study_id: str
    stain: str
    region: List
    slide_number: Union[str, int]
    layer_atlas: PositiveInt
    subset_moving_annotations: Union[str, List[str]]
    subset_fixed_annotations: Dict
    landmarks: StrictBool
    animal_id_mapping_table: FilePath
    path_model: FilePath
    atlas_path: DirectoryPath
    result_output_folder: Union[None, DirectoryPath]
    gpu_id: int
    side_to_annotate: List[Literal["L", "R"]]
    part_of_code_to_run: List[Literal[1, 2]]
    zeiss_gray_background: StrictBool


class BrainsCutting(BaseModel):
    base_path_input: DirectoryPath
    base_path_output: Union[None, DirectoryPath]
    path_model: FilePath
    gpu_id: int


class Registration(BaseModel):
    results_path: DirectoryPath
    moving_images_path: DirectoryPath
    moving_images_annotations_path: Union[str, Path]
    moving_images_pairing: FilePath
    fixed_images_path: DirectoryPath
    fixed_images_annotations_path: DirectoryPath
    fixed_images_pairing: FilePath
    channel: Dict
    image_matching: List[Dict]
    affine_registration_method: str
    elastic_registration_method: str
    gpu_id: int
    save_loss_csv: bool
    subset_moving_annotations: Union[None, List[str]]
    subset_fixed_annotations: Dict
    landmarks: bool


class ConfigInitInfo(BaseModel):
    init_info: InitInfo


class ConfigBrainsCutting(BaseModel):
    init_info: InitInfo
    brains_cutting: BrainsCutting


class ConfigRegistration(BaseModel):
    init_info: InitInfo
    brains_cutting: BrainsCutting
    registration: Registration

    @validator("registration")
    def invalid_atlas_layer(cls, v, values):
        # values - a dict containing the name-to-value mapping of any previously-validated fields
        if values["init_info"].layer_atlas > len(
            os.listdir(v.fixed_images_path)
        ):
            raise ValueError("atlas layer out of range")

        return v

    @validator("registration")
    def annotation_not_present_in_selected_layer(cls, v, values):
        print(f"INPUTDATA: {cls} {v} {values}")
        v.subset_fixed_annotations = convert_dict_values_to_list(
            v.subset_fixed_annotations
        )
        if v.subset_fixed_annotations not in ["None", None]:
            atlas_layer = values["init_info"].layer_atlas
            pairing = pd.read_csv(v.fixed_images_pairing, index_col=0)
            annotations_folder = pairing.loc[atlas_layer].anno
            annotations_folder_path = (
                Path(v.fixed_images_annotations_path) / annotations_folder
            )
            annotations_list = os.listdir(annotations_folder_path)
            annotations_names_list = [
                x.split("_")[0] for x in annotations_list
            ]

            intersection_of_annotations = set(
                annotations_names_list
            ).intersection(v.subset_fixed_annotations)
            if intersection_of_annotations != set(v.subset_fixed_annotations):
                annotations_missing_set = (
                    set(v.subset_fixed_annotations)
                    - intersection_of_annotations
                )

                raise ValueError(
                    "these annotations are not present in selected layer: {}".format(
                        annotations_missing_set
                    )
                )

        return v

    @validator("registration")
    def annotation_check(cls, v, values):
        atlas_layer = values["init_info"].layer_atlas
        pairing = pd.read_csv(v.fixed_images_pairing, index_col=0)
        annotations_folder = pairing.loc[atlas_layer].anno
        annotations_folder_path = (
            Path(v.fixed_images_annotations_path) / annotations_folder
        )

        empty_annotations_dict = {}
        for ann in os.listdir(annotations_folder_path):
            if (
                v.subset_fixed_annotations is not None
                and ann.split("_")[0] not in v.subset_fixed_annotations
            ):
                continue

            an = pd.read_csv(annotations_folder_path / ann)

            n_points = an.shape[0]
            if n_points == 0:
                empty_annotations_dict[ann] = n_points

        if len(empty_annotations_dict) > 0:
            raise ValueError(
                "contact developers, these annotations are empty: {}".format(
                    empty_annotations_dict
                )
            )

        return v


class AnimalID(BaseModel):
    region: List[str]
    scanner_id: Union[List[str], List[int]]
    slide_id: Union[List[str], List[int]]
    row_col: List[PositiveInt]
    slide_number: Optional[Union[str, int]]

    @validator("region", "scanner_id", "slide_id")
    def unique(cls, v):
        if len(set(v)) != 1:
            raise ValueError("values in column must be unique")
        return v

    @validator("row_col")
    def is_sorted(cls, v):
        if v != sorted(v):
            raise ValueError("values in column must be ordered")
        return v

    @root_validator
    def is_slide_number_in_table(cls, values):
        scanner_id, slide_id = (
            values.get("scanner_id")[0],
            values.get("slide_id")[0],
        )
        slide_number = str(values.get("slide_number"))
        if str(scanner_id) != slide_number and str(slide_id) != slide_number:
            raise ValueError(
                "slide_number from config not found in animal_id mapping table"
            )
        return values


# HIGH: WRAPPER FUNCTIONS FOR VALIDATION ==========================================
def validate(config: Dict, template: pydantic.main.ModelMetaclass):
    """Validate config dictionary to match given template."""
    try:
        template(**config)
    except ValidationError as e:
        logger.error(e)
        print("CRITICAL ERROR: INPUT INVALID !")
        sys.exit(1)


def validate_animal_id_mapping_table(
    table_path: Union[str, Path], slide_number: Union[str, int], template
):
    """Validate animal id mapping table, wrapper for validate function."""
    config = pd.read_csv(table_path).to_dict(orient="list")
    # test first 4 columns
    col_order = ["region", "scanner_id", "slide_id", "row_col"]
    if list(config.keys())[0:4] != col_order:
        logger.warning(
            "validate_animal_id_mapping_table: first 4 columns of animal_id_mapping_table should be in this order: {}".format(
                col_order
            )
        )
    # rest of the columns 4: are numeric
    try:
        map(int, list(config.keys())[4:])
    except ValueError:
        logger.warning(
            "validate_animal_id_mapping_table: columns 4 and forward are not numeric"
        )
    # rest of the columns is ordered
    numeric_columns = list(map(int, list(config.keys())[4:]))
    if numeric_columns != sorted(numeric_columns):
        logger.warning(
            "validate_animal_id_mapping_table: numeric columns must be ordered, missmatch possible"
        )
    # test brains ids are unique except blank
    df_unique_ids = pd.DataFrame(config).iloc[:, 4:]
    flatted = df_unique_ids.values.flatten()
    flatted = [x for x in flatted if x != "blank"]
    n_length = len(flatted)
    n_length_unique = len(np.unique(flatted))

    if n_length_unique < n_length:
        brains_ids_unique = False
    else:
        brains_ids_unique = True

    if not brains_ids_unique:
        logger.warning(
            "validate_animal_id_mapping_table: animal id values must be unique"
        )

    # add slide_number to config
    config["slide_number"] = slide_number
    # run normal validator on dictionary
    validate(config, template)


def validate_compatibility_detected_brains_and_animal_id(
    path_detected_brains: Union[str, Path], path_animal_id: Union[str, Path]
):
    """Validate if # detected brains differs from the animal_id template."""
    brains = pd.read_csv(path_detected_brains)
    animal_id = pd.read_csv(path_animal_id)
    # number of rows must be same in both csv
    if brains.shape[0] != animal_id.shape[0]:
        logger.warning(
            "number of detected brains ({}) differs from the animal_id template ({})".format(
                brains.shape[0], animal_id.shape[0]
            )
        )

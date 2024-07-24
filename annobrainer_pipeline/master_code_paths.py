# DEPENDENCIES =================================================================

import os
from pathlib import Path
import sys
import subprocess
import json

sys.path.append(".")
from helpers import config_parser  # noqa E402


def setup_paths(
        user=None,
	gpu_id=None,
):
    """
    Author: Michael Mateju
    This function will set up paths for running the master code. There will be
    some default paths for debugging.

        :param user: str, optional
        :param experiment_folder: str, optional
        :param atlas_path: str, optional
        :param path_model: str, optional
        :param result_output_folder: str, optional
        :param gpu_id: int, optional

        :return: path_list = dict of paths for master code.
    """
    if not gpu_id:
        select_avail_gpu()
    gpu_id = 0
    config_vars = config_parser.config_params('config/path_config.ini')
    experiment_folder = str(config_vars['PATHS']['experiment_folder']).replace("${ANNOBRAINER_HOME}", f"{os.environ['ANNOBRAINER_HOME']}")
    atlas_path = config_vars['PATHS']['atlas_path'].replace("${ANNOBRAINER_HOME}", f"{os.environ['ANNOBRAINER_HOME']}")
    path_model = config_vars['PATHS']['path_model'].replace("${ANNOBRAINER_HOME}", f"{os.environ['ANNOBRAINER_HOME']}")
    log_path = config_vars['PATHS']['log_path'].replace("${ANNOBRAINER_HOME}", f"{os.environ['ANNOBRAINER_HOME']}")
    config_file_path = str(Path(experiment_folder) / "config_init.json")
    animal_id_file_path = str(Path(experiment_folder) / "animal_id.csv")
    with open(config_file_path, "r") as json_file_ptr:
        json_config = json.load(json_file_ptr)
    result_output_folder = None

    path_dict = {
        "experiment_folder": experiment_folder,
        "config_path": config_file_path,
        "animal_id_path": animal_id_file_path,
        "atlas_path": atlas_path,
        "path_model": path_model,
        "result_output_folder": result_output_folder,
        "gpu_id": gpu_id,
        "log_path": log_path,
    }

    return path_dict


def select_avail_gpu():
    try:
        process_free_info = ""
        for i in range(8):
            check_all_gpu_statuses = "nvidia-smi -i %s --query-compute-apps=pid --format=csv,noheader" % str(i)
            _output_to_list = lambda x: x.decode('ascii').split('\n')
            process_free_info = _output_to_list(subprocess.check_output(check_all_gpu_statuses.split()))[0]
            if process_free_info == '':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
                print("Using gpu ", str(i))
                return i
        if process_free_info != '':
            raise IOError("NO FREE GPU")
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs is not selected', e)
        raise IOError("Nvidia-smi is not installed.")

# DEPENDENCIES =================================================================
from pathlib import Path
import sys
sys.path.append(".")

import annobrainer_pipeline.master_code_annotation_parsing as mc_an_parse
import annobrainer_pipeline.master_code_brain_position_mapping as bpm
import annobrainer_pipeline.master_code_brains_cutting as mc_bcw
import annobrainer_pipeline.master_code_errors_and_warnings as mc_ew
import annobrainer_pipeline.master_code_final_output as mc_fout
import annobrainer_pipeline.master_code_mov_fix_mapping as mc_mov_fix_map
import annobrainer_pipeline.master_code_pairing_csv as mc_pair_csv
import annobrainer_pipeline.master_code_paths as mc_paths
import annobrainer_pipeline.master_code_registration as mc_reg
import annobrainer_pipeline.master_code_settings as mc_settings


def run_pipeline():
    # HIGH: SETTING ENVIROMENT =====================================================
    paths_dict = mc_paths.setup_paths(user=None)
    # logging
    mc_ew.set_up_logging(paths_dict)
    # validation
    mc_ew.validate(paths_dict, template=mc_ew.PathsDict)

    # HIGH: SETTING CONFIG =========================================================
    # Loading Initial Config file with default annotation process parameters
    config = mc_settings.load_init_config(
        experiment_folder=paths_dict["experiment_folder"],
        config_path=paths_dict["config_path"],
        path_model=paths_dict["path_model"],
        atlas_path=paths_dict["atlas_path"],
        animal_id_path=paths_dict["animal_id_path"],
        result_output_folder=paths_dict["result_output_folder"],
        gpu_id=paths_dict["gpu_id"],
    )
    if not "subset_moving_annotations" in config["init_info"].keys():
        config["init_info"]["subset_moving_annotations"] = "None"
    print(f"VALIDATING CONFIG {config}")
    # Validating config file whether is matches expected format.
    mc_ew.validate(config, template=mc_ew.ConfigInitInfo)
    # extract third layer from svs file and save it as png file
    #for img_file_name in \
    #        [imgfile for imgfile in os.listdir(paths_dict["experiment_folder"]) if imgfile.endswith(".svs")]:
    #prepare_image(paths_dict["experiment_folder"], img_file_name)


    # PART1 - REGISTRATION AND FINAL OUTPUT
    if 1 in config["init_info"]["part_of_code_to_run"]:
        # HIGH: BRAINS CUTTING =========================================================

        # Adding another information info a config file, parameters regarding the brain cutting process
        config["brains_cutting"] = mc_bcw.set_brains_cutting_parameters(
            config=config,
            base_path_input=paths_dict["experiment_folder"],
            base_path_output=config["init_info"]["result_output_folder"],
            path_model=config["init_info"]["path_model"],
            gpu_id=config["init_info"]["gpu_id"],
        )

        # Validating config file whether it matches the expected format (After adding brain cutting parameters)
        mc_ew.validate(config, template=mc_ew.ConfigBrainsCutting)

        # run brains cutting
        mc_bcw.brain_cutting_wrapper(**config["brains_cutting"])

        # HIGH: ANNOTATION PARSING - this will be in future release ====================
        mc_an_parse.parsing_of_annotation_file(config)

        # HIGH: CREATION OF CSV PAIRING.CSV FILE =======================================
        config["pairing_file"] = mc_pair_csv.create_pairing_file(config)
        print(f"CONFIG 0 {config}")
        # HIGH: MAP DETECTED BRAINS TO ANIMAL ID =======================================
        # Validating config file whether it matches the expected format
        mc_ew.validate_animal_id_mapping_table(
            table_path=config["init_info"]["animal_id_mapping_table"],
            slide_number=config["init_info"]["slide_number"],
            template=mc_ew.AnimalID,
        )
        print(f"CONFIG {config}")
        # Adding another information info a config file, parameters regarding the brain cutting process
        config["image_brain_table"] = bpm.cut_brain_to_animalid_mapping(config)
        print(f"CONFIG2 {config}")
        # Validating config file whether it matches the expected format added from the step above
        mc_ew.validate_compatibility_detected_brains_and_animal_id(
            config["pairing_file"],
            path_animal_id=Path(config["brains_cutting"]["base_path_output"]) / "animal_id_grid.csv",
        )
        print(f"MATCHINGCONFIG {config}")
        # HIGH: PREPARE IMAGE TO TEMPLATE MATCHING FOR REGISTRATION ====================
        config[
            "image_matching"
        ] = mc_mov_fix_map.pairing_of_moving_to_fixed_images(config)

        # HIGH: SAVE CONFIG ============================================================
        mc_fout.save_current_config(config)

    # Running 2 Part of the script where REGISTRATION AND FINAL OUTPUT creation is performed
    if 2 in config["init_info"]["part_of_code_to_run"]:

        if config["init_info"]["part_of_code_to_run"] != [1]:
            # Validating config file whether it matches the expected format for this part of the code
            mc_settings.validate_existing_config(config)

        # HIGH: REGISTRATION ===========================================================
        # Iterating Over the Sides to be Annotated
        for index, currently_annotated_side in enumerate(
                config["init_info"]["side_to_annotate"]
        ):
            # Adding another information info a config file, parameters regarding the brain registration and exports
            config["registration"] = mc_reg.set_registration_parameters(
                config=config,
                results_path=str(
                    Path(config["init_info"]["result_output_folder"])
                    / "registration_results"
                ),
                base_moving_folder=config["init_info"]["result_output_folder"],
                moving_images_pairing=config["pairing_file"],
                base_fixed_folder=config["init_info"]["atlas_path"],
                fixed_images_path=str(
                    Path(config["init_info"]["atlas_path"])
                    / "images_fixed_{}".format(currently_annotated_side)
                ),
                fixed_images_annotations_path=str(
                    Path(config["init_info"]["atlas_path"])
                    / "annotations_named_complete_csv_{}".format(
                        currently_annotated_side
                    )
                ),
                save_loss_csv=False,
                gpu_id=config["init_info"]["gpu_id"],
                increase_folder_version=[False, True][index],
                landmarks=config["init_info"]["landmarks"],
            )

            config["registration"]["image_matching"] = config["image_matching"]

            # Validating config file whether it matches the expected format for this part of the code
            mc_ew.validate(config, template=mc_ew.ConfigRegistration)

            # Performing The registration process
            mc_reg.run_registration_process(**config["registration"])

        # HIGH: FINALIZING CSVs ========================================================
        # Adding another information info a config file, parameters regarding the exports
        config["final_output"] = mc_fout.final_structure_creation(config)

        # Printing Current config File
        mc_settings.print_current_config_file(config)

        # Exporting all results of the registration process into structured folders.
        mc_settings.copy_registration_results_into_experiment_folder(
            config, paths_dict["experiment_folder"]
        )


if __name__ == "__main__":
    run_pipeline()

# LOAD DEPENDENCIES ============================================================
import os
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from skimage.measure import label
from skimage.morphology import square, dilation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import annobrainer_pipeline.master_code_settings as mc_settings

from collections import Counter

# ENVIROMENT ===================================================================

# Visualize bounding boxes
BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (0, 0, 0)

transform = transforms.Compose(
    [transforms.Resize((512, 512)), transforms.ToTensor()]
)


# AUX FUNCTIONS ================================================================
def set_brains_cutting_parameters(
    config, base_path_input, path_model, base_path_output=None, gpu_id=0, 
):
    """ Author: Valerii Tynianskaia
        Function sets parameters for brains cuting function.

           Parameters:
               config : json-like dictionary, required
               base_path_input : str or Path, required
                   Path where the image to process is saved.
               path_model: str or Path, required
               base_path_output : str or Path, required
                   Path where the processed images will be saved.
                   default = Path(os.environ["USERPROFILE"]) /
                                  "brains_cut" /
                                  str(date.today()) /
                                  "brains_cut" /
                                  results_experiment_name
               gpu_id : int, required
                   index which gpu core to use.
                   default = 0.

           Returns:
               config : json-like dictionary

       """

    if base_path_output is None:
        results_experiment_name = (
            config["init_info"]["study_id"]
            + "_"
            + config["init_info"]["stain"]
            + "_"
            + config["init_info"]["region"]
            + "_"
            + config["init_info"]["slide_number"]
            + "_"
            + config["init_info"]["layer_atlas"]
        )

        base_path_output = str(
            mc_settings.machine_specific_default_path()
            / "brains_cut"
            / str(date.today())
            / "brains_cut"
            / results_experiment_name
        )
        # create path
        Path(base_path_output).mkdir(parents=True, exist_ok=False)

    parameters = {
        "base_path_input": str(base_path_input),
        "base_path_output": str(base_path_output),
        "path_model": str(path_model),
        "gpu_id": gpu_id,
        "zeiss_gray_background": config['init_info']['zeiss_gray_background']
    }

    return parameters


def get_instance_segmentation_model(num_classes):
    """
        AUXILIARY FUNCTION FOR BRAINS CUTTING. Documentation will be later.
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False
    )
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def visualize_bbox(img, bbox, class_id, label, color=BOX_COLOR, thickness=2):
    """
           AUXILIARY FUNCTION FOR BRAINS CUTTING. Documentation will be later.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness
    )

    class_name = str(round(class_id, 2)) + "_" + str(label)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )

    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR,
        -1,
    )

    cv2.putText(
        img,
        class_name,
        (x_min, y_min - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )

    return img


def visualize(annotations, show: bool = True, box_color=(139, 0, 0)):
    """
           AUXILIARY FUNCTION FOR BRAINS CUTTING. Documentation will be later.
    """
    img = annotations["image"].copy()
    for idx, bbox in enumerate(annotations["bboxes"]):
        img = visualize_bbox(
            img,
            bbox,
            annotations["category_id"][idx],
            annotations["label"][idx],
            box_color,
        )
    if show:
        plt.figure(figsize=(9, 9))
        plt.imshow(img)
        pass
    return img


def standardize_gray_background(image):

    """Author: Petr Hrobar
    Function for adjusting gray background as a gray mirror glass can be present in some images

    Parameters:
        image : numpy array image in RGB color format

    Returns:
        image: numpy array image in RGB color format with adjusted background
    """

    # Lambda function for computing the most frequent value in the numpy array (can be changed)
    most_common = lambda x: Counter(x).most_common(1)[0][0]

    # Copy original Image and save changed background values to it

    image_corrected = image.copy()

    # Iterate Over RGB channels.
    for channel in range(np.shape(image)[2]):

        img = image[:, :, channel].copy()

        img_bw = img == 255
        img_bw = np.where(img_bw == True, 1, 0)

        # Dilatation of the mask (Check of this is outside or inner border of color.)
        img_dilatated = dilation(img_bw, square(3))

        # Substract the image from dilated image - get border region
        subtracted = img_dilatated - img_bw

        # Index of the border
        border_index = subtracted == 1

        # Label The BW Image
        labeled = label(img_bw)

        # Calculate most common value in region border
        try:
            f = most_common(img[border_index])
        except IndexError:
            return image

        # Some images can have 3 level labeles (Border of the background glass expends to the very edges of the image)

        if len(np.unique(labeled)) > 1:

            img[border_index] = f

            background_index = labeled != 0
        
            img[background_index] = f

            image_corrected[:, :, channel] = img

    return image_corrected

# MAIN FUNCTION ================================================================


def brain_cutting_wrapper(
    base_path_input, base_path_output, gpu_id, path_model, zeiss_gray_background
):
    """ Author: Valerii Tynianskaia
        Function which loads experiment image file and cuts it in single brains.
        It creates folder structure and saves cut images there. Returns nothing.
        All arguments should be created by function
        \"set_brains_cutting_parameters\".

        Parameters:
            base_path_input : str or Path, required
                Path where the image to process is saved.
            base_path_output : str or Path, required
                Path where the processed images will be saved.
                default = Path(os.environ["USERPROFILE"]) /
                               "brains_cut" /
                               str(date.today()) /
                               "brains_cut" /
                               results_experiment_name
            gpu_id : int, required
                index which gpu core to use.
                default = 0.
            path_numbers : str or Path, required

        Returns:
            None

    """

    ##########################################################################
    # SET INPUTS
    ###########################################################################
    path_images_for_cutting = Path(base_path_input)
    path_results_root = Path(base_path_output)
    # target image size (all images must have the same size for registration)

    # select objects with confidence threshold (0-1) greater than
    th = 0.8

    # set gpu id
    device = torch.device("cuda:{}".format(gpu_id))
    ###########################################################################
    # LOAD DETECTION MODEL/S
    ###########################################################################
    # new model to detect objects and numbers

    num_classes = 3
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))

    ###########################################################################
    # DETECTION LOOP
    ###########################################################################

    # get image files
    files_in_folder = [
        file
        for file in os.listdir(path_images_for_cutting)
        if len(file.split(".")) > 1 and file.split(".")[1] in ["tif", "png"]
    ]

    for single_image in files_in_folder:
        print("detection running: {}".format(single_image))
        path = path_images_for_cutting / single_image
        # path_results_single_image_root = Path(
        #     "{}/{}/".format(path_results_root, path.stem)
        # )
        path_results_single_image_root = Path(path_results_root)
        path_results_single_image_images = (
            path_results_single_image_root / "images"
        )

        # load image
        img = transform(Image.open(path).convert("RGB"))

        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        img_orig = Image.fromarray(
            img.mul(255).permute(1, 2, 0).byte().numpy()
        ).convert("L")

        annotations = {
            "image": np.array(img_orig),
            "bboxes": prediction[0]["boxes"][
                np.where(prediction[0]["scores"].cpu() > th)
            ].tolist(),
            "category_id": prediction[0]["scores"][
                np.where(prediction[0]["scores"].cpu() > th)
            ].tolist(),
            "label": prediction[0]["labels"][
                np.where(prediction[0]["scores"].cpu() > th)
            ].tolist(),
        }
        vis_img_prediction = visualize(annotations, False)

        selection = np.where(prediction[0]["scores"].cpu() > th)
        mask = torch.max(prediction[0]["masks"][selection], dim=0)
        mask = mask.values[-1].mul(255).byte().cpu().numpy()

        label = mask
        label[label > 0] = 255

        image_label_overlay = label2rgb(label, vis_img_prediction, bg_label=0)
        image_label_overlay = image_label_overlay * 255

        # save image
        dest = path_results_single_image_root / "Detected_objects_{}{}".format(
            path.stem, ".png"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_label_overlay.astype(np.uint8)).save(dest)

        img = Image.open(path).convert("RGB")
        img_size = img.size
        img = np.array(img)

        labels = np.array(prediction[0]["labels"].tolist())
        scores = np.array(prediction[0]["scores"].tolist())
        boxes = np.array(prediction[0]["boxes"].tolist())

        selection = np.logical_and(labels == 1, scores > th)
        b = boxes[selection]
        r = 512
        b = [
            [
                x_min * (img_size[0] / r),
                y_min * (img_size[1] / r),
                x_max * (img_size[0] / r),
                y_max * (img_size[1] / r),
            ]
            for x_min, y_min, x_max, y_max in b
        ]
        b = np.array(b).astype(np.int)

        row_column = []

        for bb in b:
            img_cropped = img[bb[1] : bb[3], bb[0] : bb[2], :]

            if zeiss_gray_background == True:
                img_cropped = standardize_gray_background(img_cropped)

            
            img_cropped = Image.fromarray(img_cropped)
            img_cropped.save(f"{path_results_single_image_root}/zeiss.png")
            top = bb[1]
            left = bb[0]

            img_name = "{}xx{}_{}{}".format(path.stem, left, top, path.suffix)

            center_y = (bb[1] + bb[3]) / 2
            center_x = (bb[0] + bb[2]) / 2

            row_column.append([img_name, center_x, center_y])

            dest = path_results_single_image_images / img_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            img_cropped.save(dest)

        # save row_column information
        pd.DataFrame(row_column, columns=["name", "x", "y"]).to_csv(
            str(path_results_single_image_root / "row_column.csv"), index=False
        )

        print("Brains Cutting Done")

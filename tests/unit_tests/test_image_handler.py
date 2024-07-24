import sys
from pathlib import Path
from PIL import Image
from math import sqrt
sys.path.append(str(Path(__file__).absolute().parents[2]))  # add the root project folder to the paths
from src.image_handler import HandleSingleSvs, HandlePng

single_svs_object = HandleSingleSvs(str(Path(__file__).absolute().parents[2]/"tests/samples/003 test_image.svs"))
single_png_object = HandlePng(str(Path(__file__).absolute().parents[2]/"tests/samples/test_image.png"))

def test_return_nr_of_layers():
    assert single_svs_object.return_nr_of_layers() == 1


def test_return_filename():
    assert single_svs_object.return_filename() == "003 test_image.svs"


def test_return_downsamples():
    assert single_svs_object.return_downsamples() == (1.0,)


def test_save_thumbnail():
    thumbnail_path = Path(__file__).absolute().parents[2]/"tests/samples/003 test_image_thumbnail.tif"
    if thumbnail_path.is_file():
        thumbnail_path.unlink()
    if thumbnail_path.is_file():
        assert False  # if file is not deleted, raise error
    single_svs_object.save_thumbnail(str(thumbnail_path)) # save thumbnail
    if thumbnail_path.is_file():
        assert True


def test_extract_xth_layer():
    test_img_path = Path(__file__).absolute().parents[2]/"tests/samples/003 test_image_thumbnail.tif"
    first_layer_of_img = Image.open(test_img_path).histogram()
    first_layer_extracted = single_svs_object.extract_xth_layer(0).histogram()
    # compare histograms of the images, if they are the same, images will be probably the same
    rms = sqrt(sum((a - b) ** 2 for a, b in zip(first_layer_of_img, first_layer_extracted)) / len(first_layer_of_img))
    assert rms == 0.0


def test_return_metadata():
    assert single_svs_object.return_metadata()['aperio.OriginalWidth'] == "46000"


def test_return_layer_resolution():
    assert single_svs_object.return_layer_resolution(0) == (2220, 2967)


def test_png():
    load_img = Image.open(str(Path(__file__).absolute().parents[2]/"tests/samples/test_image.png"))
    single_png_object.load_png()
    assert single_png_object.png_image == load_img


def test_png_size():
    assert single_png_object.return_max_resolution() == (512, 512)

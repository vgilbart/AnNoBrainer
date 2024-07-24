import sys
from pathlib import Path
sys.path.append(".")
from src.csv2haloxml_new import generate_xml

annotation_string = '<Annotations>' \
                    '<Annotation LineColor="65535" Name="Layer 1_Test_Notes" Visible="1">' \
                    '<Regions>' \
                    '<Region HasEndcaps="0" NegativeROA="1" Type="Polygon">' \
                    '<Vertices>' \
                    '<V X="154688" Y="123200" />' \
                    '<V X="154912" Y="123216" />' \
                    '<V X="155360" Y="122992" />' \
                    '<V X="155392" Y="122992" />' \
                    '<V X="156288" Y="120000" />' \
                    '<V X="156512" Y="120016" />' \
                    '<V X="156960" Y="119792" />' \
                    '<V X="156992" Y="119792" />' \
                    '</Vertices>' \
                    '</Region>' \
                    '</Regions>' \
                    '</Annotation>' \
                    '</Annotations>'


test_csv_folder = Path(".") / "tests" / "samples" / "annotations"
output_annotation_file = Path(".") / "tests" / "samples" / "annotations" / "annotation_file.annotations"


def test_generate_xml():
    generate_xml(test_csv_folder, output_annotation_file, "Test_Notes", 16)
    with open(output_annotation_file, "r") as annotation_file:
        output_annotations = annotation_file.read()
    Path.unlink(output_annotation_file)
    assert output_annotations == annotation_string

import xml.etree.ElementTree as ET
import csv
import os
import sys


def generate_xml(source_folder, output_file_path, notes="", scale=16):
    """
    vomit the annotation tree
    :param source_folder:
    :param output_file_path:
    :param scale:
    :param notes:
    :return:
    """

    annotation_tree = ET.Element("Annotations")  # Create xml tree
    if notes != "":
        notes = "_" + notes
    for csv_file in os.listdir(source_folder):
        k = 0
        if not csv_file.endswith("csv"):
            continue
        with open(os.path.join(source_folder, csv_file)) as csv_file_ptr:
            csv_reader = csv.reader(csv_file_ptr, delimiter=",")
            first_row = next(csv_reader)  # fetch info from the first row
            annotation = ET.SubElement(annotation_tree, "Annotation", LineColor=str(first_row[5]),
                                       Name=first_row[4] + "{0}".format(notes), Visible=first_row[6])
            regions = ET.SubElement(annotation, "Regions")
            csv_file_ptr.seek(0)  # get back to first row
            region = ET.SubElement(regions, "Region", Type="Polygon", HasEndcaps="0", NegativeROA=str(first_row[2]))
            vertices = ET.SubElement(region, "Vertices")
            for row in csv_reader:
                ET.SubElement(vertices, 'V', X=str(scale*int(round(float(row[0])))), Y=str(scale*int(round(float(row[1])))))
    with open(output_file_path, "wb") as output_file:
        output_file.write(ET.tostring(annotation_tree))


if __name__ == "__main__":
    if len(sys.argv) == 5:
        generate_xml(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:
        generate_xml(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        generate_xml(sys.argv[1], sys.argv[2])

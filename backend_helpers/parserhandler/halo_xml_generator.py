import xml.etree.ElementTree as ET
import csv
import os
import argparse


def generate_xml(csv_stream, csv_source):
    """
    vomit the annotation tree
    :param pos_files:
    :param neg_files:
    :param layer_name:
    :param layer:
    :param color:
    :return:
    """

    current_layer = [[], [], []]
    region_metadata = [[], [], []]
    scale = 16
    annotation_tree = ET.Element("Annotations")  # Create xml tree

    new_rows = [x for x in csv_stream]

    for idx, row in enumerate(csv_source):
        if current_layer != [row[4], row[5], row[6]]:
            current_layer = [row[4], row[5], row[6]]
            annotation = ET.SubElement(annotation_tree, "Annotation", LineColor=str(current_layer[1]), Name=current_layer[0], Visible=current_layer[2])
            regions = ET.SubElement(annotation, "Regions")
        #if region_metadata != [row[2], row[3]]:
            region_metadata = [row[2], row[3]]
            region = ET.SubElement(regions, "Region", Type="Polygon", HasEndcaps="0", NegativeROA=str(row[3]))
            vertices = ET.SubElement(region, "Vertices")
        ET.SubElement(vertices, 'V', X=str(scale*int(round(float(new_rows[idx][0])))), Y=str(scale*int(round(float(new_rows[idx][1])))))
    return annotation_tree


def parse_csv(path):
    out_path = os.path.abspath(path)
    source_path = os.path.abspath(path)
    os.chdir(out_path)
    files = os.listdir("./")
    print(files)
    for file in files:
        if file.endswith(".csv"):
            csv_io_stream = open(file, "r")
            file_source = open(os.path.join(source_path, file), "r")

            csv_row_stream = csv.reader(csv_io_stream, delimiter=",")
            csv_source_stream = csv.reader(file_source, delimiter=",")

            annot_tree = generate_xml(csv_row_stream, csv_source_stream)

            with open("halo_" + file + ".annotation", 'wb') as annot_file:
                annot_file.write(ET.tostring(annot_tree))
        else:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-study", "--studypath", required=True,
                            help="path to study folder (with annots)")
    args = vars(parser.parse_args())

    parse_csv(args['studypath'])



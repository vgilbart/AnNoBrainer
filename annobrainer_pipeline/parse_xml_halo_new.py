import xml.etree.ElementTree as EmTr
import csv
import os
import argparse


def parse_xml(halo_annotation_file):
    k = 0
    tree = EmTr.parse(halo_annotation_file)
    root = tree.getroot()
    csv_file = open(halo_annotation_file.split(".")[0] + ".csv", 'w', newline='')
    for_matlab_file = open(halo_annotation_file.split(".")[0] + "_matlab.csv", 'w', newline='')

    for annotation in root:
        layer_param_list = []
        if "Layer" not in annotation.attrib["Name"]:
            continue
        for metadata_key in annotation.attrib.keys():
            layer_param_list.append(annotation.attrib[metadata_key])
        for regions in annotation:
            for region in regions:
                particle_tag = str(0)
                if 'NegativeROA' in region.attrib.keys():
                    print(region.attrib['NegativeROA'])
                    if region.attrib['NegativeROA'] == '1':
                        particle_tag = str(1)

                csv_writer = csv.writer(csv_file, delimiter=",")
                matlab_writer = csv.writer(for_matlab_file, delimiter=",")
                for vertices in region:
                    for V in vertices:
                        csv_writer.writerow([V.attrib['X'], V.attrib['Y'], k, particle_tag] + layer_param_list)
                        matlab_writer.writerow([V.attrib['X'], V.attrib['Y']])
                k += 1
    csv_file.close()


def main(path):
    os.chdir(os.path.join(path, "Annots"))
    files = os.listdir("./")
    for file in files:
        if file[-12:] != ".annotations":
            continue
        parse_xml(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-study", "--studypath", required=True,
                        help="path to study folder (with annots)")
    args = vars(parser.parse_args())

    main(args['studypath'])
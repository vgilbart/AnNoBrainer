import sys
import unittest
import os
import base64
from pathlib import Path
sys.path.append(".")
from backend_helpers.halo_api_handler.ConnectSSH import ConnectUsingSSH
home_folder = Path(".")/"tests"


class TestSSHConnection(unittest.TestCase):
    def test_ssh_connector(self):
        with open(home_folder/"unit_tests"/"credentials_to_hpc.txt") as config_file:
            username = config_file.readline()
            pwd = base64.b64decode(config_file.readline())
        print(username, pwd)
        ssh_connector = ConnectUsingSSH(username.replace("\n", "").replace(" ",""), pwd)
        ssh_connector.upload_file(home_folder/"samples"/"annotation_file.annotations",
                                  "./img_reg_tests/annotation_file.annotations")
        [stderr_text, stdout_text, stdin_text] = ssh_connector.run_command("ls img_reg_tests")
        ssh_connector.download_file("./img_reg_tests/annotation_file.annotations",
                                    home_folder/"samples"/"sample_annotation_file2.annotations")
        [stderr_text_rem, stdout_text_rem, stdin_text_rem] = \
            ssh_connector.run_command("rm ./img_reg_tests/annotation_file.annotations")
        with open(home_folder/"samples"/"sample_annotation_file2.annotations") as annotation_file:
            annotation_string_downloaded = annotation_file.read()
        os.remove(home_folder/"samples"/"sample_annotation_file2.annotations")
        with open(home_folder/"samples"/"annotation_file.annotations") as annotation_file:
            annotation_string_original = annotation_file.read()
        self.assertEqual(annotation_string_downloaded, annotation_string_original)
        self.assertEqual([stderr_text, stdout_text, stdin_text], [b"", b"annotation_file.annotations\n", ""])
        self.assertEqual([stderr_text_rem, stdout_text_rem, stdin_text_rem],
                         [b"", b"", ""])

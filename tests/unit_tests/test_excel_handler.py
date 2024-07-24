import sys
sys.path.append(".")
from backend_helpers.excel_handler.excelhandler import HandleExcelFile
excel_file = HandleExcelFile("tests/samples/annoBrainer_DEMO2.xlsx")


def test_excel_sheet_names():
    excel_file.return_sheet_names()
    assert excel_file.sheet_names == ['EmptyTemplate', 'Study_ID_72', 'Study_ID_75']


def test_excel_sheet_dict():
    id_72 = excel_file.dump_sheet_to_dict("Study_ID_72", {})
    id_75 = excel_file.dump_sheet_to_dict("Study_ID_75", {})
    assert id_72['2121'] == {'rows': [[2121.0, 19.0, 1.0, 2.0, 6.0, 10.0, 14.0, 'blank'],
                                      [2121.0, 19.0, 2.0, 26.0, 30.0, 12.0, 16.0, 'blank'],
                                      [2121.0, 19.0, 3.0, 50.0, 54.0, 34.0, 38.0, 'blank'],
                                      [2121.0, 19.0, 4.0, 'blank', 'blank', 36.0, 40.0, 'blank'],
                                      [2121.0, 19.0, 5.0, 'blank', 60.0, 58.0, 62.0, 64.0]],
                            'notes': ['STR', 'STR', 'STR', 'STR', 'STR']}
    assert id_72['2162'] == {'rows': [[2162.0, 20.0, 1.0, 2.0, 6.0, 10.0, 14.0, 'blank'],
                                      [2162.0, 20.0, 2.0, 26.0, 30.0, 12.0, 16.0, 'blank'],
                                      [2162.0, 20.0, 3.0, 50.0, 54.0, 34.0, 38.0, 'blank'],
                                      [2162.0, 20.0, 4.0, 'blank', 'blank', 36.0, 40.0, 'blank'],
                                      [2162.0, 20.0, 5.0, 'blank', 60.0, 58.0, 62.0, 64.0]],
                             'notes': ['STR', 'STR', 'STR', 'STR', 'STR']}
    assert id_75['2165'] == {'rows': [[2165.0, 19.0, 1.0, 2.0, 6.0, 10.0, 14.0, 'blank'],
                                     [2165.0, 19.0, 2.0, 26.0, 30.0, 12.0, 16.0, 'blank'],
                                     [2165.0, 19.0, 3.0, 50.0, 54.0, 34.0, 38.0, 'blank'],
                                     [2165.0, 19.0, 4.0, 'blank', 'blank', 36.0, 40.0, 'blank'],
                                     [2165.0, 19.0, 5.0, 'blank', 60.0, 58.0, 62.0, 64.0]],
                            'notes': ['STR', 'STR', 'STR', 'STR', 'STR']}
    assert id_75['2168'] == {'rows': [[2168.0, 20.0, 1.0, 2.0, 6.0, 10.0, 14.0, 'blank'],
                                     [2168.0, 20.0, 2.0, 26.0, 30.0, 12.0, 16.0, 'blank'],
                                     [2168.0, 20.0, 3.0, 50.0, 54.0, 34.0, 38.0, 'blank'],
                                     [2168.0, 20.0, 4.0, 'blank', 'blank', 36.0, 40.0, 'blank'],
                                     [2168.0, 20.0, 5.0, 'blank', 60.0, 58.0, 62.0, 64.0]],
                            'notes': ['STR', 'STR', 'STR', 'STR', 'STR']}

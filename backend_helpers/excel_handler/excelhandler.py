import xlrd
import csv
import os
from backend_helpers.errorhandler.errorhandler import ACError as ACError
from backend_helpers.errorhandler.errorhandler import ExceptionType as ExceptionType
from backend_helpers.loggerhandler.logger_settings import logger

allowed_extensions = ["xls", "xlsb", "xlsx", "xlsm"]


class HandleExcelFile:
    def __init__(self, file_path_xls):
        try:
            self.workbook = xlrd.open_workbook(file_path_xls)
        except FileNotFoundError:
            logger.error("Excel file {0} not found".format(file_path_xls))
            raise ACError(ExceptionType.IO, "Excel file not found!")
        logger.info("Loaded excel file {0}".format(file_path_xls))
        self.sheet_names = []
        self.sheet = None

    def return_sheet_names(self):
        self.sheet_names = self.workbook.sheet_names()

    def dump_sheet_to_dict(self, sheet_name, excel_dict):
        try:
            sheet = self.workbook.sheet_by_name(sheet_name)
        except xlrd.biffh.XLRDError:
            logger.error("Excel sheet {0} not found".format(sheet_name))
            raise ACError(ExceptionType.IO, "Sheet not found!")
        actual_slide = "this_is_nonsense_name_for_slide"
        for row_num in range(1, sheet.nrows):
            try:
                current_slide = str(int(sheet.row_values(row_num)[1]))
            except ValueError:
                current_slide = sheet.row_values(row_num)[1]
            if current_slide != actual_slide:
                actual_slide = current_slide
                excel_dict[actual_slide] = {}
                excel_dict[actual_slide]['rows'] = []
                excel_dict[actual_slide]['notes'] = []
            current_row_values = sheet.row_values(row_num)
            for i in range(0,len(current_row_values)):
                try:
                    current_row_values[i] = int(current_row_values[i])
                except ValueError:
                    pass
            excel_dict[actual_slide]['rows'].append(current_row_values[1:])
            excel_dict[actual_slide]['notes'].append(current_row_values[0])
        return excel_dict


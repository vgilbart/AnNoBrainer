from enum import Enum
import sys
import os


class ExceptionType(Enum):
    ARGUMENTS = "arguments"
    IO = "R/W"
    CONNECTION = "connection"
    UNKNOWN = "unknown exception"
    HPC_JOB_ERROR ="Job failed"
    DB = "Database error"
    OS = "OS error"


class Error(Exception):
    pass


class ACError(Error):
    def __init__(self, exception_type, msg=None):
        self.exceptionType = exception_type
        self.msg = msg
        self.sms = []

    def emsg(self, arg=None):
        if self.msg:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.sms.append(str(fname))
            self.sms.append(str(', '))
            self.sms.append(str(exc_tb.tb_lineno))
            ret = 'AC ERROR: {0} > {1} > {2}'.format(self.exceptionType.value, self.msg, self.sms)
        else:
            ret = 'AC ERROR: {0} >  {1}'.format(self.exceptionType.value, self.sms)

        if arg is 'asString':
            return ret
        else:
            print(ret)

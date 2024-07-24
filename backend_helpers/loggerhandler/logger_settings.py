import logging
import sys


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[1;32;40m'
    WARNING = '\[\033[4;31m\]'
    FAIL = '\033[0;31;47m'
    ERROR = '\033[0;31;47m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# config logger
logging.basicConfig(filename="logs/application.log",
                    format='%(asctime)s %(message)s',
                    filemode='a+')
file_handler = logging.FileHandler("logs/application.log", mode='a+')

# create logger
logger = logging.getLogger('-')

if not logger.handlers:
    logger = logging.getLogger('-')
    logger.setLevel(logging.DEBUG)
    #logger.setLevel(logging.ERROR)
    ch = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(file_handler)
    logger.propagate = False

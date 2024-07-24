from backend_helpers.loggerhandler import logger_settings
import configparser as ConfigParser
import json
import os.path


def check_if_config_exists(config_file):
    try:
        if os.path.exists(config_file):
            logger_settings.logging.info('File {0} exists'.format(config_file))
            return True
        else:
            logger_settings.logging.info('File {0} does NOT exists'.format(config_file))
            return False
    except IOError as e:
        raise Exception("Problem accesing {0} -> {1} ".format(config_file, e))


def config_params(config_file):
    local_facts = dict()
    check_if_config_exists(config_file)
    try:
        # Handle conversion of INI style facts file to json style
        ini_facts = ConfigParser.ConfigParser()
        ini_facts.read(config_file)
        for section in ini_facts.sections():
            local_facts[section] = dict()
            for key, value in ini_facts.items(section):
                local_facts[section][key] = value

    except (ConfigParser.MissingSectionHeaderError, ConfigParser.ParsingError):
        try:
            with open(config_file, 'r') as facts_file:
                local_facts = json.load(facts_file)
        except (ValueError, IOError) as e:
            raise Exception("Problem accesing {0} -> {1} ".format(config_file, e))

    return local_facts

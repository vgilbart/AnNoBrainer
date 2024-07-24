from socket import timeout, gaierror
from backend_helpers.loggerhandler.logger_settings import logger
from backend_helpers.errorhandler.errorhandler import ACError, ExceptionType
from paramiko import ssh_exception


def ssh_exception_wrapper(wrapped_function):
    def _wrapper(*args, **kwargs):
        try:
            # do something before the function call
            result = wrapped_function(*args, **kwargs)
            # do something after the function call
        except gaierror as e:
            logger.error('Basic error SSH: {0}'.format(e))
            raise (ACError(ExceptionType.CONNECTION, 'SSH connection error: {0}'.format(e)))
        except ssh_exception.NoValidConnectionsError as e:
            logger.error('Valid Connections Error SSH: {0}'.format(e))
            raise (ACError(ExceptionType.CONNECTION, 'No valid connection error: {0}'.format(e)))
        except ssh_exception.AuthenticationException as e:
            logger.error('Authetincation Error SSH: {0}'.format(e))
            raise (ACError(ExceptionType.CONNECTION, 'Authentication error: {0}'.format(e)))
        except timeout as e:
            logger.error('Operation timed out: {0}'.format(e))
            raise (ACError(ExceptionType.CONNECTION, 'Operation timed out: {0}'.format(e)))
        except Exception as e:
            logger.error('Unknown exception: {0}'.format(e))
            raise (ACError(ExceptionType.CONNECTION, 'Unknown exception: {0}'.format(e)))
        return result
    return _wrapper

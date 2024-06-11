"""
logging.py -- Logging setup for CoolDwarf

This module contains the setup_logging function, which is used to set up the logging configuration for CoolDwarf.

Example usage
-------------
>>> from CoolDwarf.utils.misc.logging import setup_logging
>>> setup_logging(debug=True)
"""
import logging
import logging.config

def setup_logging(debug: bool = False):
    """
    This function is used to set up the logging configuration for CoolDwarf.

    Parameters
    ----------
    debug : bool, default=False
        If True, sets the logging level to DEBUG. Otherwise, sets the logging level to INFO.
    """
    if debug:
        ll = "DEBUG"
    else:
        ll = "INFO"
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 30,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': f'{ll}',
                'formatter': 'standard',
                'filename': 'CoolDwarf.log',
            },
        },
        'loggers': {
            'my_module': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
    }
    
    logging.config.dictConfig(logging_config)

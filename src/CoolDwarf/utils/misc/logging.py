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

# Define a log level for evolutionary steps
EVOLVE_LEVEL = 43
logging.addLevelName(EVOLVE_LEVEL, "EVOLVE")

def evolve_log(self, message, *args, **kwargs):
    if self.isEnabledFor(EVOLVE_LEVEL):
        self._log(EVOLVE_LEVEL, message, args, **kwargs)

logging.Logger.evolve = evolve_log

# Custom filter to include only logs with level 43
class CustomFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == EVOLVE_LEVEL

# Custom filter to exclude logs with level 43
class ExcludeCustomFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != EVOLVE_LEVEL

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
            'term': {
                'format': '%(levelname)s: %(message)s',
                }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 30,
                'formatter': 'term',
                'stream': 'ext://sys.stdout',
            },
            'file_all_except_custom': {
                'class': 'logging.FileHandler',
                'level': f'{ll}',
                'formatter': 'standard',
                'filename': 'CoolDwarf.log',
                'filters': ['exclude_custom'],
            },
            'file_custom': {
                'class': 'logging.FileHandler',
                'level': 'EVOLVE',
                'formatter': 'standard',
                'filename': 'CoolDwarf.evolve',
                'filters': ['custom_only'],
            },
        },
        'filters': {
            'custom_only': {
                '()': CustomFilter,
            },
            'exclude_custom': {
                '()': ExcludeCustomFilter,
            },
        },
        'loggers': {
            'my_module': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_all_except_custom', 'file_custom'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file_all_except_custom', 'file_custom'],
        },
    }

    logging.config.dictConfig(logging_config)
    

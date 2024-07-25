__version__ = "1.1.0"
from CoolDwarf.utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger('CoolDwarf')
logger.info("CoolDwarf Initialized")

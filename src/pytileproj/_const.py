import sys

DEFAULT_SEG_NUM = 25000  # number of segments for a geometry
DECIMALS = 9  # least significant digit after the command for rounding
TIMEOUT = 60  # Timeout for requests

VIS_INSTALLED = "matplotlib" in sys.modules and "cartopy" in sys.modules

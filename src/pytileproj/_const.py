from importlib.util import find_spec

DEF_SEG_LEN_DEG = 0.1  # number of segments for a lon lat geometry
DEF_SEG_LEN_M = 100  # number of segments for a projected geometry
DEF_SEG_NUM = 5_000  # number of segments for a projected geometry
DECIMALS = 9  # least significant digit after the command for rounding
TIMEOUT = 60  # timeout for requests
JSON_INDENT = 4  # indentation for JSON strings

VIS_INSTALLED = None not in [find_spec(pkg) for pkg in ["matplotlib", "cartopy"]]

GEO_INSTALLED = None not in [find_spec(pkg) for pkg in ["geopandas", "pyarrow"]]

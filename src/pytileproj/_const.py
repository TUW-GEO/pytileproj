import importlib

DEFAULT_SEG_NUM = 25000  # number of segments for a geometry
DECIMALS = 9  # least significant digit after the command for rounding
TIMEOUT = 60  # timeout for requests
JSON_INDENT = 4  # indentation for JSON strings

VIS_INSTALLED = None not in [
    importlib.util.find_spec(pkg) for pkg in ["matplotlib", "cartopy"]
]

GEO_INSTALLED = None not in [importlib.util.find_spec(pkg) for pkg in ["geopandas"]]

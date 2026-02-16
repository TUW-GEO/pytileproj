import sys
import warnings

import pytest

from pytileproj.projgeom import fetch_proj_zone


@pytest.fixture(autouse=True)
def vis_installed(request):  # noqa: ANN001
    if request.node.get_closest_marker("vis_installed"):
        vis_mods_installed = "cartopy" in sys.modules and "matplotlib" in sys.modules
        if not vis_mods_installed:
            pytest.skip()


@pytest.fixture(scope="package")
def epsg_api_accessible() -> bool:
    try:
        fetch_proj_zone(4326)
    except ConnectionError as e:
        warnings.warn(str(e), stacklevel=1)
        return False
    else:
        return True

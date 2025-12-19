import sys

import pytest


@pytest.fixture(autouse=True)
def vis_installed(request):  # noqa: ANN001
    if request.node.get_closest_marker("vis_installed"):
        vis_mods_installed = "cartopy" in sys.modules and "matplotlib" in sys.modules
        if not vis_mods_installed:
            pytest.skip()

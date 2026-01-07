# pytileproj

[![Build Status](https://github.com/TUW-GEO/pytileproj/workflows/ubuntu/badge.svg)](https://github.com/TUW-GEO/pytileproj/actions/workflows/ubuntu.yml) [![Build Status](https://github.com/TUW-GEO/pytileproj/workflows/windows/badge.svg)](https://github.com/TUW-GEO/pytileproj/actions/workflows/windows.yml) [![codecov](https://coveralls.io/repos/github/TUW-GEO/pytileproj/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/pytileproj?branch=master) [![pypi](https://badge.fury.io/py/pytileproj.svg)](https://badge.fury.io/py/pytileproj) [![docs](https://img.shields.io/badge/pytileproj-documentation-blue)](https://tuw-geo.github.io/pytileproj)


A python package for working with projected tiling systems.

Projected tiling systems define a tiling scheme for multiple levels (tiling or zoom levels) in a certain projection. The whole concept can be disentangled into the following components:

- *projection:* In `pytileproj`, a projection is represented via a CRS definition (EPSG, PROJ4, WKT, ...) and a projection zone defining the validity of coordinates (optional).
- *tiling:* A tiling is put on top of the projection to subdivide space into smaller units a.k.a. tiles. Tilings can be either regular or irregular:
  - *irregular:* Tiles can have arbitrary shapes and the overall tiling does not need to cover a certain extent. The only restriction is that tiles are not allowed to intersect.
  - *regular:* Tiles need to have the same shape and the regular tiling needs to fill a certain extent (no holes). The regular tiling follows the [OGC standard](https://docs.ogc.org/is/17-083r4/17-083r4.html).
- *tiling_system:* Multiple tilings covering the same extent are grouped into a tiling system.
- *grid:* Multiple tiling systems with the same tiling scheme but different projections are grouped into a grid. This allows to represent grid systems like the [Equi7Grid](https://github.com/TUW-GEO/Equi7Grid).

## How does `geospade` fit into the geospatial stack?

`pytileproj` heavily relies on [morecantile](https://github.com/developmentseed/morecantile) and extends its capabilities into a more generic framework for representing geospatial datacubes. As long as compliance is possible, `pytileproj` follows OGC standards, but it is not limited to web map representations of tiling systems, e.g. a quadtree.

## Installation

This package can be installed via pip:

```bash
pip install pytileproj
```

If you want to use `pytileproj`'s visualisation features, then you can install the required optional dependencies with:

```bash
pip install pytileproj[vis]
```

If you want to export tile tables and shapefiles, then you need to install the `geo` extension:

```bash
pip install pytileproj[geo]
```

## Contribute

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

### Development setup

For development you can either use a `conda/mamba` or `uv` environment. After that you should be able to run `uv run pytest` to run the test suite.

#### uv (recommended)
Here is an example using only `uv` for creating the environment and managing dependencies.

First, install `uv`:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Next, create your virtual environment, e.g.

```bash
uv venv --python 3.12
```

Finally, you can add all required and optional dependencies to it:

```bash
uv pip install -r pyproject.toml -e . --all-extras
```

#### mamba
Here is an example using `mamba` together with `uv` for managing dependencies.

First, install conda and set the path:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniforge.sh
bash miniforge.sh -b -p $HOME/miniforge
export PATH="$HOME/miniforge/bin:$PATH"
```

Next, create a virtual environment:

```bash
conda create -n pytileproj python=3.12 mamba
source activate pytileproj
mamba install -c conda-forge uv
```

Finally, use `uv` to install all other dependencies and `pytileproj` itself, e.g.:

```bash
uv pip install -r pyproject.toml -e . --all-extras
uv pip install -e . --no-deps
```


### Guidelines

If you want to contribute please follow these steps:

- fork the `pytileproj` repository to your account
- clone the repository
- make a new feature branch from the `pytileproj` master branch
- add your feature
- please include tests for your contributions in one of the test directories.
  We use `pytest` so a simple function called `test_my_feature` is enough
- submit a pull request to our master branch


## Citation

[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.1042555.svg)](https://doi.org/10.5281/zenodo.1042555)

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.1042555 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

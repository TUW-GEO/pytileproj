SHELL = /bin/bash
.PHONY: help clean install test version dist

help:
	@echo "make clean"
	@echo " clean all python build/compilation files and directories"
	@echo "make uv"
	@echo " install dependencies in local venv environment and creates a new one if it doesn't exist yet using uv"
	@echo "make test"
	@echo " run test with coverage"
	@echo "make version"
	@echo " update _version.py with current version tag"
	@echo "make dist"
	@echo " build the package ready for distribution and update the version tag"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive .pytest_cache
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

.venv/bin/activate:
	uv sync --group test --group trunk --no-install-package gdal && uv pip install gdal=="$(shell gdal-config --version).*"

uv: .venv/bin/activate

uv-local: uv
	uv pip install -r local-requirements.txt

uv-release: .venv/bin/activate
	uv sync --group test --no-sources --no-install-package gdal && uv pip install gdal=="$(shell gdal-config --version).*"

test:
	source .venv/bin/activate && pytest tests/ -rsx --verbose --color=yes --cov=pytileproj --cov-report term-missing

version:
	echo -e "__version__ = \"$(shell git describe --always --tags --abbrev=0)\"\n__commit__ = \"$(shell git rev-parse --short HEAD)\"" > src/pytileproj/_version.py

dist: version
	pip3 install build twine
	python3 -m build
from pathlib import Path

try:
    version_path = Path(__file__).parent.parent.parent / "VERSION.txt"
    with version_path.open() as f:
        version_content = [line.strip() for line in f.readlines()]
    __version__ = version_content[0]
    __commit__ = version_content[1]
except FileNotFoundError:
    __version__ = ""
    __commit__ = ""

name = "pytileproj"

__all__ = [
    "__commit__",
    "__version__",
]

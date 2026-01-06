from collections.abc import Generator, Mapping
from typing import Any, TypeAlias, TypeVar

from morecantile.commons import Tile as RegularTile

from pytileproj.tile import IrregularTile, RasterTile

T_co = TypeVar("T_co", covariant=True)
RT = TypeVar("RT", bound="RasterTile[Any]")
Extent: TypeAlias = tuple[float, float, float, float]
PartialExtent: TypeAlias = tuple[float, float, float | None, float | None]
AnyTile: TypeAlias = RegularTile | IrregularTile
TileGenerator: TypeAlias = Generator[AnyTile, None, None]
RegTileGenerator: TypeAlias = Generator[RegularTile, None, None]
IrregTileGenerator: TypeAlias = Generator[IrregularTile, None, None]
RasterTileGenerator: TypeAlias = Generator[RasterTile[T_co], None, None]
SamplingFloatOrMap: TypeAlias = float | Mapping[int, float]

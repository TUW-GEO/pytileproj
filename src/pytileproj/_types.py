from collections.abc import Generator
from typing import TypeVar

from morecantile.models import Tile as RegularTile

from pytileproj.tile import IrregularTile, RasterTile

T_co = TypeVar("T_co", covariant=True)
Extent = tuple[int | float, int | float, int | float, int | float]
PartialExtent = tuple[int | float, int | float, int | float | None, int | float | None]
AnyTile = RegularTile | IrregularTile
TileGenerator = Generator[AnyTile, AnyTile, AnyTile]
RegTileGenerator = Generator[RegularTile, RegularTile, RegularTile]
IrregTileGenerator = Generator[IrregularTile, IrregularTile, IrregularTile]
RasterTileGenerator = Generator[RasterTile[T_co], RasterTile[T_co], RasterTile[T_co]]

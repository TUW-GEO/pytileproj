from collections.abc import Generator

from morecantile.models import Tile as RegularTile

from pytileproj.tile import IrregularTile, RasterTile

Extent = tuple[int | float, int | float, int | float, int | float]
PartialExtent = tuple[int | float, int | float, int | float | None, int | float | None]
AnyTile = RegularTile | IrregularTile
TileGenerator = Generator[AnyTile, AnyTile, AnyTile]
RegTileGenerator = Generator[RegularTile, RegularTile, RegularTile]
IrregTileGenerator = Generator[IrregularTile, IrregularTile, IrregularTile]
RasterTileGenerator = Generator[RasterTile, RasterTile, RasterTile]
